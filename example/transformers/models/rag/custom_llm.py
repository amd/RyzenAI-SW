#
# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
#

from transformers import AutoTokenizer, OPTForCausalLM
import torch
import os
import gc
from typing import Any, Sequence
from llama_index.core.llms import CustomLLM, CompletionResponse, LLMMetadata, CompletionResponseGen
from llama_index.core.llms.callbacks import llm_completion_callback
import logging 
from transformers import AutoTokenizer, LlamaTokenizer, PreTrainedTokenizerFast, set_seed, TextStreamer
from ryzenai_llm_engine import RyzenAILLMEngine, TransformConfig
from llm_eval import OPTModelEval, LlamaModelEval, AutoModelEval
import time 

set_seed(12)

class OurLLM(CustomLLM):
    model_name: str = None
    tokenizer: Any = None
    model: Any = None
    quantized: bool = None
    w_bit: int = None
    target: str = None
    algorithm: str = None 
    group_size: str = None
    flash_attention_plus: str = None
    fast_attention: str = None
    fast_mlp: str = None
    precision: Any = None
    profilegemm: str = None
    assisted_generation: str = None
    assistant_model: Any = None
    

    def __init__(self, target=None, model_name=None, quantized=None, algorithm=None, group_size=None, w_bit=None, flash_attention_plus=None, fast_mlp=None, fast_attention=None, precision=None, profilegemm=None, assisted_generation=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_name = model_name
        self.quantized = quantized
        self.w_bit = w_bit
        self.target = target
        self.algorithm = algorithm
        self.group_size = group_size
        self.flash_attention_plus = flash_attention_plus
        self.fast_mlp = fast_mlp
        self.fast_attention = fast_attention
        self.precision = precision
        self.profilegemm = profilegemm
        self.assisted_generation = assisted_generation
        

        trust_remote_code= False
        if "opt" in self.model_name:         
            CausalLMModel = OPTModelEval
        elif ("llama" in self.model_name) or ("Llama" in self.model_name):
            CausalLMModel = LlamaModelEval
        else:                                
            CausalLMModel = AutoModelEval


        if "llama-2" in self.model_name:       
            LMTokenizer = LlamaTokenizer
        elif "Llama-3" in self.model_name:      
            LMTokenizer = PreTrainedTokenizerFast
        else:                                
            LMTokenizer = AutoTokenizer
        
        self.tokenizer = LMTokenizer.from_pretrained(self.model_name, trust_remote_code=trust_remote_code)
        
        log_dir = "./logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        log_file = log_dir + "/log_%s.log" % (self.model_name.replace("/", "_"))
        logging.basicConfig(filename=log_file, filemode="w", level=logging.CRITICAL)

        if self.quantized:
            model_short_name = (
                self.model_name.replace("facebook/", "")
                .replace("meta-llama/", "")
            )
            
            qmodels_dir = "./quantized_models"
            if not os.path.exists(qmodels_dir):
                os.makedirs(qmodels_dir)
            ckpt = qmodels_dir + "/quantized_%s_w%d_g%d_%s.pth" % (
                model_short_name,
                self.w_bit,
                self.group_size,
                self.algorithm,
            )
            
            if not os.path.exists(ckpt):
                print(f"\n\nQuantized Model not available ... {ckpt} !!! \n")
                print(f"\n\n[load_awq_model] quantize using ..\\llm\\run_awq.py --task quantize and generate quantized model first, and then run rag ... exiting ...\n")
                raise SystemExit

            self.model = torch.load(ckpt)
        else:
            if (self.model_name == "llama-2-7b-chat") or (self.model_name == "llama-2-13b-chat") or (self.model_name == "llama-2-7b") or (self.model_name == "llama-2-13b"):
                self.model = LlamaModelEval.from_pretrained(self.model_name)
                self.tokenizer = LlamaTokenizer.from_pretrained(self.model_name)
            else:
                self.model = CausalLMModel.from_pretrained("facebook/" + self.model_name)
        ##############################################################
        ### Assistant Model
        if self.assisted_generation:
            from transformers import AutoModelForCausalLM

            if "opt" in self.model_name:
                assistant_model = AutoModelForCausalLM.from_pretrained(
                    "facebook/opt-125m", torch_dtype=torch.bfloat16
                )
            else:
                assistant_model = AutoModelForCausalLM.from_pretrained(
                    "JackFram/llama-160m",
                    torch_dtype=torch.bfloat16,
                ) # llama

            assistant_model = assistant_model.to(torch.bfloat16)
            print(f"[load_models] assistant model loaded ...")
            print(assistant_model)
        else:
            assistant_model=None
        
        
        ##############################################################
        ### Step 2 - Model Transformation & Optimization
        transform_config = TransformConfig(
            flash_attention_plus=self.flash_attention_plus, 
            fast_mlp=False, 
            fast_attention=self.fast_attention,
            precision=self.precision, 
            model_name=self.model_name, 
            target=self.target, 
            w_bit=self.w_bit, 
            group_size=self.group_size, 
            profilegemm = False,
            profile_layer = False,
            mhaops = "all",
        ) 
        
        self.model = RyzenAILLMEngine.transform(self.model, transform_config)
        self.model = self.model.to(torch.bfloat16)
        self.model.eval()
        print(self.model)
        print(f"model_name: {self.model_name}")
        print(f"[load_smoothquant_model] model loaded ...")
        
        
    def decode_prompt1(self, prompt, input_ids=None, max_new_tokens=None, do_sample=False, temperature=None, assistant_model=assistant_model) ->str:
        if input_ids is None:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            input_ids_ = inputs.input_ids
        else:
            input_ids_ = input_ids
        
        logging.critical(f"[PROFILE] tokenizer:")
        attention_mask = torch.ones(input_ids_.shape)
        streamer = TextStreamer(self.tokenizer)
        if (self.model_name == "llama-2-7b-chat") or (self.model_name == "llama-2-13b-chat") or (self.model_name == "llama-2-7b") or (self.model_name == "llama-2-13b"):
            do_sample = True
            temperature = 0.1
            gen_begin = time.time()
            start = time.perf_counter()
            generate_ids = self.model.generate(
                input_ids=input_ids_, 
                attention_mask=attention_mask, 
                max_new_tokens=max_new_tokens, 
                streamer=streamer, 
                assistant_model=assistant_model, 
                do_sample=True, 
                temperature=0.1, 
                top_p=0.95, 
                pad_token_id=self.model.tokenizer.eos_token_id,
            )
            end = time.perf_counter()
            gen_end = time.time()
            gen_time = gen_end - gen_begin
            print(f"runtime of generate is {gen_time}") 
        
        elif self.model_name == "Meta-Llama-3-8B-Instruct":
            do_sample = True
            temperature = 0.1
            
            start = time.perf_counter()
            generate_ids = self.model.generate(
                input_ids=input_ids_,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                streamer=streamer,
                assistant_model=assistant_model,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
                pad_token_id=self.model.tokenizer.eos_token_id,
            )    
            end = time.perf_counter()
        else:
            start = time.perf_counter()
            generate_ids = self.model.generate(
                input_ids=input_ids_, 
                attention_mask=attention_mask, 
                max_new_tokens=max_new_tokens, 
                streamer=streamer, 
                assistant_model=assistant_model, 
                do_sample=True, 
                temperature=temperature, 
                pad_token_id=self.tokenizer.eos_token_id
            )
            end = time.perf_counter()
        
        generate_time = end - start
        prompt_tokens = input_ids_.shape[1]
        num_tokens_out = generate_ids.shape[1]
        new_tokens_generated = num_tokens_out - prompt_tokens
        time_per_token = (generate_time / new_tokens_generated) * 1e3
        response = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        # logging.critical(f"[PROFILE] generate: {generate_time} for {num_tokens_out} tokens; prompt-tokens: {prompt_tokens}; time per generated token: {time_per_token}")
        # logging.critical(f"response: {response}")
        
        len_of_prompt = len(prompt)
        # print(f"length of prompt: {len_of_prompt}")
        # print(f"num of input tokens: {len(input_ids_[0])}")
        
        len_of_resp = len(response)
        # print(f"length of response: {len_of_resp}")
        # print(f"num of tokens generated: {len(generate_ids[0])}")
        return response


    def generate_response(self, prompt, max_new_tokens=120):
        if self.quantized:
            do_sample=False
            temperature=None
            m = max_new_tokens
            logging.critical("*"*40)
            print("*"*40)
            resp = self.decode_prompt1(prompt, max_new_tokens=m, do_sample=do_sample, temperature=temperature)
            logging.shutdown()
        else:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            m = max_new_tokens
            # Generate
            gen_begin = time.time()
            generate_ids = self.model.generate(inputs.input_ids, max_new_tokens=m)#max_length=30
            gen_end = time.time()
            print(f"runtime of generate1 is {gen_end - gen_begin}") 
            dec_begin = time.time()
            resp = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            dec_end = time.time()
            print(f"runtime of decode1 is {dec_end - dec_begin}") 
        return resp

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            model_name=self.model_name,
        )


    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        response = self.generate_response(prompt)
        return CompletionResponse(text=response)
    
    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        pass

# cleanup
gc.collect()

