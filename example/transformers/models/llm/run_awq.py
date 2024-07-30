#
# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
#

import argparse
import builtins
import gc
import logging
import os
import time

import llm_eval
import llm_profile
import psutil
import torch
from llm_eval import (
    AutoModelEval,
    BloomModelEval,
    GPTBigCodeModelEval,
    LlamaModelEval,
    MistralModelEval,
    OPTModelEval,
    Phi2ModelEval,
    Qwen2ModelEval,
)
from ryzenai_llm_engine import RyzenAILLMEngine, TransformConfig
from ryzenai_llm_quantizer import QuantConfig, RyzenAILLMQuantizer

from transformers import (
    AutoModel,
    AutoTokenizer,
    LlamaTokenizer,
    PreTrainedTokenizerFast,
    set_seed,
)

supported_models = [
    "facebook/opt-125m",
    "facebook/opt-1.3b",
    "facebook/opt-2.7b",
    "facebook/opt-6.7b",
    "facebook/opt-13b",
    "llama-2-7b",
    "llama-2-7b-chat",
    "llama-2-13b",
    "llama-2-13b-chat",
    "codellama/CodeLlama-7b-hf",
    "codellama/CodeLlama-7b-instruct-hf",
    "code-llama-2-7b",
    "bigcode/starcoder",
    "google/gemma-2b",
    "google/gemma-7b",
    "THUDM/chatglm-6b",
    "THUDM/chatglm3-6b",
    "Qwen/Qwen-7b",
    "Qwen/Qwen1.5-7B",
    "Qwen/Qwen1.5-7B-Chat",
    "microsoft/phi-2",
    "mistralai/Mistral-7B-v0.1",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "state-spaces/mamba-1.4b-hf",
    "state-spaces/mamba-2.8b-hf",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama/Meta-Llama-3-8B",
    "microsoft/Phi-3-mini-4k-instruct",
]

if __name__ == "__main__":

    import transformers

    tv = transformers.__version__
    tv = tv.split(".")

    set_seed(123)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        help="mode name",
        type=str,
        default="facebook/opt-125m",
        choices=supported_models,
    )
    parser.add_argument(
        "--target",
        help="cpu, aie, npugpu",
        type=str,
        default="aie",
        choices=["cpu", "aie", "npugpu"],
    )
    parser.add_argument(
        "--profile_layer",
        help="layer profile",
        type=bool,
        default=False,
        choices=[False, True],
    )
    parser.add_argument(
        "--task",
        help="infershapes: shape inference; quantize:quantize the model; decode: Decode set of prompts; benchmark: Benchmark latency w.r.t prompt length; benchmark_long: Benchmark long sequences (compare with flash attn); perplexity: Measure perplexity on wikitext2 dataset; countgops: profile gops of the model for given workload;",
        type=str,
        default="decode",
        choices=[
            "infershapes",
            "quantize",
            "decode",
            "benchmark",
            "benchmark_long",
            "countgops",
            "perplexity",
            "benchmark_long_output",
            "profilemodel",
            "mmlu",
            "humaneval",
            "chat",
        ],
    )
    parser.add_argument(
        "--precision",
        help="w4abf16 - used for awq, awqplus, pergrp & bf16 runs on cpu",
        type=str,
        default="w4abf16",
        choices=["bf16", "w4abf16"],
    )
    parser.add_argument(
        "--flash_attention_plus",
        help="enable flash attn and other optimizations",
        action="store_true",
    )
    parser.add_argument(
        "--profilegemm",
        help="Log matmul times for prompt and token phases - supported only for AIE target",
        action="store_true",
    )
    parser.add_argument(
        "--dataset",
        help="Dataset - wikitext2-raw-v1, wikitext2-v1",
        type=str,
        default="raw",
        choices=["non-raw", "raw"],
    )
    parser.add_argument("--fast_mlp", help="enable fast mlp", action="store_true")
    parser.add_argument(
        "--fast_attention", help="enable fast attention", action="store_true"
    )
    parser.add_argument("--w_bit", help="3, 4", type=int, default=4, choices=[3, 4])
    parser.add_argument(
        "--group_size", help="128 default", type=int, default=128, choices=[32, 64, 128]
    )
    parser.add_argument(
        "--algorithm",
        help="awq, awqplus, pergrp",
        type=str,
        default="awq",
        choices=["awq", "awqplus", "pergrp"],
    )

    parser.add_argument(
        "--gen_onnx_nodes",
        help="generate onnx nodes for npu-gpu hybrid mode",
        action="store_true",
    )

    parser.add_argument(
        "--mhaops",
        help="enable ops in mha",
        type=str,
        default="all",
        choices=["bmm1", "softmax", "bmm2", "all", "pytorchmha", "libtorchflat"],
    )

    args = parser.parse_args()
    print(f"{args}")

    if int(tv[1]) >= 39:
        from gemma.modeling_gemma_amd import GemmaForCausalLM
        from gemma.tokenization_gemma_amd import GemmaTokenizer
        from llm_eval import MambaModelEval
    else:
        if ("gemma" in args.model_name) or ("mamba" in args.model_name):
            print(
                f"\n\n[run_awq] Gemma-2b/7b and Mamba models are ONLY supported with transformers==4.39.x\n Upgrade to transformers==4.39.1 and rerun\n\n"
            )
            raise SystemExit

    dev = os.getenv("DEVICE")

    trust_remote_code = False
    if "opt" in args.model_name:
        CausalLMModel = OPTModelEval
    elif ("llama" in args.model_name) or ("Llama" in args.model_name):
        CausalLMModel = LlamaModelEval
    elif "bloom" in args.model_name:
        CausalLMModel = BloomModelEval
    elif "starcoder" in args.model_name:
        CausalLMModel = GPTBigCodeModelEval
    elif "Qwen1.5" in args.model_name:
        CausalLMModel = Qwen2ModelEval
    elif "Qwen" in args.model_name:
        from qwen7b.modeling_qwen_amd import GenerationConfig, QWenLMHeadModel

        CausalLMModel = QWenLMHeadModel
        trust_remote_code = True
    elif "chatglm" in args.model_name:
        if "chatglm3" in args.model_name:
            from modeling_chatglm3_amd import ChatGLMForConditionalGeneration
            from tokenization_chatglm3 import ChatGLMTokenizer
        else:
            from modeling_chatglm_amd import ChatGLMForConditionalGeneration
            from tokenization_chatglm_amd import ChatGLMTokenizer
        trust_remote_code = True
    elif "Phi-3" in args.model_name:
        from llm_eval import Phi3ModelEval

        CausalLMModel = Phi3ModelEval
        trust_remote_code = True
    elif "phi" in args.model_name:
        CausalLMModel = Phi2ModelEval
    elif "mistral" in args.model_name:
        CausalLMModel = MistralModelEval
    elif "mamba" in args.model_name:
        CausalLMModel = MambaModelEval
    else:
        CausalLMModel = AutoModelEval

    if "llama-2" in args.model_name:
        LMTokenizer = LlamaTokenizer
    elif "Llama-3" in args.model_name:
        LMTokenizer = PreTrainedTokenizerFast
    else:
        LMTokenizer = AutoTokenizer

    log_dir = "./logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = log_dir + "/log_%s.log" % (args.model_name.replace("/", "_"))
    logging.basicConfig(filename=log_file, filemode="w", level=logging.CRITICAL)

    model_short_name = (
        args.model_name.replace("state-spaces/", "")
        .replace("facebook/", "")
        .replace("meta-llama/", "")
        .replace("bigscience/", "")
        .replace("bigcode/", "")
        .replace("codellama/", "")
        .replace("google/", "")
        .replace("THUDM/", "")
        .replace("Qwen/", "")
        .replace("microsoft/", "")
        .replace("mistralai/", "")
        .replace("TinyLlama/", "")
    )
    qmodels_dir = "./quantized_models/"
    if not os.path.exists(qmodels_dir):
        os.makedirs(qmodels_dir)
    ckpt = qmodels_dir + "/quantized_%s_w%d_g%d_%s.pth" % (
        model_short_name,
        args.w_bit,
        args.group_size,
        args.algorithm,
    )

    ############################################################################################
    ### Step 1 - Model Quantization
    ### Step 2 - Model Transformation & Optimization
    ### Step 3 - Inference
    ############################################################################################

    if args.task == "quantize":
        if not os.path.exists(ckpt):
            if "Qwen1.5" in args.model_name:
                model = CausalLMModel.from_pretrained(
                    args.model_name,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=trust_remote_code,
                    attn_implementation="eager",
                )
                model.tokenizer = LMTokenizer.from_pretrained(
                    args.model_name, trust_remote_code=trust_remote_code
                )
            elif "Qwen" in args.model_name:
                tokenizer = LMTokenizer.from_pretrained(
                    args.model_name,
                    pad_token="<|extra_0|>",
                    eos_token="<|endoftext|>",
                    padding_side="left",
                    trust_remote_code=True,
                )
                model = QWenLMHeadModel.from_pretrained(
                    args.model_name,
                    pad_token_id=tokenizer.pad_token_id,
                    device_map="cpu",
                    trust_remote_code=True,
                    attn_implementation="eager",
                )
                model.tokenizer = tokenizer
                model.generation_config = GenerationConfig.from_pretrained(
                    args.model_name,
                    pad_token_id=tokenizer.pad_token_id,
                    trust_remote_code=True,
                )
                model.model_name = model_short_name
            elif "chatglm" in args.model_name:
                model = (
                    ChatGLMForConditionalGeneration.from_pretrained(
                        args.model_name,
                        trust_remote_code=trust_remote_code,
                        attn_implementation="eager",
                    )
                    .to(torch.bfloat16)
                    .to("cpu")
                )
                model.tokenizer = ChatGLMTokenizer.from_pretrained(
                    args.model_name, trust_remote_code=trust_remote_code
                )
            elif "gemma" in args.model_name:
                model = (
                    GemmaForCausalLM.from_pretrained(
                        args.model_name,
                        trust_remote_code=trust_remote_code,
                        attn_implementation="eager",
                    )
                    .to(torch.bfloat16)
                    .to("cpu")
                )
                model.tokenizer = GemmaTokenizer.from_pretrained(
                    args.model_name, trust_remote_code=trust_remote_code
                )
            else:
                model = CausalLMModel.from_pretrained(
                    args.model_name,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=trust_remote_code,
                    attn_implementation="eager",
                )
                model.tokenizer = LMTokenizer.from_pretrained(
                    args.model_name, trust_remote_code=trust_remote_code
                )
            model.model_name = model_short_name
            print(model)
            # set use_scales = False in quant config to calculate new awq scales
            use_qscales = True
            quant_config = QuantConfig(
                quant_mode=args.algorithm,
                model_name=model.model_name,
                dataset="raw",
                w_bit=args.w_bit,
                group_size=args.group_size,
                use_qscales=use_qscales,
            )

            ##############################################################
            ### Step 1 - Model Quantization
            model = RyzenAILLMQuantizer.quantize(model, quant_config=quant_config)
            print(model)
            ##############################################################

            torch.save(model, ckpt)
            print(f"\n\nSaved Quantized Model ... : {ckpt} !!! \n")
        else:
            print(f"\n\nFound quantized Model on disk : {ckpt} - nothing to do\n")

    else:
        if args.precision == "bf16":
            if "Qwen1.5" in args.model_name:
                model = CausalLMModel.from_pretrained(
                    args.model_name,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=trust_remote_code,
                    attn_implementation="eager",
                )
                model.tokenizer = LMTokenizer.from_pretrained(
                    args.model_name, trust_remote_code=trust_remote_code
                )
            elif "Qwen" in args.model_name:
                model = QWenLMHeadModel.from_pretrained(
                    args.model_name,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=trust_remote_code,
                    attn_implementation="eager",
                )
                model.tokenizer = LMTokenizer.from_pretrained(
                    args.model_name, trust_remote_code=trust_remote_code
                )
            elif model_short_name == "chatglm-6b" or model_short_name == "chatglm3-6b":
                model = (
                    ChatGLMForConditionalGeneration.from_pretrained(
                        args.model_name,
                        trust_remote_code=trust_remote_code,
                        attn_implementation="eager",
                    )
                    .to(torch.bfloat16)
                    .to("cpu")
                )
                model.tokenizer = ChatGLMTokenizer.from_pretrained(
                    args.model_name, trust_remote_code=trust_remote_code
                )
            elif "gemma" in model_short_name:
                model = (
                    GemmaForCausalLM.from_pretrained(
                        args.model_name,
                        trust_remote_code=trust_remote_code,
                        attn_implementation="eager",
                    )
                    .to(torch.bfloat16)
                    .to("cpu")
                )
                model.tokenizer = GemmaTokenizer.from_pretrained(
                    args.model_name, trust_remote_code=trust_remote_code
                )
            else:
                model = CausalLMModel.from_pretrained(
                    args.model_name,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=trust_remote_code,
                    attn_implementation="eager",
                )
                model.tokenizer = LMTokenizer.from_pretrained(
                    args.model_name, trust_remote_code=trust_remote_code
                )
            model.model_name = model_short_name
            print(f"\n\nLoaded bf16 Model {model.model_name} ... !! \n")
            # print(model)
            # print(model.config._attn_implementation)

        else:
            if not os.path.exists(ckpt):
                print(f"\n\nQuantized Model not available ... {ckpt} !!! \n")
                print(
                    f"\n\nRun with --task quantize and generate quantized model first \n"
                )
                raise SystemExit
            if "Qwen1.5" in args.model_name:
                tokenizer = LMTokenizer.from_pretrained(args.model_name)
                model = CausalLMModel.from_pretrained(args.model_name, device_map="cpu")
                model.tokenizer = tokenizer
                model.model_name = model_short_name
            elif "Qwen" in args.model_name:
                tokenizer = LMTokenizer.from_pretrained(
                    args.model_name,
                    pad_token="<|extra_0|>",
                    eos_token="<|endoftext|>",
                    padding_side="left",
                    trust_remote_code=True,
                )
                model = QWenLMHeadModel.from_pretrained(
                    args.model_name,
                    pad_token_id=tokenizer.pad_token_id,
                    device_map="cpu",
                    trust_remote_code=True,
                    attn_implementation="eager",
                )
                model.tokenizer = tokenizer
                model.generation_config = GenerationConfig.from_pretrained(
                    args.model_name,
                    pad_token_id=tokenizer.pad_token_id,
                    trust_remote_code=True,
                )
                model.model_name = model_short_name
            elif "chatglm" in args.model_name:
                model = ChatGLMForConditionalGeneration.from_pretrained(
                    args.model_name,
                    trust_remote_code=trust_remote_code,
                    attn_implementation="eager",
                ).to("cpu")
                model.tokenizer = ChatGLMTokenizer.from_pretrained(
                    args.model_name, trust_remote_code=trust_remote_code
                )
                model.model_name = model_short_name
            model = torch.load(ckpt)
            print(model)
            print(f"model.model_name: {model.model_name}")

        ##############################################################
        ### Step 2 - Model Transformation & Optimization
        transform_config = TransformConfig(
            flash_attention_plus=args.flash_attention_plus,
            fast_mlp=args.fast_mlp,
            fast_attention=args.fast_attention,
            precision=args.precision,
            model_name=args.model_name,
            target=args.target,
            w_bit=args.w_bit,
            group_size=args.group_size,
            profilegemm=args.profilegemm,
            profile_layer=args.profile_layer,
            mhaops=args.mhaops,
        )

        model = RyzenAILLMEngine.transform(model, transform_config)
        model = model.to(torch.bfloat16)
        model.eval()
        print(model)
        print(f"model.mode_name: {model.model_name}")
        ##############################################################

        if args.gen_onnx_nodes:
            print(f"[RyzenAI] Generating onnx nodes ...")
            import qlinear

            onnx_weights_dir = model.model_name + "_onnx_params_fp16"
            if not os.path.exists(onnx_weights_dir):
                os.makedirs(onnx_weights)
            for n, m in model.named_modules():
                if "up_proj" in n:
                    if isinstance(m, qlinear.QLinearPerGrp):
                        newm = torch.nn.Linear(
                            in_features=m.weight.data.shape[0],
                            out_features=m.weight.data.shape[1],
                            bias=False,
                        )
                        newm.weight.data = (
                            m.weight.data.to(torch.float16).transpose(0, 1).clone()
                        )
                        x = torch.rand(1, newm.in_features).to(torch.float16)
                        torch.onnx.export(
                            newm,  # model being run
                            x,  # model input (or a tuple for multiple inputs)
                            onnx_weights_dir
                            + "\\"
                            + n
                            + ".onnx",  # where to save the model (can be a file or file-like object)
                            export_params=True,  # store the trained parameter weights inside the model file
                            opset_version=10,  # the ONNX version to export the model to
                            do_constant_folding=False,  # whether to execute constant folding for optimization
                            input_names=["input"],  # the model's input names
                            output_names=["output"],  # the model's output names
                        )
                        print(
                            f"[RyzenAILLMEngine] Saved onnx model of layer {n} for token phase in {onnx_weights_dir}"
                        )

        ##############################################################
        ### Step 3 - Inference
        apply_chat_tmpl = (
            True
            if (
                ("Qwen1.5-7B-Chat" in args.model_name)
                or ("Qwen1.5-7B" in args.model_name)
                or ("TinyLlama" in args.model_name)
                or ("Phi-3" in args.model_name)
            )
            else False
        )

        if args.task == "infershapes":
            if args.target != "cpu":
                print(f"\n\n *** Set --target to CPU to infer shapes *** exiting ... ")
            else:
                llm_eval.infer_linear_shapes(model, apply_chat_tmpl=apply_chat_tmpl)

        elif args.task == "decode":
            llm_eval.decode_prompts(
                model, log_file, apply_chat_tmpl=apply_chat_tmpl, max_new_tokens=60
            )

        elif (args.task == "benchmark") or (args.task == "benchmark_long"):
            llm_eval.benchmark(
                model,
                args.dataset,
                args.task,
                log_file,
                apply_chat_tmpl=apply_chat_tmpl,
            )

        # WIP - for SinkCache
        elif args.task == "benchmark_long_output":
            llm_eval.benchmark_ag(
                model, log_file, max_seqlen=4096, assistant_model=None, do_sample=False
            )

        elif args.task == "countgops":
            llm_eval.count_gops(model)

        elif args.task == "perplexity":
            start = time.time()
            llm_eval.perplexity(model, dataset=args.dataset)
            print(f"Time taken to measure ppl on RyzenAI: {time.time() - start}s")

        elif args.task == "mmlu":
            start = time.time()
            llm_eval.mmlu(model)
            print(f"Time taken to measure mmlu on RyzenAI: {time.time() - start}s")

        elif args.task == "chat":
            while True:
                print("-" * 20)
                prompt = input("\nprompt: ")
                if prompt == "exit":
                    break
                else:
                    llm_eval.decode_prompt(
                        model,
                        model.tokenizer,
                        prompt,
                        max_new_tokens=200,
                        assistant_model=None,
                        apply_chat_tmpl=apply_chat_tmpl,
                    )
                # llm_eval.decode_prompts(model, log_file, apply_chat_tmpl=False, promptnow=[prompt], max_new_tokens=100)

        elif args.task == "humaneval":
            start = time.time()
            import llm_human_eval

            human_eval_path = (
                os.getenv("PYTORCH_AIE_PATH")
                + "\\tools\\humaneval-sub\\sixty_acc_dataset.json"
            )
            llm_human_eval.run_human_eval(model, human_eval_path)
            print(f"Time taken to measure humaneval on RyzenAI: {time.time() - start}s")

        elif args.task == "profilemodel":
            llm_profile.TorchModuleProfile.register_profile_hooks(model)
            if "code" in model.model_name.lower():
                prompt = "def fibonacci_recursive(n):"
            else:
                prompt = "Ancient Egypt was a civilization of ancient Northeast Africa. It was concentrated along the lower reaches of the Nile River, situated in the place that is now the country Egypt. Ancient Egyptian civilization followed prehistoric Egypt and coalesced around 3100 BC (according to conventional Egyptian chronology)[1] with the political unification of Upper and Lower Egypt under Menes (often identified with Narmer).[2] The history of ancient Egypt unfolded as a series of stable kingdoms interspersed by periods of relative instability known as “Intermediate Periods.” The various kingdoms fall into one of three categories: the Old Kingdom of the Early Bronze Age, the Middle Kingdom of the Middle Bronze Age, or the New Kingdom of the Late Bronze Age.Ancient Egypt reached the pinnacle of its power during the New Kingdom, ruling much of Nubia and a sizable portion of the Levant. After this period, it entered an era of slow decline. During the course of its history, Ancient Egypt was invaded or conquered by a number of foreign powers, including the Hyksos, the Nubians, the Assyrians, the Achaemenid Persians, and the Macedonians under Alexander the Great. The Greek Ptolemaic Kingdom, formed in the aftermath of Alexander's death, ruled until 30 BC, when, under Cleopatra, it fell to the Roman Empire and became a Roman province.[3] Egypt remained under Roman control until the 640s AD, when it was conquered by the Rashidun Caliphate.The success of ancient Egyptian civilization came partly from its ability to adapt to the conditions of the Nile River valley for agriculture. The predictable flooding and controlled irrigation of the fertile valley produced surplus crops, which supported a more dense population, and social development and culture. With resources to spare, the administration sponsored mineral exploitation of the valley and surrounding desert regions, the early development of an independent writing system, the organization of collective construction and agricultural projects, trade with surrounding regions, and a military intended to assert Egyptian dominance. Motivating and organizing these activities was a bureaucracy of elite scribes, religious leaders, and administrators under the control of a pharaoh, who ensured the cooperation and unity of the Egyptian people in the context of an elaborate system of religious beliefs.[4]The many achievements of the ancient Egyptians include the quarrying, surveying, and construction techniques that supported the building of monumental pyramids, temples, and obelisks; a system of mathematics, a practical and effective system of medicine, irrigation systems, and agricultural production techniques, the first known planked boats,[5] Egyptian faience and glass technology, new forms of literature, and the earliest known peace treaty, made with the Hittites.[6] Ancient Egypt has left a lasting legacy. Its art and architecture were widely copied, and its antiquities were carried off to far corners of the world. Its monumental ruins have inspired the imaginations of travelers and writers for millennia. A newfound respect for antiquities and excavations in the early modern period by Europeans and Egyptians has led to the scientific investigation of Egyptian civilization and a greater appreciation of its cultural legacy.[7]Ancient Egypt was a civilization of ancient Northeast Africa. It was concentrated along the lower reaches of the Nile River, situated in the place that is now the country Egypt. Ancient Egyptian civilization followed prehistoric Egypt and coalesced around 3100 BC (according to conventional Egyptian chronology)[1] with the political unification of Upper and Lower Egypt under Menes (often identified with Narmer).[2] The history of ancient Egypt unfolded as a series of stable kingdoms interspersed by periods of relative instability known as “Intermediate Periods.” The various kingdoms fall into one of three categories: the Old Kingdom of the Early Bronze Age, the Middle Kingdom of the Middle Bronze Age, or the New Kingdom of the Late Bronze Age.Ancient Egypt reached the pinnacle of its power during the New Kingdom, ruling much of Nubia and a sizable portion of the Levant. After this period, it entered an era of slow decline. During the course of its history, Ancient Egypt was invaded or conquered by a number of foreign powers, including the Hyksos, the Nubians, the Assyrians, the Achaemenid Persians, and the Macedonians under Alexander the Great. The Greek Ptolemaic Kingdom, formed in the aftermath of Alexander's death, ruled until 30 BC, when, under Cleopatra, it fell to the Roman Empire and became a Roman province.[3] Egypt remained under Roman control until the 640s AD, when it was conquered by the Rashidun Caliphate.The success of ancient Egyptian civilization came partly from its ability to adapt to the conditions of the Nile River valley for agriculture. The predictable flooding and controlled irrigation of the fertile valley produced surplus crops, which supported a more dense population, and social development and culture. With resources to spare, the administration sponsored mineral exploitation of the valley and surrounding desert regions, the early development of an independent writing system, the organization of collective construction and agricultural projects, trade with surrounding regions, and a military intended to assert Egyptian dominance. Motivating and organizing these activities was a bureaucracy of elite scribes, religious leaders, and administrators under the control of a pharaoh, who ensured the cooperation and unity of the Egyptian people in the context of an elaborate system of religious beliefs.[4]The many achievements of the ancient Egyptians include the quarrying, surveying, and construction techniques that supported the building of monumental pyramids, temples, and obelisks; a system of mathematics, a practical and effective system of medicine, irrigation systems, and agricultural production techniques, the first known planked boats,[5] Egyptian faience and glass technology, new forms of literature, and the earliest known peace treaty, made with the Hittites.[6] Ancient Egypt has left a lasting legacy. Its art and architecture were widely copied, and its antiquities were carried off to far corners of the world. Its monumental ruins have inspired the imaginations of travelers and writers for millennia. A newfound respect for antiquities and excavations in the early modern period by Europeans and Egyptians has led to the scientific investigation of Egyptian civilization and a greater appreciation of its cultural legacy.[7]Ancient Egypt was a civilization of ancient Northeast Africa. It was concentrated along the lower reaches of the Nile River, situated in the place that is now the country Egypt. Ancient Egyptian civilization followed prehistoric Egypt and coalesced around 3100 BC (according to conventional Egyptian chronology)[1] with the political unification of Upper and Lower Egypt under Menes (often identified with Narmer).[2] The history of ancient Egypt unfolded as a series of stable kingdoms interspersed by periods of relative instability known as “Intermediate Periods.” The various kingdoms fall into one of three categories: the Old Kingdom of the Early Bronze Age, the Middle Kingdom of the Middle Bronze Age, or the New Kingdom of the Late Bronze Age.Ancient Egypt reached the pinnacle of its power during the New Kingdom, ruling much of Nubia and a sizable portion of the Levant. After this period, it entered an era of slow decline. During the course of its history, Ancient Egypt was invaded or conquered by a number of foreign powers, including the Hyksos, the Nubians, the Assyrians, the Achaemenid Persians, and the Macedonians under Alexander the Great. The Greek Ptolemaic Kingdom, formed in the aftermath of Alexander's death, ruled until 30 BC, when, under Cleopatra, it fell to the Roman Empire and became a Roman province.[3] Egypt remained under Roman control until the 640s AD, when it was conquered by the Rashidun Caliphate.The success of ancient Egyptian civilization came partly from its ability to adapt to the conditions of the Nile River valley for agriculture. The predictable flooding and controlled irrigation of the fertile valley produced surplus crops, which supported a more dense population, and social development and culture. With resources to spare, the administration sponsored mineral exploitation of the valley and surrounding desert regions, the early development of an independent writing system, the organization of collective construction and agricultural projects, trade with surrounding regions, and a military intended to assert Egyptian dominance. Motivating and organizing these activities was a bureaucracy of elite scribes, religious leaders, and administrators under the control of a pharaoh, who ensured the cooperation and unity of the Egyptian people in the context of an elaborate system of religious beliefs.[4]The many achievements of the ancient Egyptians include the quarrying, surveying, and construction techniques that supported the building and the structure underneath it a"
            llm_eval.decode_prompt(model, model.tokenizer, prompt, max_new_tokens=11)
            llm_profile.TorchModuleProfile.generate_report(model)
            logging.shutdown()
            out_file = log_file.replace(".log", "_profile.csv")
            out_file = open(out_file, "w")
            llm_profile.ProfileLLM.analyze_profiling(log_file, out_file)
            out_file.close()

# For NPU GPU
# python run_awq.py --model_name llama-2-7b --target cpu --precision w4abf16 --gen_onnx_nodes
# python run_awq.py --model_name llama-2-7b --target npugpu
