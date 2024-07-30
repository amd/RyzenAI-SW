#
# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
#

import gzip
import json
import logging
import os
import random
import time
from collections import defaultdict
from typing import Dict, Iterable

import numpy as np
import stats
import torch
import tqdm
from human_eval.data import write_jsonl
from llm_profile import ProfileLLM
from perf_estimator import OpsProfile
from utils import Utils

import transformers
from transformers import (
    AutoModelForCausalLM,
    BloomForCausalLM,
    LlamaForCausalLM,
    MistralForCausalLM,
    OPTForCausalLM,
    PhiForCausalLM,
    Qwen2ForCausalLM,
    SinkCache,
    TextStreamer,
)
from transformers.models.gpt_bigcode.modeling_gpt_bigcode import GPTBigCodeForCausalLM

tv = transformers.__version__
tv = tv.split(".")

if int(tv[1]) >= 39:
    from modeling_mamba import MambaForCausalLM

    class MambaModelEval(MambaForCausalLM):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.tokenizer = None
            self.model_name = None

        def forward(self, *args, **kwargs):
            st = time.perf_counter()
            outputs = super().forward(*args, **kwargs)
            en = time.perf_counter()
            logging.critical(f"[PROFILE] model_decoder_forward {en-st}")
            return outputs


class OPTModelEval(OPTForCausalLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = None
        self.model_name = None

    def forward(self, *args, **kwargs):
        st = time.perf_counter()
        outputs = super().forward(*args, **kwargs)
        en = time.perf_counter()
        logging.critical(f"[PROFILE] model_decoder_forward {en-st}")
        return outputs


class LlamaModelEval(LlamaForCausalLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = None
        self.model_name = None

    def forward(self, *args, **kwargs):
        st = time.perf_counter()
        outputs = super().forward(*args, **kwargs)
        en = time.perf_counter()
        logging.critical(f"[PROFILE] model_decoder_forward {en-st}")
        return outputs


class BloomModelEval(BloomForCausalLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = None
        self.model_name = None

    def forward(self, *args, **kwargs):
        st = time.perf_counter()
        outputs = super().forward(*args, **kwargs)
        en = time.perf_counter()
        logging.critical(f"[PROFILE] model_decoder_forward {en-st}")
        return outputs


class GPTBigCodeModelEval(GPTBigCodeForCausalLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = None
        self.model_name = None

    def forward(self, *args, **kwargs):
        st = time.perf_counter()
        outputs = super().forward(*args, **kwargs)
        en = time.perf_counter()
        logging.critical(f"[PROFILE] model_decoder_forward {en-st}")
        return outputs


class Phi2ModelEval(PhiForCausalLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = None
        self.model_name = None

    def forward(self, *args, **kwargs):
        st = time.perf_counter()
        outputs = super().forward(*args, **kwargs)
        en = time.perf_counter()
        logging.critical(f"[PROFILE] model_decoder_forward {en-st}")
        return outputs


if int(tv[1]) >= 39:
    from modeling_phi3 import Phi3ForCausalLM

    class Phi3ModelEval(Phi3ForCausalLM):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.tokenizer = None
            self.model_name = None

        def forward(self, *args, **kwargs):
            st = time.perf_counter()
            outputs = super().forward(*args, **kwargs)
            en = time.perf_counter()
            logging.critical(f"[PROFILE] model_decoder_forward {en-st}")
            return outputs


class MistralModelEval(MistralForCausalLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = None
        self.model_name = None

    def forward(self, *args, **kwargs):
        st = time.perf_counter()
        outputs = super().forward(*args, **kwargs)
        en = time.perf_counter()
        logging.critical(f"[PROFILE] model_decoder_forward {en-st}")
        return outputs


class Qwen2ModelEval(Qwen2ForCausalLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = None
        self.model_name = None

    def forward(self, *args, **kwargs):
        st = time.perf_counter()
        outputs = super().forward(*args, **kwargs)
        en = time.perf_counter()
        logging.critical(f"[PROFILE] model_decoder_forward {en-st}")
        return outputs


class AutoModelEval(AutoModelForCausalLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = None
        self.model_name = None

    def forward(self, *args, **kwargs):
        st = time.perf_counter()
        outputs = super().forward(*args, **kwargs)
        en = time.perf_counter()
        logging.critical(f"[PROFILE] model_decoder_forward {en-st}")
        return outputs


class LlamaForCausalLMPadded(LlamaForCausalLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = None
        self.model_name = None

    def forward(self, *args, **kwargs):
        outputs = super().forward(*args, **kwargs)

        codellama_extra_token_number = 16
        outputs["logits"] = torch.nn.functional.pad(
            outputs["logits"], (0, codellama_extra_token_number), value=float("-inf")
        )

        return outputs


prompts = [
    "What is the meaning of life?",
    "Tell me something you don't know.",
    "What does Xilinx do?",
    "What is the mass of earth?",
    "What is a poem?",
    "What is recursion?",
    "Tell me a one line joke.",
    "Who is Gilgamesh?",
    "Tell me something about cryptocurrency.",
    "How did it all begin?",
]


prompts_code = [
    "# python code to complete some task. # Create a function to calculate the sum of a sequence of integers. [PYTHON]\ndef sum_sequence(sequence):\n  sum = 0\n  for num in sequence:\n    sum += num\n  return sum \n$$\n# Implement merge sort algorithm.",
    "# python code to complete some task. # Create a function to calculate the sum of a sequence of integers. [PYTHON]\ndef sum_sequence(sequence):\n  sum = 0\n  for num in sequence:\n    sum += num\n  return sum \n$$\n# Create a function named max_num() that takes a list of numbers named nums as a parameter. The function should return the largest number in nums.",
    "# python code to complete some task. # Create a function to calculate the sum of a sequence of integers. [PYTHON]\ndef sum_sequence(sequence):\n  sum = 0\n  for num in sequence:\n    sum += num\n  return sum \n$$\n# '写一个函数，找到两个字符序列的最长公共子序列。",
    'from typing import List, Any\n\n\ndef filter_integers(values: List[Any]) -> List[int]:\n    """ Filter given list of any python values only for integers\n    >>> filter_integers(["a", 3.14, 5])\n    [5]\n    >>> filter_integers([1, 2, 3, "abc", {}, []])\n    [1, 2, 3]\n    """\n',
    "from typing import List\n\n\ndef concatenate(strings: List[str]) -> str:\n    \"\"\" Concatenate list of strings into a single string\n    >>> concatenate([])\n    ''\n    >>> concatenate(['a', 'b', 'c'])\n    'abc'\n    \"\"\"\n",
    'def max_element(l: list):\n    """Return maximum element in the list.\n    >>> max_element([1, 2, 3])\n    3\n    >>> max_element([5, 3, -5, 2, -3, 3, 9, 0, 123, 1, -10])\n    123\n    """\n',
    'from typing import List, Any\n\n\ndef filter_integers(values: List[Any]) -> List[int]:\n    """ Filter given list of any python values only for integers\n    >>> filter_integers([\'a\', 3.14, 5])\n    [5]\n    >>> filter_integers([1, 2, 3, \'abc\', {}, []])\n    [1, 2, 3]\n    """\n',
    "def double_the_difference(lst):\n    '''\n    Given a list of numbers, return the sum of squares of the numbers\n    in the list that are odd. Ignore numbers that are negative or not integers.\n    \n    double_the_difference([1, 3, 2, 0]) == 1 + 9 + 0 + 0 = 10\n    double_the_difference([-1, -2, 0]) == 0\n    double_the_difference([9, -2]) == 81\n    double_the_difference([0]) == 0  \n   \n    If the input list is empty, return 0.\n    '''\n",
    "# python code to complete some task. # Create a function to calculate the sum of a sequence of integers. [PYTHON]\ndef sum_sequence(sequence):\n  sum = 0\n  for num in sequence:\n    sum += num\n  return sum \n[/PYTHON]\n# Implement quick sort algorithm.",
    "# python code to complete some task. # Create a function to calculate the sum of a sequence of integers. [PYTHON]\ndef sum_sequence(sequence):\n  sum = 0\n  for num in sequence:\n    sum += num\n  return sum \n[/PYTHON]\n# Create a Python list comprehension to get the squared values of a list [1, 2, 3, 5, 8, 13].",
    "# python code to complete some task. # Create a function to calculate the sum of a sequence of integers. [PYTHON]\ndef sum_sequence(sequence):\n  sum = 0\n  for num in sequence:\n    sum += num\n  return sum \n[/PYTHON]\n# Create a Python function that takes in a string and a list of words and returns true if the string contains all the words in the list.",
    "# python code to complete some task. # Create a function to calculate the sum of a sequence of integers. [PYTHON]\ndef sum_sequence(sequence):\n  sum = 0\n  for num in sequence:\n    sum += num\n  return sum \n[/PYTHON]\n# Create a program to print out the top 3 most frequent words in a given text.",
    "# python code to complete some task. # Create a function to calculate the sum of a sequence of integers. [PYTHON]\ndef sum_sequence(sequence):\n  sum = 0\n  for num in sequence:\n    sum += num\n  return sum \n[/PYTHON]\n# Produce a function that takes two strings, takes the string with the longest length and swaps it with the other.",
]


prompts_chinese = [
    "生命的意义是什么？",
    "告诉我一些你不知道的事情。",
    "Xilinx是做什么的？",
    "地球的质量是多少？",
    "诗歌是什么？",  ##### issue found with this question, qwen1.5 does not reply to this question.
    "递归是什么？",
    "讲一个一句话的笑话.",
    "谁是吉尔伽美什？",
    "告诉我一些关于比特币的事情。",
    "这一切是怎么发生的？",
]


def get_wikitext2(tokenizer, dataset="non-raw", nsamples=82, seqlen=4096):
    """gptq"""
    from datasets import load_dataset

    if dataset == "non-raw":
        traindata = load_dataset("wikitext", "wikitext-2-v1", split="train")
        testdata = load_dataset("wikitext", "wikitext-2-v1", split="test")
    elif dataset == "raw":
        traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    else:
        raise ValueError(
            "You are using an unsupported dataset, only support wikitext2-raw-v1 and wikitext2-v1."
            "Using wikitext2-raw-v1 with --dataset=raw and wikitext2-v1 with --dataset=non-raw."
        )

    trainenc = tokenizer("\n".join(traindata["text"]), return_tensors="pt")
    testenc = tokenizer("\n".join(testdata["text"]), return_tensors="pt")
    dataloader = []
    for _ in range(nsamples):
        i = random.randint(0, testenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = testenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        dataloader.append((inp, tar))
    return dataloader, testenc


def eval_nonuniform(model, input_ids):
    """from gptq"""
    cache = {"past": None}

    def clear_past(i):
        def tmp(layer, inp, out):
            if cache["past"]:
                cache["past"][i] = None

        return tmp

    for i, layer in enumerate(model.model.decoder.layers):
        layer.register_forward_hook(clear_past(i))
    tot = 0.0
    with torch.no_grad():
        attention_mask = torch.ones((1, input_ids.numel()))
        times = []
        for i in range(input_ids.numel()):
            tick = time.time()
            out = model(
                input_ids[:, i].reshape((1, -1)),
                past_key_values=cache["past"],
                attention_mask=attention_mask[:, : (i + 1)].reshape((1, -1)),
            )
            times.append(time.time() - tick)
            tot += times[-1]
            cache["past"] = list(out.past_key_values)
            del out
        import numpy as np
    return np.median(times), tot


def perplexity(model, dataset, framework="pytorch", device="cpu"):
    if "opt" in model.model_name:
        max_seqlen = 2048
    else:
        max_seqlen = 4096
    random.seed(0)
    np.random.seed(0)
    torch.random.manual_seed(0)
    print(f"Calculating Perplexity on wikitext2 test set ...")
    model = model.to(device)
    dataloader, testenc = get_wikitext2(model.tokenizer, dataset=dataset)
    test_enc = testenc.input_ids
    nsamples = 2  # test_enc.numel() // max_seqlen
    if framework == "pytorch":
        dtype = next(iter(model.parameters())).dtype

    cache = {"past": None}

    def clear_past(i):
        def tmp(layer, inp, out):
            if cache["past"]:
                cache["past"][i] = None

        return tmp

    if framework == "pytorch":
        layers = None
        if model.__class__.__name__ == "LlamaModelEval":
            layers = model.model.layers
        elif "qwen2" in str(model.__class__).lower():
            layers = model.model.layers
        elif "qwen" in str(model.__class__).lower():
            layers = model.transformer.h
        elif "chatglm" in str(model.__class__).lower():
            layers = model.transformer.encoder.layers
        elif model.__class__.__name__ == "OPTModelEval":
            layers = model.model.decoder.layers
        elif isinstance(model, BloomModelEval):
            layers = model.transformer.h
        elif "mpt" in str(model.__class__).lower():
            layers = model.transformer.blocks
        elif "falcon" in str(model.__class__).lower():
            layers = model.transformer.h
        elif "bigcode" in str(model.__class__).lower():
            layers = model.transformer.h
        elif "neox" in str(model.__class__).lower():
            layers = model.gpt_neox.layers
        else:
            raise NotImplementedError(type(model))

        for i, layer in enumerate(layers):
            layer.register_forward_hook(clear_past(i))

    loss = torch.nn.CrossEntropyLoss()
    nlls = []

    with torch.no_grad():
        attention_mask = torch.ones((1, test_enc.numel())).to(device)
        for i in range(nsamples):
            batch = test_enc[:, (i * max_seqlen) : ((i + 1) * max_seqlen)].to(device)
            if framework == "pytorch":
                out = model(
                    batch,
                    attention_mask=attention_mask[
                        :, (i * max_seqlen) : ((i + 1) * max_seqlen)
                    ].reshape((1, -1)),
                )
            else:
                out = model(batch, attention_mask=batch.new_ones(batch.shape))
            shift_labels = test_enc[:, (i * max_seqlen) : ((i + 1) * max_seqlen)][
                :, 1:
            ].to(device)
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(out.logits[0][:-1, :], shift_labels.view(-1))
            neg_log_likelihood = loss.float() * max_seqlen
            nlls.append(neg_log_likelihood)
            ppl = torch.exp(torch.stack(nlls).sum() / ((i + 1) * max_seqlen))
            print(f"Samples:{i+1} of {nsamples}: Perplexity:{ppl.item()}")

        ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * max_seqlen))
        print(f"Final Perplexity:{ppl.item()}")


def mmlu(model, max_new_tokens=50, nsamples=30):
    import os
    import re
    from typing import List

    import pandas as pd
    from thefuzz import process
    from tqdm import tqdm

    from transformers.trainer_utils import set_seed

    # set dataset path
    eval_data_path = "./mmlu_data"
    tokenizer = model.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    TASK_NAME_MAPPING = {
        "stem": [
            "abstract_algebra",
            #     "anatomy",
            #     "astronomy",
            #     "college_biology",
            #     "college_chemistry",
            #     "college_computer_science",
            #     "college_mathematics",
            #     "college_physics",
            #     "computer_security",
            #     "conceptual_physics",
            #     "electrical_engineering",
            #     "elementary_mathematics",
            #     "high_school_biology",
            #     "high_school_chemistry",
            #     "high_school_computer_science",
            #     "high_school_mathematics",
            #     "high_school_physics",
            #     "high_school_statistics",
            #     "machine_learning",
        ],
        # "Humanities": [
        #     "formal_logic",
        #     "high_school_european_history",
        #     "high_school_us_history",
        #     "high_school_world_history",
        #     "international_law",
        #     "jurisprudence",
        #     "logical_fallacies",
        #     "moral_disputes",
        #     "moral_scenarios",
        #     "philosophy",
        #     "prehistory",
        #     "professional_law",
        #     "world_religions",
        # ],
        # "other": [
        #     "business_ethics",
        #     "college_medicine",
        #     "human_aging",
        #     "management",
        #     "marketing",
        #     "medical_genetics",
        #     "miscellaneous",
        #     "nutrition",
        #     "professional_accounting",
        #     "professional_medicine",
        #     "virology",
        #     "global_facts",
        #     "clinical_knowledge",
        # ],
        # "social": [
        # "econometrics",
        # "high_school_geography",
        # "high_school_government_and_politics",
        # "high_school_macroeconomics",
        # "high_school_microeconomics",
        # "high_school_psychology",
        # "human_sexuality",
        # "professional_psychology",
        # "public_relations",
        # "security_studies",
        # "sociology",
        # "us_foreign_policy",
        # ],
    }

    SUBJECTS = [v for vl in TASK_NAME_MAPPING.values() for v in vl]
    choices = ["A", "B", "C", "D"]

    def format_example(line, include_answer=True):
        example = "Question: " + line["question"]
        for choice in choices:
            example += f'\n{choice}. {line[f"{choice}"]}'

        if include_answer:
            example += "\nAnswer: " + line["answer"] + "\n\n"
        else:
            example += "\nAnswer:"
        return example

    def format_example_chat(line):
        example = (
            "The following is a multiple-choice question. Please choose the most suitable one among A, B, C and D as the answer to this question.\n\n"
            + line["question"]
            + "\n"
        )
        for choice in choices:
            example += f'{choice}. {line[f"{choice}"]}\n'
        return example

    def generate_few_shot_prompt(k, subject, dev_df):
        def format_subject(subject):
            l = subject.split("_")
            s = ""
            for entry in l:
                s += " " + entry
            return s.strip()

        prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
            format_subject(subject)
        )

        if k == -1:
            k = dev_df.shape[0]
        for i in range(k):
            prompt += format_example(
                dev_df.iloc[i, :],
                include_answer=True,
            )
        return prompt

    def get_logits(tokenizer, model, inputs: List[str]):
        input_ids = tokenizer(inputs, padding="longest")["input_ids"]
        input_ids = torch.tensor(input_ids, device=model.device)
        max_seq_len = 4096

        if input_ids.shape[1] > max_seq_len:
            input_ids = input_ids[:, input_ids.shape[1] - max_seq_len + 1 :]
        tokens = {"input_ids": input_ids}
        attention_mask = input_ids.ne(tokenizer.pad_token_id)

        outputs = model(input_ids, attention_mask=attention_mask)["logits"]
        logits = outputs[:, -1, :]
        log_probs = torch.nn.functional.softmax(logits, dim=-1)
        return log_probs, {"tokens": tokens}

    def process_before_extraction(gen, choice_dict):
        # replace the choice by letter in the generated sentence
        # from longest one to shortest one
        for key, val in sorted(
            choice_dict.items(), key=lambda x: len(x[1]), reverse=True
        ):
            pattern = re.compile(re.escape(val.rstrip(".")), re.IGNORECASE)
            gen = pattern.sub(key, gen)
        return gen

    def extract_choice(gen, choice_list):
        # answer is A | choice is A | choose A
        res = re.search(
            r"(?:(?:[Cc]hoose)|(?:(?:[Aa]nswer|[Cc]hoice)(?![^ABCD]{0,20}?(?:n't|not))[^ABCD]{0,10}?\b(?:|is|:|be))\b)[^ABCD]{0,20}?\b(A|B|C|D)\b",
            gen,
        )

        # A is correct | A is right
        if res is None:
            res = re.search(
                r"\b(A|B|C|D)\b(?![^ABCD]{0,8}?(?:n't|not)[^ABCD]{0,5}?(?:correct|right))[^ABCD]{0,10}?\b(?:correct|right)\b",
                gen,
            )

        # straight answer: A
        if res is None:
            res = re.search(r"^(A|B|C|D)(?:\.|,|:|$)", gen)

        # simply extract the first appearred letter
        if res is None:
            res = re.search(r"(?<![a-zA-Z])(A|B|C|D)(?![a-zA-Z=])", gen)

        if res is None:
            return choices[choice_list.index(process.extractOne(gen, choice_list)[0])]
        return res.group(1)

    def extract_answer(response, row):
        gen = process_before_extraction(
            response, {choice: row[choice] for choice in choices}
        )
        pred = extract_choice(gen, [row[choice] for choice in choices])
        return pred

    @torch.no_grad()
    def eval_subject(
        model,
        tokenizer,
        subject_name,
        test_df,
        k=5,
        dev_df=None,
        few_shot=False,
        save_result_dir=None,
        batch_size=1,
        **kwargs,
    ):
        result = []
        score = []
        save_response = []

        if "Chat" in model.model_name or "chat" in model.model_name:
            for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
                question = format_example_chat(row)
                message = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": question},
                ]
                text = tokenizer.apply_chat_template(
                    message, tokenize=False, add_generation_prompt=True
                )
                model_inputs = tokenizer([text], return_tensors="pt")
                input_ids_ = model_inputs.input_ids
                attention_mask = torch.ones(input_ids_.shape)
                # start = time.perf_counter()
                streamer = None  # TextStreamer(tokenizer)
                generate_ids = model.generate(
                    input_ids=input_ids_,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    streamer=streamer,
                )
                # end = time.perf_counter()
                response = tokenizer.batch_decode(
                    generate_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                # response, _ = model.chat(
                #    tokenizer,
                #    question,
                #    history=None,
                # )
                # print(message)
                # print("response:",response)
                pred = extract_answer(response[0], row)
                save_response.append(response)
                # print("Final Answer:",pred)
                # print("======================")

                if "answer" in row:
                    correct = 1 if pred == row["answer"] else 0
                    score.append(correct)
                result.append(pred)
        else:
            few_shot_prompt = (
                generate_few_shot_prompt(k, subject_name, dev_df) if few_shot else []
            )
            all_probs = {"prob_A": [], "prob_B": [], "prob_C": [], "prob_D": []}
            # if args.debug:
            #     print(f"few_shot_prompt: {few_shot_prompt}")
            if "llama" in model.model_name:
                choices_ids = (
                    torch.tensor(
                        [tokenizer(" A")["input_ids"][2]]
                        + [tokenizer(" B")["input_ids"][2]]
                        + [tokenizer(" C")["input_ids"][2]]
                        + [tokenizer(" D")["input_ids"][2]]
                    )
                    .unsqueeze(0)
                    .to(model.device)
                )
            else:
                choices_ids = (
                    torch.tensor(
                        tokenizer(" A")["input_ids"]
                        + tokenizer(" B")["input_ids"]
                        + tokenizer(" C")["input_ids"]
                        + tokenizer(" D")["input_ids"]
                    )
                    .unsqueeze(0)
                    .to(model.device)
                )

            idx_list = list(range(0, len(test_df), batch_size))
            for i in tqdm(idx_list):
                full_prompt_list = []
                answer_list = []
                for row in test_df.iloc[i : i + batch_size].to_dict(orient="records"):
                    question = format_example(row, include_answer=False)
                    full_prompt = few_shot_prompt + question
                    full_prompt_list.append(full_prompt)
                    if "answer" in row:
                        answer_list.append(row["answer"])

                logits, input_info = get_logits(tokenizer, model, full_prompt_list)
                softval = logits.gather(
                    1, choices_ids.expand(logits.size(0), -1)
                ).softmax(1)
                if softval.dtype in {torch.bfloat16, torch.float16}:
                    softval = softval.to(dtype=torch.float32)
                probs = softval.detach().cpu().numpy()
                for i in range(len(probs)):
                    for j, choice in enumerate(choices):
                        all_probs[f"prob_{choice}"].append(probs[i][j])
                    pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(probs[i])]

                    if answer_list != []:
                        correct = 1 if pred == answer_list[i] else 0
                        score.append(correct)
                        # if args.debug:
                        #     print(f'{question} pred: {pred} ref: {answer_list[i]}')
                    result.append(pred)

        if save_result_dir:
            test_df["model_output"] = result
            if "Chat" in model.model_name or "chat" in model.model_name:
                test_df["model_response"] = save_response
            else:
                for i, choice in enumerate(choices):
                    test_df[f"prob_{choice}"] = all_probs[f"prob_{choice}"]
            if score:
                test_df["correctness"] = score
            os.makedirs(save_result_dir, exist_ok=True)
            test_df.to_csv(
                os.path.join(save_result_dir, f"{subject_name}_result.csv"),
                encoding="utf-8",
                index=False,
            )

        return score

    def cal_mmlu(res):
        acc_sum_dict = dict()
        acc_norm_sum_dict = dict()
        cnt_dict = dict()
        acc_sum = 0.0
        cnt = 0
        hard_cnt = 0
        hard_acc_sum = 0.0

        for class_ in TASK_NAME_MAPPING.keys():
            acc_sum_dict[class_] = 0.0
            acc_norm_sum_dict[class_] = 0.0
            cnt_dict[class_] = 0.0

            for tt in TASK_NAME_MAPPING[class_]:
                acc_sum += sum(res[tt])
                cnt += len(res[tt])

                acc_sum_dict[class_] += sum(res[tt])
                cnt_dict[class_] += len(res[tt])

        print("\n\n\n", "total cnt:", cnt, "\n")
        for k in TASK_NAME_MAPPING.keys():
            if k in cnt_dict:
                print("%s ACC: %.2f " % (k, acc_sum_dict[k] / cnt_dict[k] * 100))
        print("AVERAGE ACC:%.2f " % (acc_sum / cnt * 100))

    dev_result = {}
    for subject_name in tqdm(SUBJECTS):
        # val_file_path = os.path.join(args.eval_data_path, 'val', f'{subject_name}_val.csv')
        dev_file_path = os.path.join(eval_data_path, "dev", f"{subject_name}_dev.csv")
        test_file_path = os.path.join(
            eval_data_path, "test", f"{subject_name}_test.csv"
        )
        # val_df = pd.read_csv(val_file_path, names=['question','A','B','C','D','answer'])
        dev_df = pd.read_csv(
            dev_file_path, names=["question", "A", "B", "C", "D", "answer"]
        )
        test_df = pd.read_csv(
            test_file_path, names=["question", "A", "B", "C", "D", "answer"]
        )

        ### Only test nsamples entires for fast eval. est 2:30 hrs for one subject.
        test_df = test_df.head(nsamples)

        score = eval_subject(
            model,
            model.tokenizer,
            subject_name,
            test_df,
            dev_df=dev_df,
            k=5,
            few_shot=True,
            save_result_dir=f"outs/mmlu_eval_result",
            batch_size=1,
        )
        dev_result[subject_name] = score
    cal_mmlu(dev_result)


def warmup(model, max_new_tokens=30):
    print(f"Warming up ... ")
    for prompt in prompts[0:1]:
        inputs = model.tokenizer(prompt, return_tensors="pt")
        generate_ids = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens,
        )
        _ = model.tokenizer.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
    print(f"Warm up DONE!! ")


def decode_prompt(
    model,
    tokenizer,
    prompt,
    input_ids=None,
    max_new_tokens=30,
    assistant_model=None,
    do_sample=False,
    apply_chat_tmpl=False,
    temperature=None,
):
    if input_ids is None:
        # print(f"prompt: {prompt}")
        ### If apply_chat_tmpl is enabled for prompt, it would apply chat tmpl to input as in official way.
        if apply_chat_tmpl:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            model_inputs = tokenizer([text], return_tensors="pt")
            input_ids_ = model_inputs.input_ids
        else:
            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids_ = inputs.input_ids
    else:
        ### If apply_chat_tmpl is enabled for benchmark, it would add the pre_tokens and end_tokens to input_ids. This is a hack to the official way.
        if apply_chat_tmpl:
            pre_tokens = tokenizer(
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n",
                return_tensors="pt",
            )
            end_tokens = tokenizer(
                "<|im_end|>\n<|im_start|>assistant\n", return_tensors="pt"
            )

            ### If input_ids for benchmark is less than 2* tokenized chat template, the input_ids is directly added with tokenized template.
            ### Else, we remove last few tokens to make the input_ids match with desired length (keep the length same)
            if (
                2 * (pre_tokens.input_ids.shape[1] + end_tokens.input_ids.shape[1])
                > input_ids.shape[1]
            ):
                input_ids_ = torch.cat(
                    (pre_tokens.input_ids, input_ids, end_tokens.input_ids), dim=1
                )
            else:
                input_ids_ = torch.cat(
                    (
                        pre_tokens.input_ids,
                        input_ids[
                            :,
                            : -(
                                pre_tokens.input_ids.shape[1]
                                + end_tokens.input_ids.shape[1]
                            ),
                        ],
                        end_tokens.input_ids,
                    ),
                    dim=1,
                )
        else:
            input_ids_ = input_ids
    logging.critical(f"[PROFILE] tokenizer:")

    attention_mask = torch.ones(input_ids_.shape)
    streamer = TextStreamer(model.tokenizer)
    if (
        (model.model_name == "code-llama-2-7b")
        or (model.model_name == "CodeLlama-7b-hf")
        or (model.model_name == "CodeLlama-7b-instruct-hf")
    ):
        start = time.perf_counter()
        generate_ids = model.generate(
            input_ids=input_ids_,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            streamer=streamer,
            assistant_model=assistant_model,
            do_sample=True,
            temperature=0.1,
            pad_token_id=model.tokenizer.eos_token_id,
        )
        end = time.perf_counter()

    elif (
        (model.model_name == "llama-2-7b-chat")
        or (model.model_name == "llama-2-13b-chat")
        or (model.model_name == "llama-2-7b")
        or (model.model_name == "llama-2-13b")
    ):
        do_sample = True
        temperature = 0.1
        # cache = SinkCache(window_length=128, num_sink_tokens=4)
        # generate_ids = model.generate(input_ids=input_ids_, attention_mask=attention_mask, max_new_tokens=60, streamer=streamer, assistant_model=assistant_model, do_sample=True, temperature=0.1, top_p=0.95, past_key_values=cache, use_cache=True)
        start = time.perf_counter()
        generate_ids = model.generate(
            input_ids=input_ids_,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            streamer=streamer,
            assistant_model=assistant_model,
            do_sample=True,
            temperature=0.1,
            top_p=0.95,
            pad_token_id=model.tokenizer.eos_token_id,
        )
        end = time.perf_counter()

    elif model.model_name == "Meta-Llama-3-8B-Instruct":
        do_sample = True
        temperature = 0.1
        # cache = SinkCache(window_length=128, num_sink_tokens=4)
        # generate_ids = model.generate(input_ids=input_ids_, attention_mask=attention_mask, max_new_tokens=60, streamer=streamer, assistant_model=assistant_model, do_sample=True, temperature=0.1, top_p=0.95, past_key_values=cache, use_cache=True)
        start = time.perf_counter()
        generate_ids = model.generate(
            input_ids=input_ids_,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            streamer=streamer,
            assistant_model=assistant_model,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            pad_token_id=model.tokenizer.eos_token_id,
        )
        end = time.perf_counter()
    else:
        start = time.perf_counter()
        generate_ids = model.generate(
            input_ids=input_ids_,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            streamer=streamer,
            assistant_model=assistant_model,
            do_sample=True,
            temperature=temperature,
            pad_token_id=model.tokenizer.eos_token_id,
        )
        end = time.perf_counter()

    generate_time = end - start
    prompt_tokens = input_ids_.shape[1]
    num_tokens_out = generate_ids.shape[1]
    new_tokens_generated = num_tokens_out - prompt_tokens
    time_per_token = (generate_time / new_tokens_generated) * 1e3
    logging.critical(
        f"[PROFILE] generate: {generate_time} for {num_tokens_out} tokens; prompt-tokens: {prompt_tokens}; time per generated token: {time_per_token}"
    )
    response = tokenizer.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    # print(f"response: {response}")
    logging.critical(f"response: {response}")


def decode_prompts(
    model,
    log_file,
    max_new_tokens=30,
    assistant_model=None,
    do_sample=False,
    apply_chat_tmpl=False,
    temperature=None,
    promptnow=None,
):
    if promptnow is None:
        if (
            (model.model_name == "starcoder")
            or (model.model_name == "code-llama-2-7b")
            or (model.model_name == "CodeLlama-7b-hf")
            or (model.model_name == "CodeLlama-7b-instruct-hf")
        ):
            ps = prompts_code
            m = 200
        else:
            ps = prompts
            m = max_new_tokens
    else:
        ps = promptnow
        m = max_new_tokens

    for prompt in ps:
        logging.critical("*" * 40)
        print("*" * 40)
        decode_prompt(
            model,
            model.tokenizer,
            prompt,
            max_new_tokens=m,
            assistant_model=assistant_model,
            do_sample=do_sample,
            apply_chat_tmpl=apply_chat_tmpl,
            temperature=temperature,
        )
    if promptnow is None:
        logging.shutdown()
        out_file = log_file.replace(".log", "_profile.csv")
        out_file = open(out_file, "w")
        ProfileLLM.analyze_profiling(log_file, out_file)
        out_file.close()


def infer_linear_shapes(model, apply_chat_tmpl=False):
    trainloader, testenc = get_wikitext2(model.tokenizer, nsamples=2, seqlen=4096)
    seqlens = [4, 8, 16, 32, 64]
    input_ids = next(iter(trainloader))[0]
    Utils.register_shapes_hook_linear(
        model, moduletype=torch.ao.nn.quantized.dynamic.modules.linear.Linear
    )
    for seqlen in seqlens:
        print("*" * 40)
        input_ids_test = input_ids[:, :seqlen]
        decode_prompt(
            model,
            model.tokenizer,
            prompt=None,
            input_ids=input_ids_test,
            max_new_tokens=1,
            apply_chat_tmpl=apply_chat_tmpl,
        )
        all_shapes = Utils.extract_shapes_linear()
        for key in all_shapes.keys():
            print(f"Seqlen:{seqlen}  Shape:{key}  Occurances:{all_shapes[key]}")
        Utils.linear_shapes = {}


def benchmark(
    model, dataset, task, log_file, assistant_model=None, apply_chat_tmpl=False
):
    if "opt" in model.model_name:
        max_seqlen = 2048
    else:
        max_seqlen = 4096
    trainloader, testenc = get_wikitext2(
        model.tokenizer, dataset=dataset, nsamples=2, seqlen=max_seqlen
    )
    if task == "benchmark":
        seqlens = [4, 8, 16, 32, 37, 64, 128, 256]
    else:
        seqlens = [512, 1024, 1537, 2048, 3072, 4000]
    input_ids = next(iter(trainloader))[0]
    for seqlen in seqlens:
        if seqlen < max_seqlen:
            logging.critical("*" * 40)
            print("*" * 40)
            print(f"Benchmarking for {seqlen} tokens ...")
            input_ids_test = input_ids[:, :seqlen]
            with stats.CPUStats("Benchmark", os.getpid()) as bench_stats:
                decode_prompt(
                    model,
                    model.tokenizer,
                    prompt=None,
                    input_ids=input_ids_test,
                    max_new_tokens=60,
                    assistant_model=assistant_model,
                    apply_chat_tmpl=apply_chat_tmpl,
                )

            cpu_load = round(bench_stats.get_data()["proc_cpu_load"] * 100, 2)
            print(
                f"Model:{model.model_name}  Input-Prompt-Length:{seqlen}  CPU-Load:{cpu_load}%"
            )

    logging.shutdown()
    out_file = log_file.replace(".log", "_profile.csv")
    out_file = open(out_file, "w")
    ProfileLLM.analyze_profiling(log_file, out_file)
    out_file.close()


def benchmark_ag(
    model,
    log_file,
    max_seqlen=2048,
    assistant_model=None,
    do_sample=False,
    temperature=None,
    apply_chat_tmpl=False,
):
    trainloader, testenc = get_wikitext2(model.tokenizer, nsamples=2, seqlen=max_seqlen)
    seqlens = [16, 32, 64]  # , 128]
    input_ids = next(iter(trainloader))[0]
    for seqlen in seqlens:
        input_ids_test = input_ids[:, :seqlen]
        logging.critical("*" * 40)
        for max_out in [16, 32, 64, 128, 256]:
            if max_out < max_seqlen:
                print("*" * 40)
                print(
                    f"Benchmarking input:{seqlen} tokens & output:{max_out} tokens ..."
                )
                decode_prompt(
                    model,
                    model.tokenizer,
                    prompt=None,
                    input_ids=input_ids_test,
                    max_new_tokens=max_out,
                    assistant_model=assistant_model,
                    do_sample=do_sample,
                    apply_chat_tmpl=apply_chat_tmpl,
                    temperature=temperature,
                )

    logging.shutdown()
    out_file = log_file.replace(".log", "_profile.csv")
    out_file = open(out_file, "w")
    ProfileLLM.analyze_profiling(log_file, out_file)
    out_file.close()


def count_gops(model):
    trainloader, testenc = get_wikitext2(model.tokenizer, nsamples=2, seqlen=4096)
    if "opt" in model.model_name:
        max_seqlen = 2048
    else:
        max_seqlen = 4096
    seqlens = [4, 8, 16, 32, 64]  # , 128, 256, 512, 1024, 2000, 3000, 4000]
    input_ids = next(iter(trainloader))[0]
    gops_arr = []
    for seqlen in seqlens:
        if seqlen < max_seqlen:
            logging.critical("*" * 40)
            print("*" * 40)
            print(f"Profiling for {seqlen} tokens ...")
            input_ids_test = input_ids[:, :seqlen]
            attention_mask = torch.ones((1, input_ids_test.numel()))
            gops = OpsProfile.profile(model, (input_ids_test, attention_mask))
            gops_arr.append(gops)
    print("*" * 40)
    for i, gops in enumerate(gops_arr):
        print(f"Model:{model.model_name}  Seqlen:{seqlens[i]}  GOPs:{gops*1e-9:.3f}")


def benchmark_code(model, log_file, max_seqlen=512, assistant_model=None, sample=True):
    """put all the relevant function into this benchmark function"""

    def write_jsonl(filename: str, data: Iterable[Dict], append: bool = False):
        """
        Writes an iterable of dictionaries to jsonl
        """
        if append:
            mode = "ab"
        else:
            mode = "wb"
        filename = os.path.expanduser(filename)
        if filename.endswith(".gz"):
            with open(filename, mode) as fp:
                with gzip.GzipFile(fileobj=fp, mode="wb") as gzfp:
                    for x in data:
                        gzfp.write((json.dumps(x) + "\n").encode("utf-8"))
        else:
            with open(filename, mode) as fp:
                for x in data:
                    fp.write((json.dumps(x) + "\n").encode("utf-8"))

    def clip_input(
        tokenizer, prompt, task_name, max_new_tokens=512, prompt_shots="", max_seql=4096
    ):
        if task_name == "xsum":
            input_ids = tokenizer(
                prompt_shots + "Article: " + prompt["document"] + "\nSummary:",
                return_tensors="pt",
            ).input_ids
        elif task_name == "cnndm":
            input_ids = tokenizer(
                prompt_shots + "Article: " + prompt["article"] + "\nSummary:",
                return_tensors="pt",
            ).input_ids
        elif task_name == "samsum":
            input_ids = tokenizer(
                prompt_shots + "Dialogue: " + prompt["dialogue"] + "\nSummary:",
                return_tensors="pt",
            ).input_ids
        elif task_name == "humaneval":
            format_tabs = False  ##Originally: True
            if format_tabs:
                prompt = prompt["prompt"].replace("    ", "\t")
            else:
                prompt = prompt["prompt"]
            # print("prompt:\n", prompt)
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        elif task_name == "humaneval instruct":
            format_tabs = False  ##Originally: True
            if format_tabs:
                prompt = prompt.replace("    ", "\t")
            else:
                prompt = prompt
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        if len(input_ids[0]) + max_new_tokens >= max_seql:
            print("(input ids+max token)> {}".format(max_seql))
            sample_num = len(input_ids[0]) + max_new_tokens - max_seql
            input_ids = torch.cat(
                (input_ids[0][:2], input_ids[0][2:-3][:-sample_num], input_ids[0][-3:]),
                dim=0,
            ).unsqueeze(0)
        return input_ids

    # humaneval data
    def get_humaneval():
        from datasets import load_dataset

        humaneval_data = load_dataset("openai_humaneval")["test"]
        return humaneval_data

    def count_indent(text: str) -> int:
        count = 0
        for char in text:
            if char == " ":
                count += 1
            else:
                break
        return count

    def fix_indents(text: str, multiple: int = 2):
        outputs = []
        for line in text.split("\n"):
            while count_indent(line) % multiple != 0:
                line = " " + line
            outputs.append(line)
        return "\n".join(outputs)

    def filter_code(completion: str, model=None) -> str:
        completion = completion.lstrip("\n")
        return completion.split("\n\n")[0]

    testloader = get_humaneval()
    task_name = "humaneval"
    big_reorg_dict = defaultdict(list)

    print(testloader)
    print(f"Length of test loader: {len(testloader)} ")
    for i, prompt in tqdm.tqdm(enumerate(testloader)):
        task_id = prompt["task_id"]
        input_ids = clip_input(model.tokenizer, prompt, "humaneval").to(model.device)
        logging.critical(f"[PROFILE] tokenizer:")

        input_len = len(input_ids[0])
        start = time.perf_counter()
        generate_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_seqlen,
            pad_token_id=model.tokenizer.eos_token_id,
            assistant_model=assistant_model,
            do_sample=sample,
            temperature=0.1,
            top_k=10,
            top_p=0.95,
        )
        end = time.perf_counter()
        generate_time = end - start
        prompt_tokens = input_ids.shape[1]
        num_tokens_out = generate_ids.shape[1]
        new_tokens_generated = num_tokens_out - prompt_tokens
        time_per_token = (generate_time / new_tokens_generated) * 1e3
        logging.critical(
            f"[PROFILE] generate: {generate_time} for {num_tokens_out} tokens; prompt-tokens: {prompt_tokens}; time per generated token: {time_per_token}"
        )
        completion = model.tokenizer.decode(generate_ids[0, input_ids.shape[1] :])
        completion = filter_code(fix_indents(completion))
        val_completion = completion
        print(f"response: {val_completion}")
        logging.critical(f"response: {val_completion}")

        # -------------------------below is results store-------------------------------------
        comp_item = dict(task_id="{}".format(task_id), completion=val_completion)
        big_reorg_dict["yxl"].append(comp_item)
        if i == 2:
            break

    result_data_path = "./results/codellama_spec_{}_on_npu".format(task_name)
    if not os.path.exists(result_data_path):
        os.makedirs(result_data_path)

    for key, completion in big_reorg_dict.items():
        write_jsonl("{0}/{1}.jsonl".format(result_data_path, key), completion)

    # similar to other benchmark methods
    logging.shutdown()
    out_file = log_file.replace(".log", "_profile.csv")
    out_file = open(out_file, "w")
    ProfileLLM.analyze_profiling(log_file, out_file)
    out_file.close()
