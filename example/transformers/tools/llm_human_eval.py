#
# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
#

import argparse
import copy
import gzip
import json
import logging
import os
import random
import re
import subprocess
import time
from collections import defaultdict
from typing import Dict, Iterable

import numpy as np
import stats
import torch
import tqdm
from decoding import clip_input, infer_input_ids
from human_eval.data import write_jsonl
from llm_profile import ProfileLLM
from perf_estimator import OpsProfile
from utils import Utils

import transformers


def run_human_eval(model, humanevalpath):

    task_name = "humaneval"
    max_seql = 2048
    max_new_tokens = 10
    tokenizer = model.tokenizer

    with open(humanevalpath, "r") as json_file:
        humaneval_data = json.load(json_file)

    ## Note: Load full dataset
    """
    from datasets import load_from_disk
    humanevalpath = args.humanevalpath
    humaneval = load_from_disk(humanevalpath)
    humaneval_data  = humaneval['test']
    """

    prompt_shots = ""

    ########### 3. Add wanted settings, manually
    """following this format:
    wanted_setting == 'spec_dec' + 't' + wanted temperature + k + wanted max_new_tokens
    """
    added_name = "base_" + str(0.1)
    name_to_variable = "result_" + copy.deepcopy(added_name)
    main_metrics = {}
    wanted_settings = ["base"]
    wanted_settings.append(added_name)
    for name in wanted_settings:
        main_metrics["completion_{}".format(name)] = []
        main_metrics["time_{}".format(name)] = []
        main_metrics["token_time_{}".format(name)] = []
        main_metrics["generated_tokens_{}".format(name)] = []
        main_metrics["prefill_time_{}".format(name)] = []
        if "spec_dec" in name:
            main_metrics["matchness_{}".format(name)] = []
            main_metrics["num_drafted_tokens_{}".format(name)] = []
            main_metrics["matchness_list_{}".format(name)] = []

    ########### 4. Launch Speculative Decoding within a file context management
    result_data_path = "./results/code_llama_spec_humaneval_npu.json"
    with open(result_data_path, "w") as json_file:
        list_of_metrics = []
        for i, prompt in enumerate(humaneval_data):
            print(i)
            input_ids = clip_input(
                tokenizer,
                prompt,
                task_name,
                max_new_tokens=max_new_tokens,
                prompt_shots=prompt_shots,
                max_seql=max_seql,
            )

            # NOTE: Run base/draft model for further calculation of speedup, and so on.
            print("#" * 20, "Started TARGET Model Base", "#" * 20)
            result_base = infer_input_ids(
                target_model=model,
                draft_model=None,
                tokenizer=tokenizer,
                input_ids=input_ids,
                generate_fn="base",
                do_sample=False,
                max_new_tokens=max_new_tokens,
                temperature=0.1,
                early_stop=True,
                device="npu",
                draft_device="cpu",
            )
            print("#" * 20, "Finished TARGET Model Base", "#" * 20)

            ##################################### NOTE: post-process all metrics based on the outputed results ########################################
            results = [  ## to add more rested results' settings
                ("base", result_base),
                # (added_name, name_to_variable)
            ]
            # name_to_variable = 'result_' + copy.deepcopy(added_name)

            for key, result in results:
                print(key, result)
                main_metrics["time_" + key].append(result["time"])
                main_metrics["token_time_" + key].append(
                    result["time"] / result["generate_ids"].shape[1]
                )
                if "base" not in key:
                    main_metrics["matchness_" + key].append(result["matchness"])
                    main_metrics["num_drafted_tokens_" + key].append(
                        result["num_drafted_tokens"]
                    )
                    main_metrics["matchness_list_{}".format(key)].append(
                        result["matchness_list"]
                    )

            metric = {}
            for key, result in results:
                metric["Completion_" + key] = result.get("completion", None)
                metric["generated_tokens_" + key] = result["generate_ids"].shape[1]
                if "base" in key:
                    metric["prefill_time_" + key] = (
                        result["time_records"]["base_gen_0_after"]
                        - result["time_records"]["base_gen_0_before"]
                    )

                metric[f"total time {key}"] = result["time"]
                metric[f"mean token time {key}"] = np.mean(
                    main_metrics["token_time_{}".format(key)]
                )

            for key, value in metric.items():
                if isinstance(value, float):
                    metric[key] = f"{value:.4f}"

            metric["task_id"] = prompt["task_id"]
            metric["prompts tokens num"] = len(input_ids)
            # metric["init_infer_time"] = name_to_variable.get("init_infer_time", None)
            name_to_variable = "result_" + copy.deepcopy(added_name)
            print("prompt['task_id']:\t", prompt["task_id"])

            print(f"data {i},{metric}")
            list_of_metrics.append(metric)
        json.dump(list_of_metrics, json_file)

    to_accquire_pass_one_scores(result_data_path)


def to_accquire_pass_one_scores(result_data: str):

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

    cnter = 0

    sixty_acc_samples = [
        6,
        7,
        22,
        28,
        35,
        57,
        59,
        129,
        151,
        154,
    ]  ##NOTE: this is decided by the sampled specific code prompts

    to_mk_path = os.getenv("PYTORCH_AIE_PATH") + "models/llm/results/samples_{}".format(
        result_data[:-5][10:]
    )
    if not os.path.exists(to_mk_path):
        os.mkdir(to_mk_path)
    samples_dir = to_mk_path

    with open(result_data, "r") as f:
        data = json.load(f)
        big_reorg_dict = defaultdict(list)
        for inx, item in enumerate(data):
            metrics = item  ##json.loads(item)
            completion_keys = list()
            for key, val in metrics.items():
                if "Completion" in key:
                    completion_keys.append(key)
                    # val_completion = filter_code(val)
                    val_completion = "    " + filter_code(fix_indents(val))
                    # comp_item = dict(task_id="HumanEval/{}".format(inx), completion=val_completion)
                    comp_item = dict(
                        task_id="HumanEval/{}".format(sixty_acc_samples[inx]),
                        completion=val_completion,
                    )
                    big_reorg_dict[key].append(comp_item)
            cnter += 1
            print("*" * 20, cnter, "*" * 20)

    for key, completion in big_reorg_dict.items():
        write_jsonl("{0}/{1}.jsonl".format(samples_dir, key), completion)

    # Bash script file path
    def run_bash_script():
        # Running the Bash script using subprocess.run
        completion_file = os.getenv(
            "PYTORCH_AIE_PATH"
        ) + "models/llm/results/samples_{}/Completion_base.jsonl".format(
            result_data[:-5][10:]
        )
        process = subprocess.run(
            ["evaluate_functional_correctness", completion_file],
            capture_output=True,
            text=True,
        )

        if process.returncode == 0:
            print("Bash script executed successfully.")
            # Outputting the standard output and standard error of the Bash script
            print("STDOUT:", process.stdout)
            print("STDERR:", process.stderr)
        else:
            print("Bash script failed to execute.")
            print("STDERR:", process.stderr)

    print("start pass@1 score evaluating!!!")
    run_bash_script()
