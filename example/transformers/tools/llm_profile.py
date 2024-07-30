#
# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
#

import logging
import time

import tabulate as tab


class ProfileLLM:
    start_idx_arr = []
    end_idx_arr = []
    generate_times_arr = []
    num_tokens_out_arr = []
    num_tokens_in_arr = []
    exec_time_arr = []
    prompt_time_arr = []
    token_time_arr = []
    decoder_time_arr = []

    @classmethod
    def clear_entries(cls):
        cls.start_idx_arr = []
        cls.end_idx_arr = []
        cls.generate_times_arr = []
        cls.num_tokens_out_arr = []
        cls.num_tokens_in_arr = []
        cls.exec_time_arr = []
        cls.prompt_time_arr = []
        cls.token_time_arr = []
        cls.decoder_time_arr = []

    @classmethod
    def collect_sections(cls, logf):
        for i, line in enumerate(logf):
            line = line.lstrip().rstrip().split(" ")
            if len(line) > 1:
                # print(f"line: {line}")
                if line[1] == "tokenizer:":
                    cls.start_idx_arr.append(i + 1)
                elif line[1] == "generate:":
                    cls.end_idx_arr.append(i)
                    t = float(line[2])
                    cls.generate_times_arr.append(t)
                    num_tokens_out = int(line[4])
                    cls.num_tokens_out_arr.append(num_tokens_out)
                    num_tokens_in = int(line[7].rstrip(";"))
                    cls.num_tokens_in_arr.append(num_tokens_in)
        print(f"\n\nNumber of prompts found in log: {len(cls.start_idx_arr)}\n")

    @classmethod
    def parse_section(
        cls,
        outlog,
        filename,
        prompt_num,
        logf,
        start_idx,
        end_idx,
        generate_time,
        num_tokens_out,
        num_tokens_in,
    ):
        cls.exec_time_arr = []
        cls.prompt_time_arr = []
        cls.token_time_arr = []
        cls.decoder_time_arr = []
        for i in range(start_idx, end_idx, 1):
            line = logf[i].lstrip().rstrip().split(" ")
            # print(f"line : {line}")
            if line[1] != "model_decoder_forward":
                # print(f"line : {line}")
                m = int(line[1])
                k = int(line[2])
                n = int(line[4])
                exec_time_start = float(line[11])
                exec_time_end = float(line[12])
                exec_time = exec_time_end - exec_time_start
                cls.exec_time_arr.append(exec_time)
                if m > 1:
                    cls.prompt_time_arr.append(exec_time)
                else:
                    cls.token_time_arr.append(exec_time)
            else:
                # print(f"line : {line}")
                decoder_time = float(line[2])
                cls.decoder_time_arr.append(decoder_time)

        matmul_prompt_time = sum(cls.prompt_time_arr)
        matmul_token_time = sum(cls.token_time_arr)
        matmul_cumulative_time = sum(cls.exec_time_arr)
        other_layers_time = generate_time - matmul_cumulative_time
        new_tokens_generated = num_tokens_out - num_tokens_in

        if len(cls.decoder_time_arr) > 0:
            decoder_time_prefill_phase = cls.decoder_time_arr[0]
            decoder_time_token_phase = sum(cls.decoder_time_arr[1:])
            prefill_phase = decoder_time_prefill_phase * 1e3
            if new_tokens_generated > 1:
                time_per_token = (decoder_time_token_phase * 1e3) / (
                    new_tokens_generated - 1
                )
                tokens_per_sec = 1000.0 / time_per_token
            else:
                time_per_token = "na"
                tokens_per_sec = "na"
        else:
            decoder_time_prefill_phase = "na"
            decoder_time_token_phase = "na"
            prefill_phase = "na"
            time_per_token = "na"
            tokens_per_sec = "na"

        outlog.write(
            f"{filename},{prompt_num+1},{num_tokens_in},{matmul_prompt_time:.3f},{matmul_token_time:.3f},{matmul_cumulative_time:.3f},{other_layers_time:.3f},{generate_time:.3f},{decoder_time_prefill_phase},{decoder_time_token_phase},{num_tokens_out},{new_tokens_generated},{prefill_phase},{time_per_token},{tokens_per_sec}\n"
        )

        outlog.flush()

        return [
            prompt_num + 1,
            num_tokens_in,
            new_tokens_generated,
            generate_time,
            prefill_phase,
            time_per_token,
            tokens_per_sec,
        ]

    @classmethod
    def analyze_profiling(cls, in_file, out_file):
        out_file.write(
            "Filename,Example#,Num_Tokens_In,MatMul_time_Prefill_phase[s],MatMul_time_Token_phase[s],MatMul_time_Cumulative[s],All_Other_layers[s],Generate_Time[s],Decoder_time_Prefill_phase[s],Decoder_time_Token_phase[s],Num_Tokens_Out,Num_New_Tokens,Prefill_Phase[ms],Time_per_Token[ms],Tokens\sec\n"
        )
        with open(in_file, "r") as f:
            logf = f.readlines()
            cls.collect_sections(logf)

            perf_table = [
                [
                    "Example#",
                    "Prompt Length (tokens)",
                    "New Tokens Generated",
                    "Total Time (s)",
                    "Prefill Phase (ms)",
                    "Time/Token (ms)",
                    "Tokens/Sec",
                ]
            ]
            for i in range(len(cls.start_idx_arr)):
                perf_table.append(
                    cls.parse_section(
                        out_file,
                        in_file,
                        i,
                        logf,
                        cls.start_idx_arr[i],
                        cls.end_idx_arr[i],
                        cls.generate_times_arr[i],
                        cls.num_tokens_out_arr[i],
                        cls.num_tokens_in_arr[i],
                    )
                )
            print(tab.tabulate(perf_table, headers="firstrow", tablefmt="github"))


class TorchModuleProfile:
    module_start_times = {}
    module_end_times = {}

    @classmethod
    def register_profile_hooks(cls, model):
        cls.module_start_times = {}
        cls.module_end_times = {}

        def generate_hook_fn(name):
            def hook_fn(module, *args, **kwargs):
                tt = time.perf_counter()
                if cls.module_end_times.get(name) == None:
                    cls.module_end_times[name] = [tt]
                else:
                    cls.module_end_times[name].append(tt)

            return hook_fn

        def generate_pre_hook_fn(name):
            def hook_fn(module, *args, **kwargs):
                tt = time.perf_counter()
                if cls.module_start_times.get(name) == None:
                    cls.module_start_times[name] = [tt]
                else:
                    cls.module_start_times[name].append(tt)

            return hook_fn

        def register_all_layers(model):
            print(f"[TorchModuleProfile] Registering profile pre-hooks ...")
            for name, module in model.named_modules():
                module.register_forward_pre_hook(generate_pre_hook_fn(name))
            print(f"[TorchModuleProfile] Registering profile pre-hooks ... DONE")

            print(f"[TorchModuleProfile] Registering profile hooks ...")
            for name, module in model.named_modules():
                module.register_forward_hook(generate_hook_fn(name))
            print(f"[TorchModuleProfile] Registering profile hooks ... DONE")

        register_all_layers(model)

    @classmethod
    def generate_report(cls, model):
        profileoutputfile = f"./logs/log_{model.model_name}_torch_module_profile.csv"
        with open(profileoutputfile, "w") as f:
            f.write(f"ModuleName,ttft,tokentimes\n")
            for key in cls.module_start_times.keys():
                st = cls.module_start_times[key]
                en = cls.module_end_times[key]
                if key == "":
                    f.write(f"model,")
                else:
                    f.write(f"{key},")
                for i in range(len(st)):
                    tt = en[i] - st[i]
                    f.write(f"{tt},")
                f.write(f"\n")
        print(
            f"[TorchModuleProfile] Profile output written to file: {profileoutputfile}"
        )
