#
# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved. 
#

import logging
import time


class ProfileAIE:
    start_idx_arr = []
    end_idx_arr = []
    generate_times_arr = []
    num_tokens_out_arr = []
    num_tokens_in_arr = []
    exec_time_arr = []
    prompt_time_arr = []
    token_time_arr = []
    decoder_time_arr = []
    post_processing_time_arr = []
    pre_processing_time_arr = []

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
        cls.post_processing_time_arr = []
        cls.pre_processing_time_arr = []
        
    @classmethod
    def collect_sections(cls, logf):
        # print(f"{len(logf)}")
        for i, line in enumerate(logf):
            line = line.lstrip().rstrip().split(" ")
            if len(line) > 4:
                # input(line)
                if (line[4] == "tokenizer:"):
                    cls.start_idx_arr.append(i + 1)
                if (line[4] == "generate:"):
                    cls.end_idx_arr.append(i)
                    t = float(line[5])
                    cls.generate_times_arr.append(t)
                    num_tokens_out = int(line[7])
                    cls.num_tokens_out_arr.append(num_tokens_out)
                    num_tokens_in = int(line[10].rstrip(';'))
                    cls.num_tokens_in_arr.append(num_tokens_in)
        print(f"Number of prompts found in log: {len(cls.start_idx_arr)}")

    @classmethod 
    def parse_section(cls, outlog, filename, prompt_num, logf, start_idx, end_idx, generate_time, num_tokens_out, num_tokens_in, ops_profiler, custommodel):
        cls.exec_time_arr = []
        cls.prompt_time_arr = []
        cls.token_time_arr = []
        cls.decoder_time_arr = []
        cls.post_processing_time_arr = []
        cls.pre_processing_time_arr = []
        for i in range(start_idx, end_idx, 1):
            line = logf[i].lstrip().rstrip().split(" ")
            if line[4] != "model_decoder_forward":
                m = int(line[4])
                k = int(line[5])
                n = int(line[6])
                exec_time_start = float(line[14])
                exec_time_end = float(line[15])
                exec_time = (exec_time_end - exec_time_start)*1e9
                cls.exec_time_arr.append(exec_time)
                if m > 1:
                    cls.prompt_time_arr.append(exec_time)
                else:
                    cls.token_time_arr.append(exec_time)
            if line[4] == "model_decoder_forward":
                #print(line)
                decoder_time = float(line[5])
                pre_processing_time = float(line[6])
                post_processing_time = float(line[7])
                cls.decoder_time_arr.append(decoder_time)
                cls.post_processing_time_arr.append(post_processing_time)
                cls.pre_processing_time_arr.append(pre_processing_time)

        generate_time *= 1e9
        matmul_prompt_time = sum(cls.prompt_time_arr)
        matmul_token_time = sum(cls.token_time_arr)
        matmul_cumulative_time = sum(cls.exec_time_arr)
        other_layers_time =  generate_time - matmul_cumulative_time
        new_tokens_generated = num_tokens_out - num_tokens_in
        
        if len(cls.decoder_time_arr)>0:
            decoder_time_prefill_phase = cls.decoder_time_arr[0] 
            decoder_time_token_phase = sum(cls.decoder_time_arr[1:]) 
            post_processing_time_prefill_phase = cls.post_processing_time_arr[0]
            post_processing_time_token_phase = sum(cls.post_processing_time_arr[1:])
            pre_processing_time_prefill_phase = cls.pre_processing_time_arr[0]
            pre_processing_time_token_phase = sum(cls.pre_processing_time_arr[1:])
            prefill_phase = (pre_processing_time_prefill_phase + decoder_time_prefill_phase + post_processing_time_prefill_phase)*1e-6
            time_per_token = ((pre_processing_time_token_phase + decoder_time_token_phase + post_processing_time_token_phase)/(new_tokens_generated-1))*1e-6
        else:
            decoder_time_prefill_phase = 0
            decoder_time_token_phase = 0
            post_processing_time_prefill_phase = 0
            post_processing_time_token_phase = 0
            pre_processing_time_prefill_phase = 0
            pre_processing_time_token_phase = 0
            prefill_phase = 0
            time_per_token = 0
        if ops_profiler:
            if custommodel is False:
                outlog.write(f"{filename},{prompt_num+1},{num_tokens_in},{matmul_prompt_time*1e-9:.3f},{matmul_token_time*1e-9:.3f},{matmul_cumulative_time*1e-9:.3f},{other_layers_time*1e-9:.3f},{generate_time*1e-9:.3f},na,na,na,na,na,na,{num_tokens_out},{new_tokens_generated},na,na,na\n")
            else:
                outlog.write(f"{filename},{prompt_num+1},{num_tokens_in},{matmul_prompt_time*1e-9:.3f},{matmul_token_time*1e-9:.3f},{matmul_cumulative_time*1e-9:.3f},{other_layers_time*1e-9:.3f},{generate_time*1e-9:.3f},{pre_processing_time_prefill_phase*1e-9:.3f},{decoder_time_prefill_phase*1e-9:.3f},{post_processing_time_prefill_phase*1e-9:.3f},{pre_processing_time_token_phase*1e-9:.3f},{decoder_time_token_phase*1e-9:.3f},{post_processing_time_token_phase*1e-9:.3f},{num_tokens_out},{new_tokens_generated},{prefill_phase:.0f},{time_per_token:.0f},{1000.0/time_per_token:.1f}\n")
        else:
            if custommodel is False:
                outlog.write(f"{filename},{prompt_num+1},{num_tokens_in},na,na,na,na,{generate_time*1e-9:.3f},na,na,na,na,na,na,{num_tokens_out},{new_tokens_generated},na,na,na\n")
            else:
                outlog.write(f"{filename},{prompt_num+1},{num_tokens_in},na,na,na,na,{generate_time*1e-9:.3f},{pre_processing_time_prefill_phase*1e-9:.3f},{decoder_time_prefill_phase*1e-9:.3f},{post_processing_time_prefill_phase*1e-9:.3f},{pre_processing_time_token_phase*1e-9:.3f},{decoder_time_token_phase*1e-9:.3f},{post_processing_time_token_phase*1e-9:.3f},{num_tokens_out},{new_tokens_generated},{prefill_phase:.0f},{time_per_token:.0f},{1000.0/time_per_token:.1f}\n")
        outlog.flush()

        if custommodel:
            print(f"Example#:{prompt_num+1}\tPrompt-len:{num_tokens_in}\tNew-tokens-generated:{new_tokens_generated}\tTotal-time:{generate_time*1e-9:.3f}s\tPrefill-phase:{prefill_phase:.3f}ms\tTime/token:{time_per_token:.0f}ms\tTokens/sec:{1000.0/time_per_token:.1f}")
    
    @classmethod 
    def analyze_profiling(cls, ops_profiler, custommodel, in_file, out_file):
        out_file.write(
            "Filename,Example#,Num_Tokens_In,MatMul_time_Prefill_phase[s],MatMul_time_Token_phase[s],MatMul_time_Cumulative[s],All_Other_layers[s],Generate_Time[s],Preprocessing_time_Prefill_phase[s],Decoder_time_Prefill_phase[s],Postprocessing_time_Prefill_phase[s],Postprocessing_time_Token_phase[s],Decoder_time_Token_phase[s],Postprocessing_time_Token_phase[s],Num_Tokens_Out,Num_New_Tokens,Prefill_Phase[ms],Time_per_Token[ms],Tokens\sec\n")
        with open(in_file, 'r') as f:
            logf = f.readlines()
            cls.collect_sections(logf)

            for i in range(len(cls.start_idx_arr)):
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
                    ops_profiler,
                    custommodel
                    )
        
