import os
import time
from typing import Optional

import qlinear

# SmoothQuant
import smooth
import torch

# AWQ
from pre_quant import apply_awq, run_awq
from qmodule import WQLinear
from quantizer import real_quantize_model_weight
from ryzenai_llm_engine import RyzenAILLMEngine


class QuantConfig:
    def __init__(
        self,
        quant_mode: Optional[str],
        model_name: str,
        w_bit: int = 4,
        group_size: int = 128,
        use_qscales: bool = True,
        dataset: str = "raw",
    ):
        self.quant_mode = quant_mode  # awq, smoothquant, pergrp
        self.use_qscales = use_qscales  # if False, calc new scales for awq
        self.w_bit = w_bit  # only for awq/pergrp
        self.group_size = group_size  # only for awq/pergrp
        self.datatset = dataset  # raw or non-raw : wikitext2-raw-v1, wikitext2-v1

    def __repr__(self):
        return f"QuantConfig(quant_mode={self.quant_mode}, w_bit={self.w_bit}, group_size={self.group_size}, datatset={self.datatset}, use_qscales={self.use_qscales})"


class RyzenAILLMQuantizer:
    supported_modes = ["awq", "awqplus", "smoothquant", "pergrp"]
    supported_models = {
        "smoothquant": [
            "opt-125m",
            "opt-1.3b",
            "opt-2.7b",
            "opt-6.7b",
            "opt-13b",
            "llama-2-7b",
            "llama-2-7b-chat",
            "bloom-560m",
            "bloom-1b1",
            "bloom-3b",
        ],
        "awq": [
            "opt-125m",
            "opt-1.3b",
            "opt-2.7b",
            "opt-6.7b",
            "opt-13b",
            "llama-2-7b",  # local weights needed
            "llama-2-7b-chat",  # local weights needed
            "llama-2-13b-chat",  # local weights needed
            "starcoder",
            "code-llama-2-7b",  # local weights needed
            "CodeLlama-7b-hf",
            "CodeLlama-7b-instruct-hf",
            "gemma-2b",
            "gemma-7b",
            "Qwen-7b",
            "Qwen1.5-7B-Chat",
            "chatglm3-6b",
        ],
        "awqplus": [
            "opt-125m",
            "opt-1.3b",
            "opt-2.7b",
            "opt-6.7b",
            "opt-13b",
            "llama-2-7b",  # local weights needed
            "llama-2-7b-chat",  # local weights needed
            "llama-2-13b-chat",  # local weights needed
            "starcoder",
            "code-llama-2-7b",  # local weights needed
            "CodeLlama-7b-hf",
            "CodeLlama-7b-instruct-hf",
            "gemma-2b",
            "gemma-7b",
            "Qwen-7b",
            "Qwen1.5-7B-Chat",
            "chatglm3-6b",
        ],
    }

    @classmethod
    def qualify(cls, model: torch.nn.Module, quant_config: QuantConfig) -> bool:
        ok_to_proceed = False
        if quant_config.quant_mode == "pergrp":
            ok_to_proceed = True
        else:
            if quant_config.quant_mode not in cls.supported_modes:
                print(
                    f"[RyzenAILLMQuantizer] Unsupported QConfig encountered {quant_config.quant_mode} \nSupported:{cls.supported_modes} *** exiting *** \n"
                )
            else:
                for m in cls.supported_models[quant_config.quant_mode]:
                    # print(f"Checking for {model.model_name} in {m}")
                    if model.model_name in m:
                        ok_to_proceed = True
                if ok_to_proceed == False:
                    print(
                        f"[RyzenAILLMQuantizer] Unsupported Model for given QConfig encountered \n\n Change quant_mode to pergrp to run without AWQ/SmoothQuant (OR) generate new scales for AWQ and proceed ... \n\n *** exiting  *** \n"
                    )
        if (quant_config.quant_mode == "awq") or (quant_config.quant_mode == "awqplus"):
            if quant_config.group_size != 128:
                print(
                    f"[RyzenAILLMQuantizer] For AWQ, AWQPlus, only groupsize:128 is supported currently ... exiting ..."
                )
                ok_to_proceed = False
        return ok_to_proceed

    @classmethod
    def quantize(cls, model: torch.nn.Module, quant_config: QuantConfig):
        ok_to_proceed = cls.qualify(model, quant_config)
        if ok_to_proceed == False:
            print(f"quant_config: {quant_config}")
            print(f"model_name: {model.model_name}")
            raise SystemExit

        if quant_config.quant_mode == "smoothquant":
            act_scales = torch.load(
                os.getenv("PYTORCH_AIE_PATH")
                + "/ext/smoothquant/act_scales/"
                + "%s.pt" % model.model_name
            )
            print(
                f"[RyzenAILLMQuantizer] [SmoothQuant] Applying SmoothQuant scales ..."
            )
            smooth.smooth_lm(model, act_scales, 0.5)
            print(
                f"[RyzenAILLMQuantizer] [SmoothQuant] Applying SmoothQuant scales ... DONE!"
            )
            print(f"[RyzenAILLMQuantizer] [SmoothQuant] Quantizing Linear layers ...")
            torch.ao.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8, inplace=True
            )
            print(
                f"[RyzenAILLMQuantizer] [SmoothQuant] Quantizing Linear layers ... DONE!"
            )
            return model
        elif (quant_config.quant_mode == "awq") or (
            quant_config.quant_mode == "awqplus"
        ):
            awq_config = {
                "zero_point": True,
                "q_group_size": quant_config.group_size,
            }  # whether to use group quantization
            if quant_config.use_qscales == False:
                print(f"[RyzenAILLMQuantizer] [AWQ] Calculating AWQ scales ...")
                if "gemma" in model.model_name:
                    auto_scale = False
                else:
                    auto_scale = True
                awq_results = run_awq(
                    model,
                    model.tokenizer,
                    w_bit=quant_config.w_bit,
                    q_config=awq_config,
                    n_samples=quant_config.group_size,
                    seqlen=512,
                    auto_scale=auto_scale,
                )  # auto_scale False for Gemma-2b, 7b
                print(f"[RyzenAILLMQuantizer] [AWQ] Savng AWQ scales ...")
                awq_filename = "./%s-w%d-g%d-generated.pt" % (
                    model.model_name,
                    quant_config.w_bit,
                    quant_config.group_size,
                )
                torch.save(awq_results, awq_filename)
                print(f"[RyzenAILLMQuantizer] [AWQ] Saved AWQ scales to {awq_filename}")
                awq_results = torch.load(awq_filename, map_location="cpu")
                print(
                    f"[RyzenAILLMQuantizer] [AWQ] Loaded AWQ scales from {awq_filename}"
                )
            else:
                if (model.model_name == "code-llama-2-7b") or (
                    model.model_name == "CodeLlama-7b-hf"
                ):
                    awq_filename = os.getenv(
                        "AWQ_CACHE"
                    ) + "CodeLlama-7b-Instruct-hf-w%d-g128.pt" % (quant_config.w_bit)
                else:
                    awq_filename = os.getenv("AWQ_CACHE") + "\%s-w%d-g128.pt" % (
                        model.model_name,
                        quant_config.w_bit,
                    )
                if not os.path.exists(awq_filename):
                    print(f"[RyzenAILLMQuantizer] [AWQ] Looking for {awq_filename}")
                    print(
                        f"[RyzenAILLMQuantizer] [AWQ] No precalculated scales available for {model.model_name} w_bit:{quant_config.w_bit} group_size:{quant_config.group_size}"
                    )
                    raise SystemExit

                awq_results = torch.load(awq_filename, map_location="cpu")
                print(
                    f"[RyzenAILLMQuantizer] [AWQ] Loaded AWQ scales from {awq_filename}"
                )

            print(f"[RyzenAILLMQuantizer] [AWQ] Applying AWQ scales to model ...")
            apply_awq(model, awq_results)
            real_quantize_model_weight(
                model, w_bit=quant_config.w_bit, q_config=awq_config
            )

            if quant_config.quant_mode == "awqplus":
                print(
                    f"[RyzenAILLMQuantizer] [AWQplus] Applying Per Grp w4abf16 quantization with grp=32 on all other Linear layers..."
                )
                RyzenAILLMEngine.replace_node(
                    model,
                    torch.nn.Linear,
                    qlinear.QLinearPerGrp,
                    (),
                    {"device": "cpu", "w_bit": quant_config.w_bit, "group_size": 32},
                )
            model = model.to(torch.bfloat16)
            return model
        elif quant_config.quant_mode == "pergrp":
            print(
                f"[RyzenAILLMQuantizer] [PerGrp] Applying Per Grp w4abf16 quantization ..."
            )
            RyzenAILLMEngine.replace_node(
                model,
                torch.nn.Linear,
                qlinear.QLinearPerGrp,
                (),
                {
                    "device": "cpu",
                    "w_bit": quant_config.w_bit,
                    "group_size": quant_config.group_size,
                },
            )
            print(
                f"[RyzenAILLMQuantizer] [PerGrp] Applying Per Grp w4abf16 quantization ... DONE!"
            )
            model = model.to(torch.bfloat16)
            return model
        else:
            print(
                f"[RyzenAILLMQuantizer] Encountered unsupported quant_mode, exitting ..."
            )
            raise SystemExit
