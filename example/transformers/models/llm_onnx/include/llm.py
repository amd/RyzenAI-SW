##
## Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.
##

import logging
import time

import onnxruntime
from optimum.onnxruntime.modeling_decoder import ORTModelForCausalLM, ORTOPTForCausalLM


class ORTModelEval(ORTModelForCausalLM):
    def __init__(self, model: onnxruntime.InferenceSession, config, **kwargs):
        # Prepare kwargs for ORTModelForCausalLM ctor
        _use_io_binding = kwargs.get("use_io_binding", None)
        _model_save_dir = kwargs.get("model_save_dir", None)
        _preprocessors = kwargs.get("preprocessors", None)
        _generation_config = kwargs.get("generation_config", None)
        _use_cache = kwargs.get("use_cache", True)
        # Call ctor
        super().__init__(
            model,
            config,
            use_cache=_use_cache,
            use_io_binding=_use_io_binding,
            model_save_dir=_model_save_dir,
            generation_config=_generation_config,
            preprocessors=_preprocessors,
        )
        self.tokenizer = None

    @property
    def use_io_binding(self):
        return True

    @classmethod
    def _from_pretrained(cls, model_id, config, **kwargs):
        ort_model_clm = super()._from_pretrained(model_id, config, **kwargs)
        # Create class with base class session
        return cls(ort_model_clm.model, config, **kwargs)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        labels=None,
        use_cache_branch=None,
        **kwargs,
    ):
        st = time.perf_counter()
        outputs = super().forward(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            labels=labels,
            use_cache_branch=use_cache_branch,
            **kwargs,
        )
        en = time.perf_counter()
        logging.critical(f"[PROFILE] model_decoder_forward {en-st}")
        return outputs


class OPTORTModelEval(ORTOPTForCausalLM):
    def __init__(self, model: onnxruntime.InferenceSession, config, **kwargs):
        # Prepare kwargs for ORTModelForCausalLM ctor
        _use_io_binding = kwargs.get("use_io_binding", None)
        _model_save_dir = kwargs.get("model_save_dir", None)
        _preprocessors = kwargs.get("preprocessors", None)
        _generation_config = kwargs.get("generation_config", None)
        _use_cache = kwargs.get("use_cache", True)
        # Call ctor
        super().__init__(
            model,
            config,
            use_cache=_use_cache,
            use_io_binding=_use_io_binding,
            model_save_dir=_model_save_dir,
            generation_config=_generation_config,
            preprocessors=_preprocessors,
        )
        self.tokenizer = None

    @property
    def use_io_binding(self):
        return True

    @classmethod
    def _from_pretrained(cls, model_id, config, **kwargs):
        ort_model_clm = super()._from_pretrained(model_id, config, **kwargs)
        # Create class with base class session
        return cls(ort_model_clm.model, config, **kwargs)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        labels=None,
        use_cache_branch=None,
        **kwargs,
    ):
        st = time.perf_counter()
        outputs = super().forward(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            labels=labels,
            use_cache_branch=use_cache_branch,
            **kwargs,
        )
        en = time.perf_counter()
        logging.critical(f"[PROFILE] model_decoder_forward {en-st}")
        return outputs
