import warnings
import time
from typing import Any, Dict, List, Optional
from pydantic import PrivateAttr
import onnxruntime_genai as og
from langchain_core.language_models.llms import LLM

warnings.filterwarnings("ignore", category=UserWarning, module="langchain")
warnings.filterwarnings("ignore", category=DeprecationWarning)


class custom_llm(LLM):
    _model: og.Model = PrivateAttr()
    _tokenizer: og.Tokenizer = PrivateAttr()
    _tokenizer_stream: og.Tokenizer = PrivateAttr()
    profile: Dict[str, Any] = {}

    def __init__(self, model_path: str, **kwargs: Any):
        super().__init__(**kwargs)
        self._model = og.Model(model_path)
        self._tokenizer = og.Tokenizer(self._model)
        self._tokenizer_stream = self._tokenizer.create_stream()
        self.profile = {}
        

    def _prepare_generator(self, prompt: str) -> og.Generator:
        input_tokens = self._tokenizer.encode(prompt)
        self.profile["input_token_length"] = len(input_tokens)

        params = og.GeneratorParams(self._model)
        search_options = {
            "max_length": min(2048, len(input_tokens) + 1024),
            "temperature": 0.5,
            "top_k": 40,
            "top_p": 0.9
        }
        params.set_search_options(**search_options)

        generator = og.Generator(self._model, params)
        return generator, input_tokens

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        if not hasattr(self, "_call_count"):
            self._call_count = 0
        self._call_count += 1
        print(f"LLM_call invoked: {self._call_count} time(s)")

        generator, input_tokens = self._prepare_generator(prompt)
        response_tokens = []

        # === TTFT: append prompt + generate first token ===
        ttft_start = time.time()
        generator.append_tokens(input_tokens)
        generator.generate_next_token()
        ttft_end = time.time()

        first_token = generator.get_next_tokens()[0]
        response_tokens.append(first_token)
        self.profile["ttft_sec"] = f"{(ttft_end - ttft_start):.6f}"

        # === Generation Time: for tokens after the first ===
        total_gen_time = 0.0
        token_count_after_first = 0

        while not generator.is_done():
            token_start = time.time()
            generator.generate_next_token()
            token_end = time.time()

            token = generator.get_next_tokens()[0]
            response_tokens.append(token)

            total_gen_time += (token_end - token_start)
            token_count_after_first += 1

        decoded_tokens = [self._tokenizer_stream.decode(t) for t in response_tokens]
        response = "".join(decoded_tokens)
        total_tokens = len(response_tokens)

        # Safe TPS calculation
        if token_count_after_first > 0 and total_gen_time > 0:
            tps = token_count_after_first / total_gen_time
        else:
            tps = 0.0

        self.profile["generation_time_sec"] = round(total_gen_time, 6)
        self.profile["tps"] = round(tps, 2)
        self.profile["output_word_count"] = len(response.strip().split())
        self.profile["output_token_count"] = total_tokens

        return response.strip()

    def get_profile(self) -> Dict[str, Any]:
        return self.profile

    @property
    def _llm_type(self) -> str:
        return "onnx-llama-unified"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"model_name": "onnx-llama"}
