from typing import List, Dict, Any
from langchain_core.embeddings import Embeddings
import onnxruntime as ort
from transformers import AutoTokenizer
import numpy as np
import time

class custom_embeddings(Embeddings):
    def __init__(self, model_path: str, tokenizer_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.session = ort.InferenceSession(
            model_path,
            providers=["VitisAIExecutionProvider"],
            provider_options=[{
                "config_file": "custom_embedding/vaiml_config.json",
                "cache_dir": "./",
                "cacheKey": "modelcachekey_bge"
            }]
        )
        print("NPU session created successfully.")
        self.profile: Dict[str, Any] = {}

    def _embed(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        total_input_tokens = 0
        start = time.time()

        for text in texts:
            inputs = self.tokenizer(
                text,
                max_length=512,
                padding="max_length",
                truncation=True,
                return_tensors="np",
                return_token_type_ids=False
            )
            input_ids = inputs["input_ids"]
            total_input_tokens += np.count_nonzero(input_ids)
            onnx_inputs = {
                "input_ids": input_ids.astype(np.int64),
                "attention_mask": inputs["attention_mask"].astype(np.int64)
            }

            outputs = self.session.run(None, onnx_inputs)
            embedding = outputs[1][0]  # pooler_output
            embeddings.append(embedding.tolist())

        end = time.time()

        self.profile["embedding_time_sec"] = round(end - start, 4)
        self.profile["input_token_length"] = total_input_tokens
        return embeddings
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._embed([text])[0]

    def get_profile(self) -> Dict[str, Any]:
        return self.profile
