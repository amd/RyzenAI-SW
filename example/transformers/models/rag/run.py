#
# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
#

import argparse
import psutil
import os
import os
import numpy as np
import time
import argparse
import gradio as gr
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from custom_llm import OurLLM
from vector_store_faiss import FaissEmbeddingManager
from transformers import set_seed


supported_models = ["llama-2-7b-chat",]

if __name__ == "__main__":

    import transformers
    tv = transformers.__version__
    tv = tv.split(".")


    set_seed(123)

    parser = argparse.ArgumentParser(description='AMD Chatbot Parameters')
    parser.add_argument("--model_name", help="mode name", type=str, default="llama-2-7b-chat", choices=supported_models)
    parser.add_argument("--target", help="cpu, aie", type=str, default="aie", choices=["cpu", "aie"])
    parser.add_argument('--precision', help="w4abf16 - used for awq, awqplus, pergrp & bf16 runs on cpu", type=str, default="w4abf16", choices=["bf16", "w4abf16"])
    parser.add_argument('--profilegemm', help="Log matmul times for prompt and token phases - supported only for AIE target", action='store_true')
    parser.add_argument("--w_bit", help="3, 4", type=int, default=4, choices=[3, 4])
    parser.add_argument("--group_size", help="128 default", type=int, default=128, choices=[32, 128])
    parser.add_argument("--algorithm", help="awq, awqplus, pergrp", type=str, default="awq", choices=["awq", "awqplus", "pergrp"])
    parser.add_argument("--direct_llm", help="Run query through query engine(--no-direct_llm) or llm directly(--direct_llm).", required=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--quantized", help="Run quantized(--quantized) or original model(--no-quantized).", required=True, action=argparse.BooleanOptionalAction)
    #speculative decoding
    parser.add_argument(
        "--assisted_generation", help="Enable assisted generation", action="store_true"
    )
    
    args = parser.parse_args()
    print(f"{args}")
    
    dev = os.getenv("DEVICE")
    if dev == "stx":
        fast_attention = False
        fast_mlp = False
        flash_attention_plus = True
    else:
        fast_attention = False # phx
        fast_mlp = False
        flash_attention_plus = True

    try:
        Settings.llm  = OurLLM(target=args.target, model_name=args.model_name, quantized=args.quantized, algorithm=args.algorithm, w_bit=args.w_bit, group_size=args.group_size, flash_attention_plus=flash_attention_plus, fast_mlp=fast_mlp, fast_attention=fast_attention, precision=args.precision, profilegemm=args.profilegemm, assisted_generation=args.assisted_generation)
    except Exception as e:
        print(e)

    #embedding model
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    # vector store
    data_dir = "./dataset"
    embeddding_dim = 384
    faiss_storage = FaissEmbeddingManager(data_directory=data_dir, dimension=embeddding_dim, embedding_model=Settings.embed_model)

    # Query and print response
    query_engine = faiss_storage.get_query_engine(top_k=2)

    # prompts
    def prompt(query_text, direct_llm, history=None):
        if args.direct_llm:
            res = Settings.llm.complete(query_text)
        else:
            import re
            response_str = query_engine.query(query_text)
            match = re.search(r'Answer: (.+)', str(response_str.response))
            if match:
                res = match.group(1)
            else:
                res = response_str.response
        return str(res)

    # answer = prompt(query_text="What is Vitis AI?", direct_llm=args.direct_llm)
    # print(answer)
    
    # Gradio UI setup
    interface = gr.ChatInterface(fn=prompt, title=" Rag Chat with Llama2 quantized on AIE", description="Ask me anything!")
    interface.launch(server_name="localhost")
   
