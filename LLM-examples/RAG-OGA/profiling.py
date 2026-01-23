import time
from typing import Dict
from statistics import mean

def run_profiling(llm, embed_model, retriever, chain, questions: Dict[str, str], runs: int = 5):
    summary = {}

    # Extract PromptTemplate from chain
    prompt_template = chain.first if hasattr(chain, "first") else None

    for qid, question in questions.items():
        llm_metrics = {
            "ttft_sec": [],
            "tps": [],
            "input_token_length": [],
            "output_token_count": []
        }
        embed_metrics = {
            "embedding_time_sec": [],
            "input_token_length": []
        }

        for _ in range(runs):
            # --- Embedding Pass ---
            if hasattr(embed_model, "embed_query"):
                start = time.time()
                tokens = embed_model.embed_query(question)
                end = time.time()
                if hasattr(embed_model, "get_profile"):
                    profile = embed_model.get_profile()
                    embed_metrics["embedding_time_sec"].append(
                        float(profile.get("embedding_time_sec", end - start))
                    )
                    embed_metrics["input_token_length"].append(
                        int(profile.get("input_token_length", len(tokens)))
                    )

            # --- Retrieve context chunks ---
            retrieved_docs = retriever.invoke(question)
            question_tokens = llm._tokenizer.encode(question)
            total_tokens = len(question_tokens) + 100  # buffer for formatting
            context_chunks = []
            for doc in retrieved_docs:
                doc_tokens = llm._tokenizer.encode(doc.page_content)
                if total_tokens + len(doc_tokens) <= 2048:
                    context_chunks.append(doc.page_content)
                    total_tokens += len(doc_tokens)
                else:
                    break
            context_str = "\n\n".join(context_chunks)

            
            input_dict = {"context": context_str, "question": question}
            if prompt_template:
                formatted_prompt_text = prompt_template.format(**input_dict)
            else:
                formatted_prompt_text = question

            # --- Tokenize final prompt ---
            prompt_tokens = llm._tokenizer.encode(formatted_prompt_text)
            llm_metrics["input_token_length"].append(len(prompt_tokens))

            # --- Run LLM ---
            chain.invoke(input_dict)

            # --- Collect profile
            llm_profile = llm.get_profile()
            llm_metrics["ttft_sec"].append(float(llm_profile.get("ttft_sec", 0)))
            llm_metrics["tps"].append(float(llm_profile.get("tps", 0)))
            llm_metrics["output_token_count"].append(int(llm_profile.get("output_token_count", 0)))

        summary[qid] = {
            "Avg Input Tokens": round(mean(llm_metrics["input_token_length"])),
            "Avg Output Tokens": round(mean(llm_metrics["output_token_count"])),
            "Avg TTFT(Sec)": round(mean(llm_metrics["ttft_sec"]), 6),
            "Avg TPS": round(mean(llm_metrics["tps"]), 2),
            
        }

    return summary

def print_profiling_summary(summary: Dict[str, Dict[str, float]]):
    print("\n--- Aggregated Profiling Summary ---")
    for qid, metrics in summary.items():
        print(f"\n{qid}:")
        for k, v in metrics.items():
            print(f"  {k:<30}: {v}")
