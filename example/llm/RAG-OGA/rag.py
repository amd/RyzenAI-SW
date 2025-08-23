import os
import warnings
import argparse
from glob import glob
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader
from langchain_core.documents import Document
from langchain_text_splitters import TokenTextSplitter
from custom_llm.custom_llm import custom_llm
from custom_embedding.custom_embedding import custom_embeddings
# from custom_embedding.huggingface_bge_profile import HuggingFaceBGEWithProfile
from profiling import run_profiling, print_profiling_summary
from gradio_app import gradio_launch_app
from langchain_community.embeddings import HuggingFaceEmbeddings

warnings.filterwarnings("ignore", category=DeprecationWarning)

# --- Argument Parser ---
parser = argparse.ArgumentParser(description="Run LLM with or without retrieval")
parser.add_argument("--direct_llm", action="store_true", help="Skip retrieval and send query directly to LLM")
# parser.add_argument("--embed_npu", action="store_true", help="Use ONNX embedding model on VitisAI NPU")
parser.add_argument("--profiling", action="store_true", help="Enable profiling: prints TTFT, TPS, tokens, and word counts")
parser.add_argument("--gradio", action="store_true", help="Launch Gradio web app")
args = parser.parse_args()


if args.gradio:
    gradio_launch_app(dataset_path=r"./Dataset", model_path= r"C:\huggingface") #update this path
    exit()

# --- Paths ---

dataset_path = r"./Dataset" #update this path
faiss_index_path = "faiss_index"

# --- Embedding Model ---
print("Using ONNX embedding model on VitisAI NPU...")
embedding_model = custom_embeddings(model_path="./custom_embedding/bge-large-en-v1.5.onnx", tokenizer_name="BAAI/bge-large-en-v1.5")

# --- Load or Build FAISS Index ---
if os.path.exists(os.path.join(faiss_index_path, "index.faiss")) and os.path.exists(os.path.join(faiss_index_path, "index.pkl")):
    vectorstore = FAISS.load_local(folder_path=faiss_index_path, embeddings=embedding_model, allow_dangerous_deserialization=True)
else:
    print("Loading files and building index...")
    supported_exts = [".pdf", ".txt", ".md", ".docx", ".rst"]
    documents = []
    for ext in supported_exts:
        for file_path in glob(os.path.join(dataset_path, f"*{ext}")):
            try:
                loader = {
                    ".pdf": PyPDFLoader,
                    ".txt": lambda p: TextLoader(p, encoding="utf-8"),
                    ".md": lambda p: TextLoader(p, encoding="utf-8"),
                    ".rst": lambda p: TextLoader(p, encoding="utf-8"),
                    ".docx": UnstructuredWordDocumentLoader
                }[ext](file_path)
                pages = loader.load()
                full_text = "\n".join(page.page_content for page in pages)
                documents.append(Document(page_content=full_text, metadata={"source": file_path}))
            except Exception as e:
                print(f"Failed to load {file_path}: {e}")
    print(f"Loaded {len(documents)} documents.")
    splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=50)
    split_docs = splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(split_docs, embedding_model)
    vectorstore.save_local(faiss_index_path)
    print("FAISS index saved to disk.")

retriever = vectorstore.as_retriever(search_type='similarity',search_kwargs={"k": 3})
print("Number of vectors:", vectorstore.index.ntotal)

llm = custom_llm(model_path=r"C:\huggingface")   #update this path
# llm = custom_llm(model_path=r"C:\Users\akumar23\RAG-repo-xilinx\model")   # Example 

template = PromptTemplate.from_template("""<|system|>
Using the information contained in the context,
give a comprehensive answer to the question.
Respond only to the question asked, response should be concise and relevant to the question.
Provide the number of the source document when relevant.</s>
<|user|>
Context:
{context}
---
Now here is the question you need to answer.
Question: {question}</s>
<|assistant|>""")


chain = template | llm

if args.profiling:
    questions = {
        "Q1": "In the context of Ryzen AI Software's hybrid inference model, how does the integration of automated operator assignment, encrypted context caching, and hardware-specific xclbin configurations collectively contribute to optimizing performance, ensuring security, and minimizing compilation overhead across varying model types such as transformers and CNNs?",
        "Q2": "In the context of model quantization using AMD Quark, how do different frameworks vary in capabilities, deployment support, and quantization strategies across platforms and formats?",
        "Q3": "What specialized hardware component, inspired by the brain's neural architecture, is designed to perform parallel processing of AI workloads with low precision arithmetic and high energy efficiency?"
    }
    print("\nRunning profiling on predefined questions...\n")
    results = run_profiling(llm, embedding_model, retriever, chain, questions, runs=1)
    print_profiling_summary(results)
    exit()

query = input("\nEnter your question: ")

if not args.direct_llm:
    print("Retrieval mode is on.\nLoading existing FAISS index from disk...")
    retrieved_docs = retriever.invoke(query)
    question_tokens = llm._tokenizer.encode(query)
    max_total_tokens = 2048
    buffer_tokens = 100
    context_chunks, total_tokens = [], len(question_tokens) + buffer_tokens
    for doc in retrieved_docs:
        doc_tokens = llm._tokenizer.encode(doc.page_content)
        if total_tokens + len(doc_tokens) <= max_total_tokens:
            context_chunks.append(doc.page_content)
            total_tokens += len(doc_tokens)
        else:
            break
    context_str = "\n\n".join(context_chunks)
    response = chain.invoke({"context": context_str, "question": query})
else:
    direct_template = PromptTemplate.from_template("""Answer the following question clearly and accurately in one paragraph.\nQuestion: {question}\nAnswer:""")
    chain = direct_template | llm
    print("\nDirect_llm mode is on. No retrieval has been performed.")
    response = chain.invoke({"question": query})

print("\nAnswer:\n", response)
