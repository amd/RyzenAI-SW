# gradio_app.py

import os
import gradio as gr
from glob import glob
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader
from langchain_core.documents import Document
from langchain_text_splitters import TokenTextSplitter
from custom_llm.custom_llm import custom_llm
from custom_embedding.custom_embedding import custom_embeddings
# from custom_embedding.huggingface_bge_profile import HuggingFaceBGEWithProfile
from langchain_community.embeddings import HuggingFaceEmbeddings

def gradio_launch_app(dataset_path, model_path, use_npu_model_path="bge-large-en-v1.5.onnx"):
    faiss_index_path = "faiss_index"
    llm = custom_llm(model_path=model_path)
    embedding_model = None
    retriever = None

    retrieval_template = PromptTemplate.from_template(
        """<|system|> Using the information contained in the context, give a comprehensive answer to the question. Respond only to the question asked, response should be concise and relevant to the question. Provide the number of the source document when relevant.</s> <|user|> Context: {context} --- Now here is the question you need to answer. Question: {question}</s> <|assistant|>"""
    )
    
    direct_template = PromptTemplate.from_template(
        """Answer the following question clearly and accurately in one paragraph.\nQuestion: {question}\nAnswer:"""
    )

    def setup_vectorstore():
        nonlocal embedding_model, retriever
        # embedding_model = (
        #     custom_embeddings(model_path=use_npu_model_path, tokenizer_name="BAAI/bge-large-en-v1.5")
        #     if use_npu else HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5",encode_kwargs={"normal_embeddings": True})
        # )
        embedding_model = custom_embeddings(model_path=use_npu_model_path, tokenizer_name="BAAI/bge-large-en-v1.5")
        
        if os.path.exists(os.path.join(faiss_index_path, "index.faiss")):
            vectorstore = FAISS.load_local(
                folder_path=faiss_index_path,
                embeddings=embedding_model,
                allow_dangerous_deserialization=True
            )
        else:
            documents = []
            for ext in [".pdf", ".txt", ".md", ".docx", ".rst"]:
                for file_path in glob(os.path.join(dataset_path, f"*{ext}")):
                    try:
                        loader = {
                            ".pdf": PyPDFLoader,
                            ".txt": lambda p: TextLoader(p, encoding="utf-8"),
                            ".md": lambda p: TextLoader(p, encoding="utf-8"),
                            ".rst": lambda p: TextLoader(p, encoding="utf-8"),
                            ".docx": UnstructuredWordDocumentLoader,
                        }[ext](file_path)
                        pages = loader.load()
                        content = "\n".join([p.page_content for p in pages])
                        documents.append(Document(page_content=content, metadata={"source": file_path}))
                    except Exception as e:
                        print(f"Failed to load {file_path}: {e}")
            splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=50)
            split_docs = splitter.split_documents(documents)
            vectorstore = FAISS.from_documents(split_docs, embedding_model)
            vectorstore.save_local(faiss_index_path)
        retriever = vectorstore.as_retriever(search_type='similarity',search_kwargs={"k": 3})

    # def answer_query(query, use_direct_llm, use_npu):
    def answer_query(query, use_direct_llm):  
        nonlocal retriever, embedding_model
        if not query.strip():
            return "Please enter a valid question."
        if retriever is None or embedding_model is None:
            setup_vectorstore()
        if not use_direct_llm:
            retrieved_docs = retriever.invoke(query)
            question_tokens = llm._tokenizer.encode(query)
            context_chunks = []
            total_tokens = len(question_tokens) + 100
            for doc in retrieved_docs:
                doc_tokens = llm._tokenizer.encode(doc.page_content)
                if total_tokens + len(doc_tokens) <= 2048:
                    context_chunks.append(doc.page_content)
                    total_tokens += len(doc_tokens)
            if not context_chunks:
                return "No relevant context found."
            return (retrieval_template | llm).invoke({"context": "\n\n".join(context_chunks), "question": query})
        return (direct_template | llm).invoke({"question": query})

    def checkbox_handler(use_direct_llm,use_npu):
        if use_direct_llm and use_npu:
            return False, True
        return use_direct_llm,use_npu
    with gr.Blocks(title="SmartAsk", css="""#submit-btn { background-color: #ff6a00; color: white; font-weight: bold; width: 150px; margin: 16px auto; }""") as demo:
        gr.Markdown("<div style='text-align:center'><h2>RAG: SmartAsk</h2><p>Ask questions using your document dataset.</p></div>")
        with gr.Row():
            with gr.Column(scale=1):
                use_direct_llm = gr.Checkbox(label="Use Direct LLM (Skip Retrieval)",interactive=True)
                # use_npu = gr.Checkbox(label="Use NPU for Embedding Model",interactive=True)
            with gr.Column(scale=4):
                query = gr.Textbox(label="Enter your question", lines=2)
                submit_btn = gr.Button("Submit", elem_id="submit-btn")
                answer = gr.Textbox(label="Answer", lines=12)
        # use_direct_llm.change(fn=checkbox_handler,inputs=[use_direct_llm,use_npu],outputs=[use_direct_llm,use_npu]) 
        # use_npu.change(fn=checkbox_handler,inputs=[use_direct_llm,use_npu],outputs=[use_direct_llm,use_npu])       
        # submit_btn.click(fn=answer_query, inputs=[query, use_direct_llm, use_npu], outputs=answer)
        submit_btn.click(fn=answer_query, inputs=[query, use_direct_llm], outputs=answer)
    demo.launch()

