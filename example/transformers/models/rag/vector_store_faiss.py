#
# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
#

import faiss, os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core import StorageContext, load_index_from_storage

class FaissEmbeddingManager:
    def __init__(self, data_directory=None, dimension=384, top_k=2, embedding_model=None):
        self.dimension = dimension
        self.data_directory = data_directory
        self.top_k = top_k
        self.embedding_model = embedding_model
        self.index = self.setup_index()
        
    def setup_index(self):
        storage_path = "storage-default"
        if os.path.exists(storage_path) and os.listdir(storage_path):
            print("Loading persisted index.")
            vector_storage = FaissVectorStore.from_persist_dir(storage_path)
            storage_ctx = StorageContext.from_defaults(vector_store=vector_storage, persist_dir=storage_path)
            return load_index_from_storage(storage_context=storage_ctx)
        else:
            print("Creating new index.")
            documents = SimpleDirectoryReader(self.data_directory).load_data()
            faiss_index = faiss.IndexFlatL2(self.dimension)
            vector_storage = FaissVectorStore(faiss_index=faiss_index)
            storage_ctx = StorageContext.from_defaults(vector_store=vector_storage)
            index = VectorStoreIndex.from_documents(documents=documents, storage_context=storage_ctx)
            index.storage_context.persist(persist_dir=storage_path)
            return index
        
    def get_query_engine(self, top_k=None):
        if top_k is None:
            top_k = self.top_k
        return self.index.as_query_engine(similarity_top_k=top_k)
