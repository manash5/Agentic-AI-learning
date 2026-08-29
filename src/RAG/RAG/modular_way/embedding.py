from typing import Any, List

from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np
from sentence_transformers import SentenceTransformer

from RAG.modular_way.data_loader import load_all_documents


class EmbeeddingPipeline: 
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", chunk_size: int = 1000, chunk_overlap: int = 200): 
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = SentenceTransformer(model_name)
        print(f"[INFO] Loaded Embedding model : {model_name}")\


    def chunk_document(self, documents: List[Any])-> List[Any]: 
        splitter = RecursiveCharacterTextSplitter(
            chunk_size = self.chunk_size, 
            chunk_overlap = self.chunk_overlap, 
            length_function = len, 
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(documents)
        print(f"[INFO] split {len(documents)} documents into {len(chunks)} chunks")
        return chunks 


    def embed_chunks(self, chunks: List[Any]) -> np.ndarray: 
        texts = [chunk.page_content for chunk in chunks]
        print(f"[INFO] Generating embeddings for {len(texts)} chunks")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        print(f"[INFO] Embeddings shape: {embeddings.shape}")
        return embeddings


# Exmpale usage 
if __name__ == "__main__": 
    docs = load_all_documents("src/RAG/data")
    emb_pipe = EmbeeddingPipeline()
    chunks = emb_pipe.embed_chunks(docs)
    embeddings = emb_pipe.chunks(chunks)
    print("[INFO] Example embeddings:", embeddings[0] if len(embeddings) > 0 else None)