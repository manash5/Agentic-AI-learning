from RAG.modular_way.search import RAGSearch
from RAG.modular_way.vectorstore import FaissVectorStore
from src.RAG.modular_way.data_loader import load_all_documents


## Example usage 
if __name__ == "__main__": 
    docs = load_all_documents("src/RAG/data")
    store = FaissVectorStore("faiss_store")
    store.build_from_documents(docs)
    store.load()
    #print(store.query("What is attention mechanism?", top_k=3))
    rag_search = RAGSearch()
    query = "What is attention mechanism?"
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("Summary:", summary)