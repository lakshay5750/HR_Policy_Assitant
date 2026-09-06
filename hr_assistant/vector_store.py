"""Step 4: store chunk embeddings in Qdrant Cloud so we can search them later."""
import os
from langchain_community.vectorstores import FAISS


from hr_assistant import config
from hr_assistant.embeddings import get_embeddings_model


def build_vector_store(chunks):
    """Embed every chunk and build the faiss index in memory"""
    embeddings_model=get_embeddings_model()
    return FAISS.from_documents(chunks,embeddings_model)


def save_vector_store(vector_store,path:str=config.VECTOR_FILE_PATH)->None:
    """Save faiss index to local.we don't rebuild every time"""
    vector_store.save_local(path)
    
def load_vector_store(path:str=config.VECTOR_FILE_PATH):
    """Load the faiss index from the local"""
    embeddings_model=get_embeddings_model()
    return FAISS.load_local(path,embeddings_model,allow_dangerous_deserialization=True)

def vector_store_exists(path:str=config.VECTOR_FILE_PATH)->bool:
    """Check if a saved faiss index already exists on disk"""
    return os.path.exists(os.path.join(path,"faiss_index"))

def get_retriever(vector_store,k:int=config.TOP_K_RESULTS):
    """Turn the vector store into the retriever that turn the top-k matching chunks"""
    return vector_store.as_retriever(search_kwargs={"k":k})