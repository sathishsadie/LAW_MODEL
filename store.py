import os
import json
from pathlib import Path
# import faiss
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings  # ✅ Updated import
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# File paths and settings
JSON_FILE = "bns_6_48 (2).json"  
VECTOR_DB_FOLDER = "bns_vector_db"
VECTOR_DB_FILE = "legal_data.faiss"

# Ensure vector storage folder exists
os.makedirs(VECTOR_DB_FOLDER, exist_ok=True)

# ✅ Use updated Ollama Embeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")

# ✅ Define Text Splitter
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

def load_json(json_file):
    """ Load JSON file containing legal text data """
    with open(json_file, "r", encoding="utf-8") as file:
        return json.load(file)

def clean_and_process_text(text_list):
    """
    ✅ Remove extra new lines, replace with space, and clean text.
    """
    cleaned_text = " ".join(text_list)  
    cleaned_text = " ".join(cleaned_text.split())  
    return cleaned_text

def split_and_store(json_data, vector_db_path):
    """ ✅ Split text into chunks (size=300, overlap=50) and store in FAISS """
    documents = []

    for section, texts in json_data.items():
        cleaned_text = clean_and_process_text(texts) 
        chunks = splitter.split_text(cleaned_text)  

        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    "section": section, 
                    "index": i, 
                    "doc_id": f"{section}_{i}"  # Unique identifier
                }
            )
            documents.append(doc)

    # ✅ Store in FAISS
    vector_store = FAISS.from_documents(documents, embeddings)
    vector_store.save_local(str(vector_db_path))
    print(f"✅ FAISS index successfully created at: {vector_db_path}")

# ✅ Run the Pipeline
if __name__ == "__main__":
    print("🚀 Loading JSON data...")
    extracted_data = load_json(JSON_FILE)
    print("Json data loaded")
    split_and_store(extracted_data, Path(VECTOR_DB_FOLDER) / VECTOR_DB_FILE)
    print("✅ FAISS index successfully stored with metadata!")
