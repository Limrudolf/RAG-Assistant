import os
import shutil
import pickle
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.retrievers import ParentDocumentRetriever 
from langchain_classic.storage import InMemoryStore
from dotenv import load_dotenv

# --- CONFIGURATION ---
PDF_FILE = "competitive_programming.pdf"
VECTOR_STORE_PATH = "faiss_index"
DOC_STORE_FILE = "docstore.pkl"

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

def create_vector_db():
    print(f"--- Started Processing {PDF_FILE} ---")

    if os.path.exists(VECTOR_STORE_PATH): shutil.rmtree(VECTOR_STORE_PATH)
    if os.path.exists(DOC_STORE_FILE): os.remove(DOC_STORE_FILE)

    if not os.path.exists(PDF_FILE):
        print(f"Error: {PDF_FILE} not found.")
        return

    loader = PyPDFLoader(PDF_FILE)
    raw_documents = loader.load()
    print(f"Loaded {len(raw_documents)} pages.")

    # Parent: Large chunks (The context the LLM reads)
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=300)
    
    # Child: Small chunks (The "Search Terms" for the Index)
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400, 
        chunk_overlap=50,
        separators=["\n\n", "\n", r"(?<=\. )", " ", ""]
    )

    print("Generating Embeddings...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    
    # A. Vector Store (Children): Initialize empty
    vectorstore = FAISS.from_texts(["start"], embeddings)
    
    # B. Doc Store (Parents): Initialize on disk
    store = InMemoryStore()

    print("Ingesting documents (Splitting & Linking)")
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )
    
    retriever.add_documents(raw_documents)
    
    vectorstore.save_local(VECTOR_STORE_PATH)
    with open(DOC_STORE_FILE, "wb") as f:
        pickle.dump(store, f)
    print("--- Success! Parent-Child Index Created ---")

if __name__ == "__main__":
    create_vector_db()