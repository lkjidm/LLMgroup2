import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Specify the folder containing the PDF files
KNOWLEDGE_FOLDER = "./knowledge"
VECTOR_DB_DIR = "./chroma_db"

def build_rag_from_folder():
    if not os.path.exists(KNOWLEDGE_FOLDER):
        os.makedirs(KNOWLEDGE_FOLDER)
        print(f"Created {KNOWLEDGE_FOLDER}. Place the PDF manuals in this directory.")
        return

    print(f"Scanning and reading PDF files in '{KNOWLEDGE_FOLDER}'...")

    # 1. Load all .pdf files in the directory with DirectoryLoader
    # glob="./*.pdf" matches PDF files; loader_cls selects PyPDFLoader for parsing
    loader = DirectoryLoader(
        KNOWLEDGE_FOLDER,
        glob="./*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True  # Show loading progress
    )
    
    pages = loader.load()
    print(f"Loaded the PDF files and extracted {len(pages)} pages of text.")

    if not pages:
        print("Warning: no PDF files were found in the directory.")
        return

    # 2. Split the documents into 500-character chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    docs = text_splitter.split_documents(pages)
    print(f"Text splitting completed: {len(docs)} document chunks created.")

    # 3. Embed the chunks and persist the vector store locally
    print("Loading the embedding model and creating the vector store...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=KNOWLEDGE_FOLDER
    )
    print(f"PDF documents embedded and saved to the vector store: {KNOWLEDGE_FOLDER}.")

if __name__ == "__main__":
    build_rag_from_folder()