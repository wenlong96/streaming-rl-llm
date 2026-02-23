import os
import shutil
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

DB_DIR = "./chroma_db"

# Wipe existing ChromaDB
if os.path.exists(DB_DIR):
    shutil.rmtree(DB_DIR)
    print(f"Cleared existing ChromaDB at {DB_DIR}")

# # Load resume PDF
# print("Loading resume PDF")
# pdf_loader = PyPDFLoader("resume.pdf")
# pdf_pages = pdf_loader.load()

# Load rich context text file
print("Loading will_context.txt")
text_loader = TextLoader("will_context.txt", encoding="utf-8")
text_pages = text_loader.load()

# Combine all documents
# all_docs = pdf_pages + text_pages
all_docs = text_pages
print(f"Loaded {len(all_docs)} documents total")

# Split into chunks
print("Splitting documents")
# splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
# chunks = splitter.split_documents(all_docs)
chunks = all_docs
print(f"Split into {len(chunks)} chunks")

# Initialize Embedding Model
print("Loading embedding model")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Build and Save the ChromaDB Vector Store
print("Saving to ChromaDB")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=DB_DIR
)

print(f"Vectorized and saved to {DB_DIR}")

query = "What is Will's experience with AI?"
print(f"\nTesting Retrieval for: '{query}'")
results = vectorstore.similarity_search(query, k=4)
for i, doc in enumerate(results):
    print(f"  - Result {i+1}: {doc.page_content[:150]}")