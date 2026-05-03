import os
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# ------------------------
# Paths & Configuration
# ------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KB_FOLDER = os.path.join(BASE_DIR, "knowledge_base")
VECTORSTORE_PATH = os.path.join(BASE_DIR, "vectorstore")

md_files = ["handbook.md", "operations.md"]
csv_file = "rules.csv"


def build_vectorstore():
    """
    Callable function for Streamlit Cloud deployment.
    Builds and persists the vectorstore from the knowledge base.
    Uses relative paths so it works both locally and on Streamlit Cloud.
    """
    # ------------------------
    # Step 1: Loading documents
    # ------------------------
    print("📂 Loading documents...")
    docs = []

    # Load Markdown files
    for md in md_files:
        md_path = os.path.join(KB_FOLDER, md)
        if os.path.exists(md_path):
            with open(md_path, "r", encoding="utf-8") as f:
                text = f.read()
                docs.append(Document(page_content=text, metadata={"source": md}))
        else:
            print(f"Warning: {md_path} not found.")

    # Load CSV file
    csv_path = os.path.join(KB_FOLDER, csv_file)
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            content = (
                f"Rule ID: {row['id']}\n"
                f"Topic: {row['topic']}\n"
                f"Rule/Fact: {row['rule_or_fact']}\n"
                f"Notes/Exceptions: {row['notes_or_exceptions']}"
            )
            docs.append(Document(
                page_content=content,
                metadata={"source": f"{csv_file} (Topic: {row['topic']})"}
            ))
    else:
        print(f"Warning: {csv_path} not found.")

    if not docs:
        raise ValueError("No documents loaded. Check your knowledge_base folder.")

    # ------------------------
    # Step 2: Chunking
    # ------------------------
    print("✂️ Chunking documents...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    chunked_docs = text_splitter.split_documents(docs)

    # ------------------------
    # Step 3: Embeddings
    # ------------------------
    print("🧠 Generating embeddings (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

    # ------------------------
    # Step 4: Chroma vectorstore
    # ------------------------
    print(f"💾 Saving vectorstore to {VECTORSTORE_PATH}...")
    vectordb = Chroma.from_documents(
        documents=chunked_docs,
        embedding=embeddings,
        persist_directory=VECTORSTORE_PATH
    )

    print(f"✅ Success! Created {len(chunked_docs)} chunks from {len(docs)} documents.")
    print(f"🚀 Vectorstore ready at '{VECTORSTORE_PATH}'")
    return vectordb


# ------------------------
# Run directly if needed
# ------------------------
if __name__ == "__main__":
    build_vectorstore()