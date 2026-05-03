import os
import sys
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ---------------------------------------------------------
# 1. Configuration — API key from environment only
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VECTORSTORE_PATH = os.path.join(BASE_DIR, "vectorstore")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    print("❌ Error: GROQ_API_KEY not found.")
    print("   Set it with: export GROQ_API_KEY='your_key_here'")
    print("   Or in Colab: os.environ['GROQ_API_KEY'] = 'your_key_here'")
    sys.exit(1)

# ---------------------------------------------------------
# 2. Embeddings
# ---------------------------------------------------------
print("🔧 Loading embeddings...")
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

# ---------------------------------------------------------
# 3. Load Vectorstore
# ---------------------------------------------------------
if not os.path.exists(VECTORSTORE_PATH):
    print("❌ Vectorstore not found. Run ingest.py first.")
    sys.exit(1)

vectordb = Chroma(
    persist_directory=VECTORSTORE_PATH,
    embedding_function=embeddings
)
print("✅ Vectorstore loaded.")

# ---------------------------------------------------------
# 4. Groq LLM
# ---------------------------------------------------------
print("🧠 Connecting to Groq...")
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=GROQ_API_KEY,
    temperature=0
)

# ---------------------------------------------------------
# 5. RAG Prompt
# ---------------------------------------------------------
template = """You are a University Assistant. Use ONLY the following
pieces of context to answer the question.
If the answer is not in the context, say exactly:
"I don't know based on the provided knowledge base."

When relevant, mention specific Rule IDs from the context.
Answer in clear, direct language.

Context:
{context}

Question: {question}

Answer:"""

QA_PROMPT = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)

# ---------------------------------------------------------
# 6. RAG Chain
# ---------------------------------------------------------
retriever = vectordb.as_retriever(search_kwargs={"k": 5})

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | QA_PROMPT
    | llm
    | StrOutputParser()
)

# ---------------------------------------------------------
# 7. Ask function with citations
# ---------------------------------------------------------
def ask_question(query):
    print(f"\n🔍 Searching knowledge base...")
    try:
        answer = rag_chain.invoke(query).strip()
        source_docs = retriever.invoke(query)

        print(f"\n📝 ANSWER:\n{answer}")

        if "I don't know" not in answer:
            print("\n📚 SOURCES:")
            unique_sources = set(
                doc.metadata.get('source', 'Unknown')
                for doc in source_docs
            )
            for idx, source in enumerate(unique_sources, 1):
                print(f"  {idx}. {source}")
        else:
            print("\nℹ️ Context was found but didn't contain the answer.")

    except Exception as e:
        print(f"❌ Error: {e}")

# ---------------------------------------------------------
# 8. Main loop
# ---------------------------------------------------------
if __name__ == "__main__":
    print("\n✨ University RAG Assistant Ready")
    print("Type your question (or 'exit' to quit):\n")

    while True:
        user_input = input("❓ Question: ")
        if user_input.lower() in ['exit', 'quit']:
            print("Goodbye.")
            break
        if user_input.strip():
            ask_question(user_input)