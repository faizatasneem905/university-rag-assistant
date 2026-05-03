import streamlit as st
import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="University RAG Assistant",
    page_icon="🎓",
    layout="centered"
)

# ---------------------------------------------------------
# Auto-build vectorstore if it doesn't exist
# This runs on first deployment to Streamlit Cloud
# ---------------------------------------------------------
VECTORSTORE_PATH = "vectorstore"

if not os.path.exists(VECTORSTORE_PATH):
    with st.spinner("⚙️ Building knowledge base for first time... (takes ~30 seconds)"):
        from ingest import build_vectorstore
        build_vectorstore()
    st.success("✅ Knowledge base ready!")
    st.rerun()

# ---------------------------------------------------------
# Initialize RAG System — cached so it loads only once
# ---------------------------------------------------------
@st.cache_resource
def load_rag_system():
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

    vectordb = Chroma(
        persist_directory=VECTORSTORE_PATH,
        embedding_function=embeddings
    )

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=os.environ.get("GROQ_API_KEY"),
        temperature=0
    )

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

    retriever = vectordb.as_retriever(search_kwargs={"k": 5})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | QA_PROMPT
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever


# ---------------------------------------------------------
# UI
# ---------------------------------------------------------
st.title("🎓 University RAG Assistant")
st.markdown("Ask questions about university policies, grading, attendance, and academic rules.")
st.divider()

# Load system
with st.spinner("Loading knowledge base..."):
    rag_chain, retriever = load_rag_system()

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input
if query := st.chat_input("Ask about university policies..."):

    # Show user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):

            answer = rag_chain.invoke(query).strip()
            source_docs = retriever.invoke(query)

            st.markdown(answer)

            if "I don't know" not in answer:
                unique_sources = list(set(
                    doc.metadata.get('source', 'Unknown')
                    for doc in source_docs
                ))
                with st.expander("📚 Sources"):
                    for source in unique_sources:
                        st.write(f"• {source}")

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })