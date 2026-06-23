import streamlit as st
from sentence_transformers import SentenceTransformer
from groq import Groq
import chromadb
from dotenv import load_dotenv
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# Page config
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="centered")
st.title("🤖 RAG Document Chatbot")
st.caption("Upload a PDF and ask questions about it")

# Initialize models once
@st.cache_resource
def load_models():
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return embed_model, groq_client

embed_model, groq_client = load_models()

# Session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "collection" not in st.session_state:
    st.session_state.collection = None
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

# Sidebar — PDF uploader
with st.sidebar:
    st.header("📄 Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file:
        if uploaded_file.name != st.session_state.pdf_name:
            with st.spinner("Processing PDF..."):

                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                # Load and split PDF
                loader = PyPDFLoader(tmp_path)
                pages = loader.load()

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500,
                    chunk_overlap=50
                )
                chunks = splitter.split_documents(pages)
                chunk_texts = [c.page_content for c in chunks]

                # Embed and store in ChromaDB
                embeddings = embed_model.encode(chunk_texts).tolist()
                chroma_client = chromadb.Client()
                collection = chroma_client.get_or_create_collection(name="pdf_docs")
                collection.add(
                    documents=chunk_texts,
                    embeddings=embeddings,
                    ids=[f"chunk_{i}" for i in range(len(chunk_texts))]
                )

                st.session_state.collection = collection
                st.session_state.pdf_name = uploaded_file.name
                st.session_state.chat_history = []
                os.unlink(tmp_path)

            st.success(f"✅ {uploaded_file.name} processed!")
            st.info(f"📊 {len(chunks)} chunks created")

    if st.session_state.pdf_name:
        st.divider()
        st.caption(f"Active: {st.session_state.pdf_name}")

    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

# RAG answer function
def get_answer(question, collection):
    # Retrieve relevant chunks
    question_embedding = embed_model.encode([question]).tolist()
    results = collection.query(
        query_embeddings=question_embedding,
        n_results=3
    )
    chunks = results['documents'][0]
    context = "\n".join([f"- {c}" for c in chunks])

    # Build prompt
    prompt = f"""Answer the question using ONLY the information below.
If the answer is not found, say "I couldn't find that in the document."

Context:
{context}

Question: {question}
Answer:"""

    # Get answer from Groq
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response.choices[0].message.content
    sources = chunks

    return answer, sources

# Display chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg:
            with st.expander("📚 Sources used"):
                for i, src in enumerate(msg["sources"]):
                    st.caption(f"Chunk {i+1}: {src[:200]}...")

# Chat input
if st.session_state.collection is None:
    st.info("👈 Upload a PDF from the sidebar to start chatting")
else:
    question = st.chat_input("Ask a question about your document...")

    if question:
        # Show user message
        st.session_state.chat_history.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.write(question)

        # Get and show answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer, sources = get_answer(question, st.session_state.collection)
            st.write(answer)
            with st.expander("📚 Sources used"):
                for i, src in enumerate(sources):
                    st.caption(f"Chunk {i+1}: {src[:200]}...")

        # Save to history
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })