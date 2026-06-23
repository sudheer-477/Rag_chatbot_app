# RAG Chatbot App 

A Retrieval-Augmented Generation (RAG) chatbot built with Streamlit, ChromaDB, and Groq LLM. Upload PDF documents and chat with them intelligently.

##  Live Demo

https://ragchatbotapp-sudheer.streamlit.app/
---

##  Features

-  Upload and parse PDF documents
-  Semantic search using Sentence Transformers
-  RAG pipeline with ChromaDB vector store
-  Fast LLM responses powered by Groq
-  Clean interactive UI built with Streamlit

---

##  Tech Stack

| Component        | Technology                          |
|-----------------|--------------------------------------|
| Frontend        | Streamlit                            |
| LLM             | Groq (`llama3-8b-8192`)              |
| Embeddings      | Sentence Transformers (`all-MiniLM-L6-v2`) |
| Vector Store    | ChromaDB                             |
| PDF Parsing     | PyPDF                                |
| Text Splitting  | LangChain Text Splitters             |

---

##  Project Structure

```
rag_chatbot_app/
├── rag_chatbot_app.py      # Main Streamlit app
├── requirements.txt       
├── .env                    # Local environment variables (not committed)
├── .gitignore
└── README.md
```

---

## Local Setup

### 1. Clone the repository
```bash
git clone https://github.com/sudheer-477/Rag_chatbot_app.git
cd Rag_chatbot_app
```

### 2. Install dependencies
```bash
pip install -r requirements-local.txt
```

### 3. Set up environment variables
Create a `.env` file in the root directory:
```
GROQ_API_KEY=your_groq_api_key_here
```
Get your free Groq API key at [console.groq.com](https://console.groq.com)

### 4. Run the app
```bash
streamlit run rag_chatbot_app.py
```

---

## ☁️ Deploying to Streamlit Cloud

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your repo
3. Add your secret under **Manage app → Settings → Secrets**:
```toml
GROQ_API_KEY = "your_groq_api_key_here"
```
4. Click **Deploy**

---

##  Environment Variables

| Variable       | Description              | Required |
|---------------|--------------------------|----------|
| `GROQ_API_KEY` | Groq API key for LLM     |  Yes   |

---

##  Usage

1. Open the app in your browser
2. Upload a PDF document using the file uploader
3. Wait for the document to be processed and indexed
4. Type your question in the chat input
5. Get AI-powered answers based on your document

---

##  Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change.

---

##  Author

**Sudheer**  
GitHub: [@sudheer-477](https://github.com/sudheer-477)