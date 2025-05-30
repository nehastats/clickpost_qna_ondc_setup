# ONDC QA System using RAG (LLM + FAISS + Embeddings)

This repository implements a `Retrieval-Augmented Generation (RAG)` pipeline to perform Question Answering (QA) over ONDC policy documents using:
- SentenceTransformer (`all-MiniLM-L6-v2`) for embedding
- FAISS for vector search
- Ollama's `llama3.2` for LLM-based answer generation
- Streamlit for an interactive chat interface

## Design Architecture:
```bash
[PDF Upload via Streamlit] ──▶ [PDF Parsing + Chunking]
                                  │
                                  ▼
                          [Text Embedding using all-MiniLM-L6-v2]
                                  │
                                  ▼
                          [FAISS Vector DB (in memory)]
                                  │
                                  ▼
               [Query Input] ──▶ [Semantic Search (top-k chunks)]
                                  │
                                  ▼
                     [LLaMA3.2 generates Answer using Prompt]
                                  │
                                  ▼
                            [Streamlit Chat UI Output]

```

## Directory Structure:
```bash
├── data/
│   └── ondc_policy_docs/        # Place ONDC PDFs here
├── streamlit_app.py             # Streamlit frontend
├── QnA_setup_with_llama.py      # Backend: loading, embedding, answering
├── requirements.txt             # Python dependencies
└── README.md                    # You are here

```
## Features :
- Upload and process multiple PDFs
- Chunk documents with overlaps
- Retrieve semantically similar chunks using FAISS
- Answer questions using llama3.2 locally via Ollama
- Maintain Q&A chat history in the session
- Simple Streamlit interface

## Sample Input/Output:
Uploaded Files: upload all files 
```bash
        1) Definitions.pdf
        2) CHAPTER-[1-Onboarding-Complianc-Requirements-and-Certification-Requirements.pdf
        2) chapter_2__business_rules_v2.1.pdf
        3) CHAPTER-[3]-Commercial+Model.pdf
        4) CHAPTER-[4]-Code-of-Conduct-Ethics.pdf
        5) CHAPTER-[5]-Branding-Guidelines.pdf
        6) CHAPTER+[6]+Issue+and+Grievance+Management+Policy.pdf
```
**Sample Query:**
```bash
        1) Who bears the cost of a discount offered on an order?
        2) Can the Buyer App charge a fee to the Seller App?
```

**Output:**
```bash
        1) The cost of a discount offered on an order is typically borne by the Buyer App, as stated in section 3.2.4: "For clarity, the Network Participant offering a discount will bear the cost of such discount." This means that if the Buyer App offers a discount to the Seller, the Buyer App will absorb the associated costs, not the Seller.
        2) The Buyer App can charge a fee to the Seller App, which is referred to as the Buyer App Fee in Clause 3.2.3. This fee may be negotiated with the Seller App on a real-time basis before order completion and may be expressed as a percentage of the Total Order Value or a fixed amount per Successful Order from the Seller App.

```
## Quickstart:

1. Clone Repository
```bash
git clone git@github.com:nehastats/clickpost_qna_ondc_setup.git
cd ondc-rag-qa
```

3. Create Environment & Install Dependencies
```bash
conda create -n env_genai_qa python=3.10 -y
conda activate env_genai_qa
pip install -r requirements.txt
```

5. Download Models (Ollama)
```bash  
ollama serve  # Start Ollama server
ollama pull llama3.2
```

7. Add ONDC PDFs
Place all your ONDC policy PDFs inside:
```bash
data/ondc_policy_docs/
```
Or upload via Streamlit UI.

9. Launch the App
```bash   
streamlit run streamlit_app.py
```
Visit: http://localhost:8501

## How It Works:

- PDFs are chunked using RecursiveCharacterTextSplitter
- Chunks are embedded using all-MiniLM-L6-v2 via HuggingFaceEmbeddings
- Embeddings are stored in a FAISS vector store
- On a user query :
        - Top-k similar chunks are retrieved
        - A prompt is built using the context
        - Answer is generated using llama3.2

**Notes:**

- Ensure `llama3.2` is pulled via Ollama before use
- Models and processing run locally — no cloud calls
- Chat history is preserved across interactions during the same session

## Cost Estimation:
**Assuming:**
- 10,000 queries per day.
- 30 days per month.
Local setup using Ollama and HuggingFace (no cloud inference costs).

```bash 
Component	              Cost
Ollama (LLaMA3.2)	      Free (running locally)
HF Embeddings	              Free (CPU inference)
FAISS	                      Free (in-memory vector DB)
Streamlit UI	              Free (self-hosted)
Total Monthly	              ~$0 if self-hosted
```
 If using managed cloud inference (e.g., Replicate or OpenAI):

- Embedding: ~$0.001 per embed
- LLM: ~$0.002 per query
- Estimated Cloud Total: ~$600/month for 300,000 queries

## Production Guidelines:

- Vector DB Persistence: Use persistent FAISS storage (`save_local()/load_local()`) for faster startup.
- Batch Embedding: Embed and index offline once; only query at runtime.
- Streaming Answers: Use `LangChain` streaming output for LLMs if supported.
- Error Handling: Add exception handling for document parsing and model calls.
- Scalability: Consider migrating to `FastAPI + React` for scalable deployment.
- Security: Validate uploaded PDFs and limit file size.

## Files

- `streamlit_app.py` – Frontend and chat logic.
- `QnA_setup_with_llama.py` – Backend setup: loading PDFs, creating vector DB, embeddings, and query processing.
