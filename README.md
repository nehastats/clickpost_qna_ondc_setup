🧠 ONDC QA System using RAG (LLM + FAISS + Embeddings)


This repository implements a Retrieval-Augmented Generation (RAG) pipeline to perform Question Answering (QA) over ONDC policy documents using:

    -SentenceTransformer (all-MiniLM-L6-v2) for embedding

    -FAISS for vector search

    -Ollama's llama3.2 for LLM-based answer generation

    -Streamlit for an interactive chat interface

Design Architecture:

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


📁 Directory Structure

├── data/
│   └── ondc_policy_docs/        # Place ONDC PDFs here
├── streamlit_app.py             # Streamlit frontend
├── QnA_setup_with_llama.py      # Backend: loading, embedding, answering
├── requirements.txt             # Python dependencies
└── README.md                    # You are here

Features :
1) Upload and process multiple PDFs
2) Chunk documents with overlaps
3) Retrieve semantically similar chunks using FAISS
4) Answer questions using llama3.2 locally via Ollama
5) Maintain Q&A chat history in the session
6) Simple Streamlit interface

Sample Input/Output:
Uploaded Files: upload all files 
        1) Definitions.pdf
        2) CHAPTER-[1-Onboarding-Complianc-Requirements-and-Certification-Requirements.pdf
        2) chapter_2__business_rules_v2.1.pdf
        3) CHAPTER-[3]-Commercial+Model.pdf
        4) CHAPTER-[4]-Code-of-Conduct-Ethics.pdf
        5) CHAPTER-[5]-Branding-Guidelines.pdf
        6) CHAPTER+[6]+Issue+and+Grievance+Management+Policy.pdf
Sample Query:
        1) Who bears the cost of a discount offered on an order?
        2) Can the Buyer App charge a fee to the Seller App?
        3) Is it permissible for me to register under a brand name that I am not directly
associated with?
        4) Can Network Participants say “Partnered with ONDC” in their promotions?
        5) Who handles a grievance raised by a Buyer?
Output:
        1) The cost of a discount offered on an order is typically borne by the Buyer App, as stated in section 3.2.4: "For clarity, the Network Participant offering a discount will bear the cost of such discount." This means that if the Buyer App offers a discount to the Seller, the Buyer App will absorb the associated costs, not the Seller.
        2) The Buyer App can charge a fee to the Seller App, which is referred to as the Buyer App Fee in Clause 3.2.3. This fee may be negotiated with the Seller App on a real-time basis before order completion and may be expressed as a percentage of the Total Order Value or a fixed amount per Successful Order from the Seller App.


Quickstart:

1. Clone Repository
git clone https://github.com/yourname/ondc-rag-qa.git
cd ondc-rag-qa

2. Create Environment & Install Dependencies
conda create -n env_genai_qa python=3.10 -y
conda activate env_genai_qa
pip install -r requirements.txt

3. Download Models (Ollama)
ollama serve  # Start Ollama server
ollama pull llama3.2

4. Add ONDC PDFs
Place all your ONDC policy PDFs inside:
data/ondc_policy_docs/
Or upload via Streamlit UI.

5. Launch the App
streamlit run streamlit_app.py
Visit: http://localhost:8501

How It Works:
-PDFs are chunked using RecursiveCharacterTextSplitter
-Chunks are embedded using all-MiniLM-L6-v2 via HuggingFaceEmbeddings
-Embeddings are stored in a FAISS vector store
-On a user query:
        -Top-k similar chunks are retrieved
        -A prompt is built using the context
        -Answer is generated using llama3.2

Example Questions:
-- “What is the grievance redressal process in ONDC?”
-- “How are network participants onboarded?”
-- “Which policies cover logistics service providers?”

Notes:
--Ensure llama3.2 is pulled via Ollama before use
--Models and processing run locally — no cloud calls
--Chat history is preserved across interactions during the same session

Cost Estimation:
Assuming:
--10,000 queries per day.
--30 days per month.
Local setup using Ollama and HuggingFace (no cloud inference costs).

Component	              Cost
Ollama (LLaMA3.2)	      Free (running locally)
HF Embeddings	          Free (CPU inference)
FAISS	                  Free (in-memory vector DB)
Streamlit UI	          Free (self-hosted)
Total Monthly	          ~$0 if self-hosted

If using managed cloud inference (e.g., Replicate or OpenAI):
Embedding: ~$0.001 per embed
LLM: ~$0.002 per query
Estimated Cloud Total: ~$600/month for 300,000 queries

Production Guidelines:
1)Vector DB Persistence: Use persistent FAISS storage (save_local()/load_local()) for faster startup.
2)Batch Embedding: Embed and index offline once; only query at runtime.
3)Streaming Answers: Use LangChain streaming output for LLMs if supported.
4)Error Handling: Add exception handling for document parsing and model calls.
5)Scalability: Consider migrating to FastAPI + React for scalable deployment.
6)Security: Validate uploaded PDFs and limit file size.

📁 Files
1) streamlit_app.py – Frontend and chat logic.

2) QnA_setup_with_llama.py – Backend setup: loading PDFs, creating vector DB, embeddings, and query processing.
