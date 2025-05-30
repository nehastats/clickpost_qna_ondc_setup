import streamlit as st
import QnA_setup_with_llama as main
import os

st.title("Chat with PDFs using llama-3.2")

uploaded_files = st.file_uploader(
    "Upload one or more PDFs",
    type="pdf",
    accept_multiple_files=True
)

# Initialize session state to hold chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if uploaded_files:
    db_list = []

    for uploaded_file in uploaded_files:
        file_path = os.path.join(main.pdfs_directory, uploaded_file.name)
        main.upload_pdf(uploaded_file)  # save each file
        db = main.create_vector_store(file_path)
        db_list.append(db)

    # Merge all individual FAISS stores into one
    combined_db = db_list[0]
    for other_db in db_list[1:]:
        combined_db.merge_from(other_db)

    # Display past chat history
    for q, a in st.session_state.chat_history:
        st.chat_message("user").write(q)
        st.chat_message("assistant").write(a)

    question = st.chat_input("Ask a question based on the uploaded PDFs")

    if question:
        st.chat_message("user").write(question)
        related_documents = main.retrieve_docs(combined_db, question)
        answer = main.question_pdf(question, related_documents)
        st.chat_message("assistant").write(answer)

        # Append current interaction to history
        st.session_state.chat_history.append((question, answer))