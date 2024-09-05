import streamlit as st
from rag import chat_with_pdf

st.title("Chat with your PDF")

# Upload PDF
uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

# Query input
query = st.text_input("Ask a question about the document")

# Perform chat with PDF when both file and query are provided
if uploaded_file and query:
    pdf_path = f"./pdf_storage/{uploaded_file.name}"
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Run the chat with PDF
    response = chat_with_pdf(pdf_path, query)
    
    # Display the response
    st.write(response)
