import asyncio
import os
import streamlit as st
from rag import RAGSystem
#from updated_rag import RAGSystem

# Set up environment variables
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
os.environ["LLAMA_PARSE_API_KEY"] = st.secrets["LLAMA_PARSE_API_KEY"]

# Initialize RAG system
rag_system = RAGSystem(
    groq_api_key=os.environ["GROQ_API_KEY"],
    llama_parse_api_key=os.environ["LLAMA_PARSE_API_KEY"],
    pdf_dir="pdfs",
    db_dir="db"
)

st.title("Chat with PDF")

# File uploader
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # Save the uploaded file
    with open(os.path.join("pdfs", uploaded_file.name), "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Process the PDF
    with st.spinner("Processing PDF..."):
        asyncio.run(rag_system.process_pdf(os.path.join("pdfs", uploaded_file.name)))
    
    st.success("PDF processed successfully!")
    
    # Set up QA chain
    rag_system.setup_qa_chain()
    
    # Chat interface
    st.subheader("Ask questions about the PDF")
    user_question = st.text_input("Enter your question:")
    
    if user_question:
        with st.spinner("Generating answer..."):
            response = rag_system.ask_question(user_question)
        
        st.write("Answer:")
        st.write(response["result"])

st.sidebar.header("About")
st.sidebar.info("This app allows you to chat with a PDF document using RAG (Retrieval-Augmented Generation)")