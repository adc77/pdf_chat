import os
import textwrap
from pathlib import Path

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Qdrant
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_groq import ChatGroq
from llama_parse import LlamaParse

# Initialize paths
PDF_STORAGE = "./pdf_storage/"
EMBEDDINGS_PATH = PDF_STORAGE + "embeddings/"

# Load environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLAMA_PARSE_API_KEY = os.getenv("LLAMA_PARSE")

# Helper function to process the uploaded PDF
def process_pdf(pdf_path: str):
    instruction = """The provided document is Meta First Quarter 2024 Results.
    This form provides detailed financial information about the company's performance for a specific quarter.
    It includes unaudited financial statements, management discussion and analysis, and other relevant disclosures.
    """

    # Initialize the LlamaParse API to parse the document
    parser = LlamaParse(api_key=LLAMA_PARSE_API_KEY, result_type="markdown", parsing_instruction=instruction, max_timeout=5000)

    # Parse the PDF document
    parsed_docs = parser.load_data(pdf_path)
    parsed_doc = parsed_docs[0]

    # Save the parsed document
    document_path = Path(f"{PDF_STORAGE}/parsed_docs/{Path(pdf_path).stem}.md")
    with document_path.open("a") as f:
        f.write(parsed_doc.text)

    # Load the parsed document
    loader = UnstructuredMarkdownLoader(document_path)
    loaded_documents = loader.load()

    return loaded_documents

# Helper function to split document into smaller chunks
def split_documents(loaded_documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2048, chunk_overlap=128)
    docs = text_splitter.split_documents(loaded_documents)
    return docs

# Function to embed and store documents in Qdrant
def embed_and_store(docs):
    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    qdrant = Qdrant.from_documents(docs, embeddings, path=EMBEDDINGS_PATH, collection_name="document_embeddings")
    return qdrant

# Function to retrieve relevant documents based on a query
def retrieve_documents(query, qdrant):
    retriever = qdrant.as_retriever(search_kwargs={"k": 5})
    compressor = FlashrankRerank(model="ms-marco-MiniLM-L-12-v2")
    compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)
    return compression_retriever.invoke(query)

# Helper function to run the retrieval-augmented QA
def run_qa(query, compression_retriever):
    llm = ChatGroq(temperature=0, model_name="llama3-70b-8192")
    prompt_template = """
    Use the following pieces of information to answer the user's question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    
    Context: {context}
    Question: {question}
    
    Answer the question and provide additional helpful information, if applicable. Be succinct.
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=compression_retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt, "verbose": True},
    )
    
    response = qa.invoke(query)
    return response

# Main function to run the whole pipeline
def chat_with_pdf(pdf_path, query):
    # Process PDF
    loaded_docs = process_pdf(pdf_path)
    
    # Split docs into chunks
    docs = split_documents(loaded_docs)
    
    # Embed and store
    qdrant = embed_and_store(docs)
    
    # Retrieve relevant docs
    compression_retriever = retrieve_documents(query, qdrant)
    
    # Run the QA
    response = run_qa(query, compression_retriever)
    
    return response
