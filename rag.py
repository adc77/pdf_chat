import os
from pathlib import Path
from typing import List
import chardet

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
from langchain_core.documents import Document

class RAGSystem:
    def __init__(self, groq_api_key: str, llama_parse_api_key: str, pdf_dir: str, db_dir: str):
        self.groq_api_key = groq_api_key
        self.llama_parse_api_key = llama_parse_api_key
        self.pdf_dir = Path(pdf_dir)
        self.db_dir = Path(db_dir)
        self.qdrant = None
        self.qa = None

        os.environ["GROQ_API_KEY"] = self.groq_api_key

    async def process_pdf(self, pdf_path: str) -> None:
        instruction = """The provided document is a PDF file.
        Try to be precise while answering the questions based on this document."""

        parser = LlamaParse(
            api_key=self.llama_parse_api_key,
            result_type="markdown",
            parsing_instruction=instruction,
            max_timeout=5000,
        )

        llama_parse_documents = await parser.aload_data(pdf_path)
        parsed_doc = llama_parse_documents[0]
        document_path = self.pdf_dir / f"{Path(pdf_path).stem}_parsed.md"
        with document_path.open("w") as f:
            f.write(parsed_doc.text)

         # Detect the file encoding
        with open(document_path, 'rb') as file:
            raw_data = file.read()
            result = chardet.detect(raw_data)
            file_encoding = result['encoding']

        # without using UnstructuredMarkdownLoader
        # Decode the content using the detected encoding
        content = raw_data.decode(file_encoding)

        # Create a Document object
        doc = Document(page_content=content, metadata={"source": str(document_path)})

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2048,
            chunk_overlap=128,
        )

        # docs = text_splitter.split_documents(loaded_documents)
        # or
        docs = text_splitter.split_documents([doc])

        embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")

        self.qdrant = Qdrant.from_documents(
            docs,
            embeddings,
            path=str(self.db_dir),
            collection_name="document_embeddings",
        )

    def setup_qa_chain(self) -> None:
        retriever = self.qdrant.as_retriever(search_kwargs={"k": 5})
        compressor = FlashrankRerank(model="ms-marco-MiniLM-L-12-v2")
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=retriever
        )

        llm = ChatGroq(temperature=0, model_name="llama-3.1-8b-instant")

        prompt_template = """
        Use the following pieces of information to answer the user's question.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        Context: {context}
        Question: {question}

        Answer the question and provide additional helpful information,
        based on the pieces of information, if applicable. Be succinct.

        Responses should be properly formatted to be easily read.
        """
        prompt = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        self.qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=compression_retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt, "verbose": False},
        )

    def ask_question(self, question: str) -> dict:
        if not self.qa:
            raise ValueError("QA chain not set up. Call setup_qa_chain() first.")
        return self.qa.invoke(question)

def print_response(response: dict) -> None:
    response_txt = response["result"]
    for chunk in response_txt.split("\n"):
        if not chunk:
            print()
            continue
        print("\n".join(textwrap.wrap(chunk, 100, break_long_words=False)))