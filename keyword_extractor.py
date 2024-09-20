import os
from pathlib import Path
import chardet
from typing import List

from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Qdrant
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_groq import ChatGroq
from llama_parse import LlamaParse
from langchain_core.documents import Document

class KeywordExtractor:
    def __init__(self, groq_api_key: str, llama_parse_api_key: str, pdf_dir: str, db_dir: str):
        self.groq_api_key = groq_api_key
        self.llama_parse_api_key = llama_parse_api_key
        self.pdf_dir = Path(pdf_dir)
        self.db_dir = Path(db_dir)
        self.qdrant = None

        os.environ["GROQ_API_KEY"] = self.groq_api_key

    async def process_pdf(self, pdf_path: str) -> None:
        instruction = """The provided document is a PDF file.
        Extract the main keywords and concepts from this document."""

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

        with open(document_path, 'rb') as file:
            raw_data = file.read()
            result = chardet.detect(raw_data)
            file_encoding = result['encoding']

        content = raw_data.decode(file_encoding)

        doc = Document(page_content=content, metadata={"source": str(document_path)})

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2048,
            chunk_overlap=128,
        )

        docs = text_splitter.split_documents([doc])

        embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")

        self.qdrant = Qdrant.from_documents(
            docs,
            embeddings,
            path=str(self.db_dir),
            collection_name="document_embeddings",
        )

    def extract_keywords(self, k: int) -> List[str]:
        if not self.qdrant:
            raise ValueError("Document not processed. Call process_pdf() first.")

        llm = ChatGroq(temperature=0, model_name="llama-3.1-8b-instant")

        prompt_template = f"""
        Extract the top {k} keywords (only keywords) from the following context.
        Provide only the list of keywords, without any additional explanation.

        Context: {{context}}

        Top {k} Keywords:
        """
        prompt = PromptTemplate(template=prompt_template, input_variables=["context"])

        retriever = self.qdrant.as_retriever(search_kwargs={"k": 5})
        contexts = retriever.get_relevant_documents("")

        combined_context = "\n".join([doc.page_content for doc in contexts])
        
        response = llm.predict(prompt.format(context=combined_context))
        
        keywords = [keyword.strip() for keyword in response.split("\n") if keyword.strip()]
        return keywords[:k]