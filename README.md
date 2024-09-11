# Chat with PDF

This is a Streamlit application that allows you to interact with PDF documents using Retrieval-Augmented Generation (RAG). The application processes PDF files, extracts their content, and enables you to ask questions about the content in the document.

## Features

- Upload PDF file for processing
- Ask questions related to the content of the uploaded PDF
- Get answers from a language model (llama 3 here), which now has the context of the uploaded document


## File Structure

- `app.py`: handles the Streamlit interface and user interactions.
- `rag.py`: Contains the `RAGSystem` class responsible for processing PDFs, creating embeddings and setting up the QA chain.

## Code Explanation

### `app.py`

1. **Imports**: Imports necessary libraries including `asyncio`, `os`, `streamlit`, and the `RAGSystem` class from `rag.py`.

2. **API keys**: Set up API keys as environment variables in streamlit secrets.

3. **RAG System Initialization**: Initializes the `RAGSystem` with the provided API keys and directories for PDFs and database storage.

4. **Streamlit Interface**:
   - Displays a title and a file uploader for PDF files.
   - Upon file upload, the PDF is saved to the `pdfs` directory.
   - The PDF is processed asynchronously, and a success message is displayed upon completion.
   - Sets up the QA chain after processing the PDF.
   - Provides a text input for users to ask questions about the PDF, and displays the generated answers.


### `rag.py`

1. **Imports**: Imports necessary libraries for handling PDFs, embeddings, and document processing.

2. **RAGSystem Class**:
   - **Initialization**: Sets up API keys, directories, and initializes variables for Qdrant and QA systems.
   - **PDF Processing**: 
     - Uses `LlamaParse` to parse the PDF and save it as a markdown file.
     - Detects file encoding and reads the content.
     - Splits the content into manageable chunks using `RecursiveCharacterTextSplitter`.
     - Creates embeddings and stores them in Qdrant.
   - **QA Chain Setup**: Configures the QA chain using a language model (llama 3) and a retriever with compression.
   - **Question Handling**: Provides a method to ask questions and retrieve answers.

## Running the Application

## clone the repo

```bash
git clone https://github.com/adc77/pdf_chat.git
```
open folder in VScode/cursor or any other IDE

## create venv

```bash
python -m venv venv
```
## activate venv

```bash
venv/scripts/activate
```
## Install requirements

```bash
pip install -r requirements.txt
```
## Set up Environment Variables

Set the following environment variables in your Streamlit secrets:

- `GROQ_API_KEY`: Your groq API key from https://console.groq.com/login.
- `LLAMA_PARSE_API_KEY`: Your LlamaParse API key from https://cloud.llamaindex.ai/login.

To run the application:

```bash
streamlit run app.py
```

Open your browser and navigate to `http://localhost:8501` to interact with the application.

