# Chat with PDF

This project is a Streamlit application that allows users to interact with PDF documents using Retrieval-Augmented Generation (RAG) technology. The application processes PDF files, extracts their content, and enables users to ask questions about the content.

## Features

- Upload PDF files for processing.
- Ask questions related to the content of the uploaded PDF.
- Retrieve answers using a language model.

## Requirements

Ensure you have the following Python packages installed:
plaintext
streamlit==1.24.0
langchain==0.1.17
langchain-groq==0.1.3
llama-parse==0.1.3
qdrant-client==1.9.1
unstructured[md]==0.13.6
fastembed==0.2.7
flashrank==0.2.4
chardet==5.1.0
langchain_core


You can install the requirements using:
bash
pip install -r requirements.txt


## Environment Variables

Set the following environment variables in your Streamlit secrets:

- `GROQ_API_KEY`: API key for the GROQ service.
- `LLAMA_PARSE_API_KEY`: API key for the LlamaParse service.

## File Structure

- `app.py`: Main application file that handles the Streamlit interface and user interactions.
- `rag.py`: Contains the `RAGSystem` class responsible for processing PDFs and setting up the QA chain.

## Code Explanation

### `app.py`

1. **Imports**: Imports necessary libraries including `asyncio`, `os`, `streamlit`, and the `RAGSystem` class from `rag.py`.

2. **Environment Setup**: Sets up environment variables using Streamlit secrets for API keys.

3. **RAG System Initialization**: Initializes the `RAGSystem` with the provided API keys and directories for PDFs and database storage.

4. **Streamlit Interface**:
   - Displays a title and a file uploader for PDF files.
   - Upon file upload, the PDF is saved to the `pdfs` directory.
   - The PDF is processed asynchronously, and a success message is displayed upon completion.
   - Sets up the QA chain after processing the PDF.
   - Provides a text input for users to ask questions about the PDF, and displays the generated answers.

5. **Sidebar**: Contains information about the application.

### `rag.py`

1. **Imports**: Imports necessary libraries for handling PDFs, embeddings, and document processing.

2. **RAGSystem Class**:
   - **Initialization**: Sets up API keys, directories, and initializes variables for Qdrant and QA systems.
   - **PDF Processing**: 
     - Uses `LlamaParse` to parse the PDF and save it as a markdown file.
     - Detects file encoding and reads the content.
     - Splits the content into manageable chunks using `RecursiveCharacterTextSplitter`.
     - Creates embeddings and stores them in Qdrant.
   - **QA Chain Setup**: Configures the QA chain using a language model and a retriever with compression.
   - **Question Handling**: Provides a method to ask questions and retrieve answers.

## Running the Application

To run the application, execute the following command in your terminal:
bash
streamlit run app.py


Open your browser and navigate to `http://localhost:8501` to interact with the application.

## License

This project is licensed under the MIT License.