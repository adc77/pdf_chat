import asyncio
import os
from keyword_extractor import KeywordExtractor

async def main():
    groq_api_key = os.environ.get("GROQ_API_KEY")
    llama_parse_api_key = os.environ.get("LLAMA_PARSE_API_KEY")
    
    if not groq_api_key or not llama_parse_api_key:
        print("Please set GROQ_API_KEY and LLAMA_PARSE_API_KEY environment variables.")
        return

    pdf_dir = "pdf_files"
    db_dir = "vector_db"

    extractor = KeywordExtractor(groq_api_key, llama_parse_api_key, pdf_dir, db_dir)

    # Hardcoded PDF file path
    pdf_path = "pdfs\sample.pdf"  # Replace this with the actual path to your PDF file
    
    print(f"Processing PDF: {pdf_path}")
    await extractor.process_pdf(pdf_path)

    k = int(input("Enter the number of keywords to extract: "))
    keywords = extractor.extract_keywords(k)

    print(f"\nTop {k} keywords extracted from the document:")
    for i, keyword in enumerate(keywords, 1):
        print(f"{i}. {keyword}")

if __name__ == "__main__":
    asyncio.run(main())