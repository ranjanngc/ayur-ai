# pdf_to_dataset.py - Convert PDF to RAG dataset (JSONL format)

import json
import argparse
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract raw text from PDF."""
    print(f"Extracting text from {pdf_path}...")
    reader = PdfReader(pdf_path)
    text = ""
    for page_num, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n\n"
        print(f"Processed page {page_num}/{len(reader.pages)}")
    print(f"Extracted {len(text)} characters.")
    return text

def clean_text(text: str) -> str:
    """Basic cleaning: remove excessive whitespace, headers/footers if needed."""
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    # Remove very short lines (likely page numbers, headers)
    lines = [line for line in lines if len(line) > 30]
    return "\n\n".join(lines)

def split_into_passages(text: str, chunk_size=800, chunk_overlap=100):
    """Split text into chunks suitable for RAG."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_text(text)
    print(f"Created {len(chunks)} passages.")
    return chunks

def save_as_jsonl(passages, output_path: str):
    """Save as JSONL with the format you used: assistant messages."""
    print(f"Saving to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        for passage in passages:
            record = {
                "messages": [
                    {"role": "assistant", "content": passage.strip()}
                ]
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Dataset saved: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Convert PDF to RAG dataset (JSONL)")
    parser.add_argument("pdf_path", type=str, help="Path to input PDF file")
    parser.add_argument("-o", "--output", type=str, default="charaka_dataset.jsonl",
                        help="Output JSONL file (default: charaka_dataset.jsonl)")
    parser.add_argument("--chunk_size", type=int, default=800, help="Chunk size (default: 800)")
    parser.add_argument("--chunk_overlap", type=int, default=100, help="Chunk overlap (default: 100)")

    args = parser.parse_args()

    # Validate PDF exists
    pdf_path = Path(args.pdf_path)
    if not pdf_path.exists():
        print(f"Error: PDF file not found: {pdf_path}")
        return

    # Process
    raw_text = extract_text_from_pdf(str(pdf_path))
    cleaned_text = clean_text(raw_text)
    passages = split_into_passages(cleaned_text, args.chunk_size, args.chunk_overlap)
    save_as_jsonl(passages, args.output)

    print("\nConversion complete! 🌿")
    print(f"Use {args.output} as your dataset for the RAG system.")

if __name__ == "__main__":
    main()


# python PDFtoJSON.py "/home/ranjan/work/ranjan/ayur/resources/CarakaSamhita_priyadaranjanRay.pdf" -o charaka_clean-v1.jsonl