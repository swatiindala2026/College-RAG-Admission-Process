from pypdf import PdfReader
from pathlib import Path
import json

pdf_path = Path("C:\College RAG Admission Process\data\Indala_College_of_Engineering_Information_2026-27.pdf")

reader = PdfReader(str(pdf_path))

chunks = []

chunk_size = 1000
overlap = 150

for page_number, page in enumerate(reader.pages, start=1):

    text = page.extract_text() or ""
    text = text.strip()

    start = 0

    while start < len(text):
        end = start + chunk_size

        chunk_text = text[start:end]

        if chunk_text.strip():
            chunks.append({
                "text": chunk_text,
                "source": pdf_path.name,
                "page": page_number
            })

        start += chunk_size - overlap

output_path = Path("outputs/chunks.json")
output_path.parent.mkdir(exist_ok=True)

output_path.write_text(
    json.dumps(chunks, indent=4, ensure_ascii=False),
    encoding="utf-8"
)

print("Total Chunks:", len(chunks))
print("Chunks saved successfully!")