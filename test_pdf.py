from pypdf import PdfReader

pdf_path = "C:\College RAG Admission Process\data\Indala_College_of_Engineering_Information_2026-27.pdf"

reader = PdfReader(pdf_path)

for page in reader.pages:
    print(page.extract_text())