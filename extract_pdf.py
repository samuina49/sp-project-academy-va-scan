import fitz
import os

pdf_path = "2024 FaceSocial #Image.pdf"
output_dir = "test_samples" 
try:
    doc = fitz.open(pdf_path)
    # Save pages 5 to 13 (TOC usually here)
    for page_num in range(5, 14):
        page = doc.load_page(page_num)
        pix = page.get_pixmap(dpi=150)
        pix.save(os.path.join(output_dir, f"toc_{page_num}.png"))
        print(f"Saved TOC page {page_num}")
except Exception as e:
    print(f"Error: {e}")
