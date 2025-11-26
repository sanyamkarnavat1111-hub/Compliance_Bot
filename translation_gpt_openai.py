import base64
import fitz  # PyMuPDF
from tqdm import tqdm
import re
from tabulate import tabulate
from PIL import Image
import os
from dotenv import load_dotenv
from functools import lru_cache
from openai import OpenAI
from logger import custom_logger as llog
from config import arabic_threshold, other_threshold
import string

class PDFProcessor:
    def __init__(self, log_save_file_name):
        load_dotenv()
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.model_name = "gpt-4o"
        self.log_save_file_name = log_save_file_name

    @lru_cache(maxsize=500)
    def translate_text(self, text, source_lang="Other", target_lang="English"):
        # Simply return the original text without translation
        return text.strip() if text else text

    def is_table_row(self, text):
        return bool(re.match(r'^\s*[\|\+].*[\|\+]\s*$', text) and text.count('|') > 1)

    def extract_table(self, lines, start_index):
        table_lines = []
        i = start_index
        while i < len(lines) and self.is_table_row(lines[i]):
            table_lines.append(lines[i])
            i += 1
        return table_lines, i

    def process_table(self, table_text):
        llog("PDFProcessor", "Processing table", self.log_save_file_name)
        lines = table_text.split('\n')
        # Just split the lines and preserve original content
        table_data = [re.split(r'\s*\|\s*', line.strip('|')) for line in lines if line.strip()]
        return tabulate(table_data, headers="firstrow", tablefmt="grid")

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def generate_image_caption(self, image_path):
        llog("PDFProcessor", f"Generating caption for image: {image_path}", self.log_save_file_name)
        base64_image = self.encode_image(image_path)

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """You are a precise document transcriber. Your task is to convert this image into text exactly as it appears, maintaining all formatting and content. Please:

                            1. Transcribe ALL visible text exactly as written, including:
                               - Headers and footers
                               - Main body text
                               - Table contents
                               - Bullet points and numbered lists
                               - Any visible numbers, references, or citations
                            
                            2. Preserve the formatting using markdown:
                               - Use tables for tabular data
                               - Maintain bullet points and numbering
                               - Preserve paragraph breaks
                               - Keep text alignment (left/center/right)
                            
                            3. For any graphics or images:
                               - Describe their location and content
                               - Include any text or labels within them
                            
                            4. If any part is unclear or partially visible:
                               - Transcribe what you can see
                               - Note any uncertainties with [unclear: description]
                            
                            Output the content exactly as it appears, maintaining the original structure and format."""
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=4000
        )
        llog("PDFProcessor", "Image caption generated successfully", self.log_save_file_name)
        return response.choices[0].message.content

    def process_pdf(self, input_pdf, output_txt):
        llog("PDFProcessor", f"Starting to process PDF", self.log_save_file_name)
        doc = input_pdf
        output = []

        try:
            for page_num in tqdm(range(len(doc)), desc="Processing pages"):
                page = doc[page_num]
                text = page.get_text()
                
                # Add page marker at the start of each page
                output.append(f"\n\n\n\n\n--- Page {page_num + 1} ---\n\n")
                
                # Process text content
                if text.strip():
                    lines = text.split('\n')
                    i = 0
                    while i < len(lines):
                        if self.is_table_row(lines[i]):
                            table_lines, i = self.extract_table(lines, i)
                            table_text = '\n'.join(table_lines)
                            output.append(table_text)
                        else:
                            # Preserve original text without translation
                            line = lines[i].strip()
                            if line:
                                output.append(line)
                            i += 1

                # Process images
                for img in page.get_images(full=True):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_rect = page.get_image_bbox(img)
                    
                    # Calculate image dimensions relative to page
                    image_height = image_rect.y1 - image_rect.y0
                    page_height = page.rect.height
                    image_height_percentage = (image_height / page_height) * 100
                    
                    # Skip if it's a small header/footer image (less than 10% of page height)
                    header_region = page.rect.height * 0.1
                    footer_region = page.rect.height * 0.9
                    
                    is_small_header = image_rect.y0 < header_region and image_height_percentage < 10
                    is_small_footer = image_rect.y1 > footer_region and image_height_percentage < 10
                    
                    if not (is_small_header or is_small_footer):
                        temp_image_path = f"temp_image_{page_num}_{xref}.jpg"
                        
                        try:
                            with open(temp_image_path, "wb") as image_file:
                                image_file.write(base_image["image"])
                            
                            image_caption = self.generate_image_caption(temp_image_path)
                            if image_caption:
                                output.append(f"\n[Image Description: {image_caption}]\n")
                        finally:
                            if os.path.exists(temp_image_path):
                                os.remove(temp_image_path)

            # Write output maintaining original format
            with open(output_txt, "w", encoding="utf-8") as f:
                for item in output:
                    f.write(f"{item}\n")

        except Exception as e:
            llog("translation_gpt", f"Critical error in PDF processing: {str(e)}", self.log_save_file_name)
            return f"Critical error in PDF processing: {str(e)}", "Failed to process the PDF document"

        return None, None

# Usage
if __name__ == "__main__":
    processor = PDFProcessor(log_save_file_name="log_save_file_name")
    input_pdf = "/home/cb_manager/tendor_poc/image_only_rfp.pdf"
    output_txt = "translated_1.txt"
    pdf_document = fitz.open(input_pdf) # Load binary data into PDF using Fitz
    llog("PDFProcessor", "Starting PDF processing", "main_process")
    processor.process_pdf(pdf_document, output_txt)
    llog("PDFProcessor", "PDF processing completed", "main_process")