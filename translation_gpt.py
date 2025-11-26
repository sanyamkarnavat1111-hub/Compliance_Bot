import base64
import requests
import torch
import fitz  # PyMuPDF
from tqdm import tqdm
import re
from tabulate import tabulate
import io
from PIL import Image
import os
from dotenv import load_dotenv
from functools import lru_cache
from openai import AzureOpenAI
from logger import custom_logger as llog

class PDFProcessor:
    def __init__(self, log_save_file_name):
        load_dotenv()
        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        self.deployment_name = os.getenv("AZURE_DEPLOYMENT_NAME")
        self.log_save_file_name = log_save_file_name

    @lru_cache(maxsize=500)
    def translate_text(self, text, source_lang="Arabic", target_lang="English"):
        if not re.search(r'[\u0600-\u06FF]', text):
            return text

        prompt = f"Translate the following {source_lang} text to {target_lang}:\n\n{text}\n\nTranslation:"
        
        # try:
        llog("PDFProcessor", f"Attempting to translate text", self.log_save_file_name)
        response = self.client.chat.completions.create(
            model=self.deployment_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that translates text accurately."},
                {"role": "user", "content": prompt}
            ]
        )
            
        translation = response.choices[0].message.content.strip()
        llog("PDFProcessor", f"Text translated successfully", self.log_save_file_name)
        return translation.split("Translation:")[-1].strip()
        # except Exception as e:
        #     llog("PDFProcessor", f"Translation error: {e}", self.log_save_file_name)
        #     return "[Translation error]"

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
        table_data = [re.split(r'\s*\|\s*', line.strip('|')) for line in lines if line.strip()]
        
        processed_data = []
        for row in table_data:
            processed_row = [self.translate_text(cell) for cell in row if cell]
            processed_data.append(processed_row)
        
        llog("PDFProcessor", "Table processed successfully", self.log_save_file_name)
        return tabulate(processed_data, headers="firstrow", tablefmt="grid")

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def generate_image_caption(self, image_path):
        llog("PDFProcessor", f"Generating caption for image: {image_path}", self.log_save_file_name)
        base64_image = self.encode_image(image_path)

        response = self.client.chat.completions.create(
            model=self.deployment_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Please provide an accurate caption for the text in this image in English."
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
            max_tokens=3000
        )
        llog("PDFProcessor", "Image caption generated successfully", self.log_save_file_name)
        return response.choices[0].message.content

    def process_pdf(self, input_pdf, output_txt):
        doc = input_pdf
        error_msg_for_developer = None
        error_msg_for_user = None
        output = []
        chance_to_try_procss = 0 
        for page_num in tqdm(range(len(doc)), desc="Processing pages"):
            # try:
            llog("PDFProcessor", f"Processing page {page_num + 1}", self.log_save_file_name)
            page = doc[page_num]
            text = page.get_text()
            lines = text.split('\n')
            if text.strip():
                i = 0
                while i < len(lines):
                    if self.is_table_row(lines[i]):
                        llog("PDFProcessor", f"Table detected on page {page_num + 1}", self.log_save_file_name)
                        table_lines, i = self.extract_table(lines, i)
                        table_text = '\n'.join(table_lines)
                        try:
                            processed_table = self.process_table(table_text)
                            output.append({"type": "table", "content": processed_table, "page": page_num + 1})
                        except Exception as e:
                            if "ResponsibleAIPolicyViolation" in str(e):
                                llog("translation_gpt", f"Content filtering error occurred so ignore the error and continue the process at page : {page_num + 1}", self.log_save_file_name)
                                output.append({"type": "table", "content": "Content filtering error occurred so ignore the error and continue the process", "page": page_num + 1})
                            else:
                                llog("translation_gpt", f"Error occurred while processing the pdf in self.process_table part: {str(e)}", self.log_save_file_name)
                                chance_to_try_procss += 1
                                if chance_to_try_procss >= 2:
                                    error_msg_for_user = "We encountered an issue while processing a table in the PDF. The content may be unsupported or corrupted."
                                    error_msg_for_developer = f"Error occurred while processing the pdf: {str(e)}"
                                    return error_msg_for_developer, error_msg_for_user
                                else:
                                    pass
                    else:
                        paragraph = []
                        while i < len(lines) and not self.is_table_row(lines[i]):
                            paragraph.append(lines[i])
                            i += 1
                        if paragraph:
                            para_text = ' '.join(paragraph)
                            try:
                                llog("translation_gpt", f"going to translate text of page : {page_num + 1}", self.log_save_file_name)

                                translated_para = self.translate_text(para_text)

                                output.append({"type": "text", "content": translated_para, "page": page_num + 1})
                            except Exception as e:
                                if "ResponsibleAIPolicyViolation" in str(e):
                                    llog("translation_gpt", f"Content filtering error occurred so ignore the error and continue the process at page : {page_num + 1}", self.log_save_file_name)
                                    output.append({"type": "table", "content": "Content filtering error occurred so ignore the error and continue the process", "page": page_num + 1})
                                else:
                                    llog("translation_gpt", f"Error occurred while processing the pdf at self.translate_text part: {str(e)}", self.log_save_file_name)

                                    chance_to_try_procss += 1
                                    if chance_to_try_procss >= 2:
                                        error_msg_for_user = "We encountered an issue while translating text in the PDF. The content may be unsupported or corrupted."
                                        error_msg_for_developer = f"Error occurred while processing the pdf: {str(e)}"
                                        return error_msg_for_developer, error_msg_for_user
                                    else:
                                        pass
                
        
            for img in page.get_images(full=True):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_rect = page.get_image_bbox(img)
                                
                # Define header and footer regions
                header_threshold = page.rect.height * 0.1
                footer_threshold = page.rect.height * 0.9
                
                # Exclude images in header and footer regions, but include full-page images
                if (image_rect.y0 > header_threshold and image_rect.y1 < footer_threshold) or (image_rect.y0 == 0 and image_rect.y1 == page.rect.height):
                    temp_image_path = f"temp_image_{page_num}_{xref}.jpg"
                    with open(temp_image_path, "wb") as image_file:
                        image_file.write(base_image["image"])
                    
                    try:
                        image_caption = self.generate_image_caption(temp_image_path)
                        os.remove(temp_image_path)

                        output.append({
                            "type": "image_caption",
                            "content": image_caption,
                            "page": page_num + 1
                        })
                    except Exception as e:
                            if "ResponsibleAIPolicyViolation" in str(e):
                                llog("translation_gpt", f"Content filtering error occurred so ignore the error and continue the process at page : {page_num + 1}", self.log_save_file_name)

                                output.append({"type": "table", "content": "Content filtering error occurred so ignore the error and continue the process", "page": page_num + 1})
                            else:
                                llog("translation_gpt", f"Error occurred while processing the pdf at self.generate_image_caption part: {str(e)}", self.log_save_file_name)

                                chance_to_try_procss += 1
                                if chance_to_try_procss >= 2:
                                    error_msg_for_user = "We encountered an issue while processing an image in the PDF. The image may be unsupported or corrupted."
                                    error_msg_for_developer = f"Error occurred while processing the pdf: {str(e)}"
                                    if os.path.exists(temp_image_path):
                                        os.remove(temp_image_path)
                                    return error_msg_for_developer, error_msg_for_user
                                else:
                                    pass
                    if os.path.exists(temp_image_path):
                        os.remove(temp_image_path)
                    
        try:
            with open(output_txt, "w", encoding="utf-8") as f:
                for item in output:
                    f.write(f"Type: {item['type']}\n")
                    f.write(f"Page: {item['page']}\n")
                    if item['type'] == 'image_caption':
                        f.write(f"Position: {item['position']}\n")
                    f.write(f"Content: {item['content']}\n\n")
        except Exception as e:
            llog("translation_gpt", f"Error occurred while writing to the file: {str(e)}", self.log_save_file_name)
            error_msg_for_user = "We encountered an issue while processing your PDF. This could be due to file corruption, unsupported formatting."
            error_msg_for_developer = f"Error occurred while writing to the file: {str(e)}"
            return error_msg_for_developer, error_msg_for_user
        return error_msg_for_developer, error_msg_for_user


# Usage
if __name__ == "__main__":
    processor = PDFProcessor()
    input_pdf = "/home/cb_manager/rfp_completeness/RFPs/كراسة الشروط والمواصفات والعروض.pdf"
    output_txt = "translated_1.txt"
    pdf_document = fitz.open(input_pdf) # Load binary data into PDF using Fitz
    llog("PDFProcessor", "Starting PDF processing", "main_process")
    processor.process_pdf(pdf_document, output_txt, "main_process")
    llog("PDFProcessor", "PDF processing completed", "main_process")