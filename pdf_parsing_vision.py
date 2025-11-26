import pdfplumber
import fitz
import os
import base64
import tempfile
from dotenv import load_dotenv
from logger import custom_logger as llog # Revert to llog import
from openai import OpenAI
import re
from tqdm import tqdm

class PDFParser():
    def __init__(self, log_save_file_name: str = "custom_logs", max_workers: int = 20):
        load_dotenv()
        self.log_save_file_name = log_save_file_name
        self.max_workers = max_workers

    def __get_openrouter_client(self):
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            llog("PDFParser", "OPENROUTER_API_KEY environment variable is required but not set", self.log_save_file_name)
            raise ValueError("OPENROUTER_API_KEY environment variable is required but not set")
        return OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )

    def __encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def __render_page_to_image(self, page, log_save_file_name: str):
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            temp_img_path = tmp.name
        zoom = 4
        mat = fitz.Matrix(zoom, zoom)
        try:
            pix = page.get_pixmap(matrix=mat, alpha=False)
            pix.save(temp_img_path)
            if os.path.getsize(temp_img_path) > 1000:
                return temp_img_path
        except Exception as e:
            llog("PDFParser", f"Error rendering page to image: {str(e)}", log_save_file_name)
        if os.path.exists(temp_img_path):
            os.remove(temp_img_path)
        return None

    def __prompt_extract_from_image(self, image_path: str, log_save_file_name: str) -> str:
        base64_image = self.__encode_image(image_path)
        openrouter_client = self.__get_openrouter_client()

        prompt = """
        You are a precise document transcriber. Your task is to convert this image into text exactly as it appears, maintaining all formatting and content. Please:

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

        5. Look carefully for any faint text, watermarks, or security patterns.

        Output the content exactly as it appears, maintaining the original structure and format in markdown.
        """

        try:
            response = openrouter_client.chat.completions.create(
                model="qwen/qwen2.5-vl-72b-instruct",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ]
                    }
                ],
                max_tokens=3000,
                temperature=0.05,
                top_p=0.25,
                seed=42,
                timeout=120
            )
            content = getattr(response.choices[0].message, "content", None)
            if not content:
                llog("PDFParser", "OpenRouter API returned empty content", log_save_file_name)
                return "[Error: Empty content from OpenRouter API]"
            regex = r'```markdown\n|```'
            return re.sub(regex, '', content)
        except Exception as e:
            llog("PDFParser", f"Error extracting text from image: {str(e)}", log_save_file_name)
            return f"[Error extracting text: {str(e)}]"

    def __extract_text_from_page(self, plumber_page, fitz_page):
        text = ""
        try:
            text = plumber_page.extract_text(x_tolerance=2, y_tolerance=2)
        except Exception:
            pass
        if not text or not text.strip():
            try:
                text = fitz_page.get_text("text") or ""
            except Exception:
                text = ""
        if not text or len(text.strip()) < 15 or "(cid:" in text or re.search(r"\(cid:[0-9]+\)", text):
            return None
        return text.strip()

    def parse_pdfs_to_text(self, input_pdfs: list, output_txt_path: str, log_save_file_name: str = "custom_logs"):
        if log_save_file_name:
            self.log_save_file_name = log_save_file_name
        output = []

        for input_pdf in input_pdfs:
            pdf_plumber = pdfplumber.open(input_pdf)
            pdf_fitz = fitz.open(input_pdf)
            total_pages = len(pdf_plumber.pages)
            page_results = [None] * total_pages
            to_api = []

            for i in tqdm(range(total_pages), desc=f"Scanning {os.path.basename(input_pdf)}"):
                try:
                    plumber_page = pdf_plumber.pages[i]
                    fitz_page = pdf_fitz[i]
                    text = self.__extract_text_from_page(plumber_page, fitz_page)
                    if text:
                        page_results[i] = text
                    else:
                        image_path = self.__render_page_to_image(fitz_page, log_save_file_name)
                        if image_path:
                            to_api.append((i, image_path))
                        else:
                            page_results[i] = "[Error: Could not render page image for OCR]"
                except Exception as e:
                    llog("PDFParser", f"Error processing page {i}: {e}", log_save_file_name)
                    page_results[i] = f"[Error processing page {i}]"
            pdf_plumber.close()
            pdf_fitz.close()

            def call_api(idx_img):
                idx, img_path = idx_img
                try:
                    llm_text = self.__prompt_extract_from_image(img_path, log_save_file_name)
                finally:
                    try:
                        os.remove(img_path)
                    except Exception:
                        pass
                return (idx, llm_text)

            if to_api:
                for idx, img_path in tqdm(to_api, desc="LLM OCR (sequential)"):
                    llm_text = self.__prompt_extract_from_image(img_path, log_save_file_name)
                    page_results[idx] = llm_text

            output.extend(page_results)

        try:
            with open(output_txt_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(output))
        except Exception as write_error:
            llog("PDFParser", f"Error writing output file: {str(write_error)}", log_save_file_name)
        return output

if __name__ == "__main__":
    input_pdf_paths = [
        "/home/aman/Downloads/image_pdf_sample_testing.pdf"
    ]
    output_txt_path = "output/02/tmp_output_1.txt"
    document_parser = PDFParser(log_save_file_name="test_pdf_parsing_vision_log", max_workers=20)
    output = document_parser.parse_pdfs_to_text(
        input_pdf_paths,
        output_txt_path,
        "test_pdf_parsing_vision_log"
    )
    llog("PDFParser", f"Length of output: {len(output)}", "test_pdf_parsing_vision_log")
    llog("PDFParser", f"Output content: {output}", "test_pdf_parsing_vision_log")
