
import pdfplumber
import fitz  # PyMuPDF
from tqdm import tqdm
import os
from dotenv import load_dotenv
import tempfile
from PIL import Image
import base64
from openai import OpenAI
import re
import time
from logger import custom_logger as llog
from typing import Optional, Tuple

class PDFStandardParser:
    def __init__(self, log_save_file_name: str, model: str = "openai"):
        """
        Initialize the PDF parser for standard documents using Qwen-VL 2.5 32B or OpenAI GPT-4o.
        
        Args:
            log_save_file_name: Unique identifier for logging.
            model: Model to use, either "opensource" (Qwen-VL 2.5) or "openai" (GPT-4o). Defaults to "opensource".
        """
        load_dotenv()
        
        # Validate model selection
        if model not in ["opensource", "openai"]:
            llog("PDFStandardParser", f"Invalid model '{model}', defaulting to opensource", log_save_file_name)
            model = "opensource"
        
        llog("PDFStandardParser", f"Initializing PDF parser with model: {model}", log_save_file_name)
        self.log_save_file_name = log_save_file_name
        
        # Initialize OpenAI client for OpenRouter or OpenAI
        if model == "opensource":
            self.client = OpenAI(
                api_key=os.getenv("OPENROUTER_API_KEY"),
                base_url="https://openrouter.ai/api/v1"
            )
            self.model_name = "qwen/qwen2.5-vl-32b-instruct"
        else:  # openai
            self.client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url="https://api.openai.com/v1"
            )
            llog("PDFStandardParser", f"Using OpenAI GPT-4o model {os.getenv('OPENAI_API_KEY')}", self.log_save_file_name)
            self.model_name = "gpt-4o"
        
        llog("PDFStandardParser", f"Initialized with model: {self.model_name}", self.log_save_file_name)
        
        self.chinese_pattern = r'[\u4e00-\u9fff]'

    def __encode_image(self, image_path: str, log_save_file_name: str) -> str:
        """Encode an image to base64 for API calls."""
        llog("PDFStandardParser", f"Encoding image: {image_path}", log_save_file_name)
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def __render_page_to_image(self, page, page_num: int, log_save_file_name: str, attempt: int = 1, max_attempts: int = 3) -> Optional[str]:
        """Render a PDF page to an image with retry logic."""
        llog("PDFStandardParser", f"Rendering page {page_num + 1} to image (attempt {attempt}/{max_attempts})", log_save_file_name)
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            temp_img_path = tmp.name
        
        zoom = 4 + (attempt - 1) * 2
        mat = fitz.Matrix(zoom, zoom)
        try:
            pix = page.get_pixmap(matrix=mat, alpha=False)
            pix.save(temp_img_path)
            if os.path.exists(temp_img_path) and os.path.getsize(temp_img_path) > 1000:
                return temp_img_path
            else:
                llog("PDFStandardParser", f"Warning: Rendered image too small: {os.path.getsize(temp_img_path)} bytes", log_save_file_name)
                if attempt < max_attempts:
                    llog("PDFStandardParser", f"Retrying render for page {page_num + 1}", log_save_file_name)
                    if os.path.exists(temp_img_path):
                        os.remove(temp_img_path)
                    return self.__render_page_to_image(page, page_num, log_save_file_name, attempt + 1, max_attempts)
                return None
        except Exception as e:
            llog("PDFStandardParser", f"Error rendering page to image: {str(e)}", log_save_file_name)
        if attempt < max_attempts:
            llog("PDFStandardParser", f"Retrying render for page {page_num + 1}", log_save_file_name)
            if os.path.exists(temp_img_path):
                os.remove(temp_img_path)
            return self.__render_page_to_image(page, page_num, log_save_file_name, attempt + 1, max_attempts)
        
        if os.path.exists(temp_img_path):
            os.remove(temp_img_path)
        return None

    def __generate_page_content(self, image_path: str, page_num: int, model: str, log_save_file_name: str) -> Tuple[str, bool]:
        """Extract text from a page image using Qwen-VL 2.5 32B or OpenAI GPT-4o."""
        llog("PDFStandardParser", f"Generating content for page {page_num + 1} with model {model}", log_save_file_name)
        base64_image = self.__encode_image(image_path, log_save_file_name)
        
        prompt = """You are a precise document transcriber for a standard compliance document. Your task is to extract ALL content from this PDF page image and transcribe it EXCLUSIVELY in English ONLY, maintaining exact formatting and structure. Follow these strict instructions:

        1. **Transcription**:
           - Extract ALL visible text, including:
             - Headers, footers, and sidebars
             - Main body text
             - Table contents
             - Bullet points, numbered lists, and nested lists
             - Numbers, references, citations, and technical terms (e.g., "Microsoft SQL Server 2022")
           - Transcribe ALL text exactly as it appears, including repetitive or nonsensical text.
           - If text is in Arabic, translate it accurately to English and include the original Arabic in brackets: [Arabic: original term].
           - Correct OCR errors (e.g., "Oracleı" → "Oracle", "الهacker الإلكترونية" → "cyberattacks [Arabic: الهجمات الإلكترونية]").
           - Preserve ALL technical terms exactly as written (e.g., "Oracle", not "Oracleı").

        2. **Formatting**:
           - Use markdown to preserve structure:
             - Tables: Use markdown table syntax (e.g., | Col1 | Col2 |).
             - Bullet points: Use `-` or `*` for unordered, `1.` for ordered.
             - Paragraphs: Separate with blank lines.
             - Headings: Use `#` for hierarchy (e.g., `# Header1`, `## Header2`).
           - Maintain text alignment (left/center/right) via markdown notation if applicable.
           - Correct OCR formatting errors (e.g., "،" → ",").

        3. **Images and Graphics**:
           - Describe image location (e.g., "Top-right corner") and content (e.g., "Diagram of network architecture").
           - Transcribe any text or labels within images.
           - Note if the image is decorative (e.g., "[Decorative: Company logo]").

        4. **Edge Cases**:
           - For faint text, watermarks, or security patterns, transcribe ALL visible text and note: [Note: Possible watermark or faint text].
           - For unclear or partially visible text, transcribe what’s visible and note: [Note: Unclear text].
           - If the page is blank, output: [Note: Blank page].
           - Do NOT filter or summarize repetitive text; transcribe it exactly.
           - If a table or list spans multiple pages, note: [Continued on next page].

        5. **Language**:
           - Output EXCLUSIVELY in English. Do NOT include Chinese characters or other languages except as noted (e.g., [Arabic: original term]).
           - For non-English text, provide accurate translations and retain original terms in brackets.

        6. **Validation**:
           - Ensure no content is omitted. Review the entire page carefully.
           - Do NOT discard any text, even if repetitive or nonsensical.

        Output only the transcribed content in markdown format. Do not include markdown code block delimiters (```).
        NOTE* RESULT IS IN ENGLISH LANGUAGE ONLY.
        """
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                llog("PDFStandardParser", f"Sending request to {model} API (attempt {attempt + 1}/{max_retries})", log_save_file_name)
                
                if model == "opensource":
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt},
                                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                                ]
                            }
                        ],
                        max_tokens=6000,
                        temperature=0.1,
                        timeout=120
                    )
                elif model == "openai":
                    llog("PDFStandardParser", f"EA_standard_parser: Sending request to OpenAI API (attempt {attempt + 1}/{max_retries})", log_save_file_name)
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt},
                                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                                ]
                            }
                        ],
                        max_tokens=6000,
                        temperature=0.1,
                        timeout=1200
                    )
                
                content = response.choices[0].message.content
                if not content or not isinstance(content, str):
                    raise ValueError("Invalid or empty response content")
                
                content = re.sub(r'^```markdown\s*\n?', '', content.strip())
                content = re.sub(r'\n?```$', '', content)
                
                if re.search(self.chinese_pattern, content):
                    llog("PDFStandardParser", f"Chinese characters detected: {re.findall(self.chinese_pattern, content)[:10]}", log_save_file_name)
                    content = re.sub(self.chinese_pattern, '[Removed: Chinese character]', content)
                
                llog("PDFStandardParser", f"Page {page_num + 1} content length: {len(content)} characters", log_save_file_name)
                
                return content or f"[Note: Blank page {page_num + 1}]", True
            
            except Exception as e:
                llog("PDFStandardParser", f"API attempt {attempt + 1} failed for page {page_num + 1}: {str(e)}", log_save_file_name)
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                llog("PDFStandardParser", f"Fallback: Failed to generate content for page {page_num + 1} after {max_retries} retries", log_save_file_name)
                return f"[Error extracting content: {str(e)}]", False

    def parse_pdf_to_text(self, input_pdf, output_txt_path: str, model: str = "openai", log_save_file_name: str = "custom_logs") -> Tuple[Optional[str], Optional[str]]:
        """
        Extract text from a standard PDF using Qwen-VL 2.5 32B or OpenAI GPT-4o, outputting in English.
        
        Args:
            input_pdf: File path (str) or PyMuPDF document object.
            output_txt_path: Path to save extracted text.
            model: Model to use, either "opensource" (Qwen-VL 2.5) or "openai" (GPT-4o). Defaults to "opensource".
            log_save_file_name: Name of the log file for this process.
        
        Returns:
            Tuple[Optional[str], Optional[str]]: (error_msg_for_developer, error_msg_for_user)
        """
        llog("PDFStandardParser", f"Starting PDF processing with model {model}", log_save_file_name)
        if model not in ["opensource", "openai"]:
            llog("PDFStandardParser", f"Invalid model '{model}', defaulting to openai", log_save_file_name)
            model = "openai"
    
        llog("PDFStandardParser", f"Starting PDF processing with model {model}: {input_pdf}", log_save_file_name)
        output = []
        
        try:
            if isinstance(input_pdf, str):
                llog("PDFStandardParser", f"Opening PDF: {input_pdf}", log_save_file_name)
                pdf_document = fitz.open(input_pdf)
            else:
                llog("PDFStandardParser", "Using provided PDF document", log_save_file_name)
                pdf_document = input_pdf
            
            if not pdf_document:
                llog("PDFStandardParser", "Failed to open PDF document", log_save_file_name)
                return "Failed to open PDF document", "Failed to open the document. Please ensure it's a valid PDF."
            
            llog("PDFStandardParser", f"Total pages: {len(pdf_document)}", log_save_file_name)
            
            for page_num in tqdm(range(len(pdf_document)), desc="Processing pages"):
                output.append(f"\n\n--- Page {page_num + 1} ---\n\n")
                page = pdf_document[page_num]
                
                image_path = self.__render_page_to_image(page, page_num, log_save_file_name)
                if not image_path:
                    output.append(f"[Failed to render page {page_num + 1} as image]")
                    continue
                
                try:
                    content, success = self.__generate_page_content(image_path, page_num, model, log_save_file_name)
                    output.append(content)
                    if not success:
                        llog("PDFStandardParser", f"Content extraction failed for page {page_num + 1}, using fallback content", log_save_file_name)
                
                finally:
                    if os.path.exists(image_path):
                        os.remove(image_path)
            
            try:
                with open(output_txt_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(output))
                llog("PDFStandardParser", f"Output written to: {output_txt_path}", log_save_file_name)
            except Exception as e:
                error_msg = f"Error writing output file: {str(e)}"
                llog("PDFStandardParser", error_msg, log_save_file_name)
                return error_msg, "Failed to save the processed text"
            
            return None, None
        
        except Exception as e:
            error_msg = f"Error processing PDF: {str(e)}"
            llog("PDFStandardParser", error_msg, log_save_file_name)
            return error_msg, "An error occurred while processing the PDF"  