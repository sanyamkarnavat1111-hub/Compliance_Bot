# Do add the table like support as that of the old file and then integrate it with the pipeline.py

import os
import base64
import io # Added for BytesIO
from openai import OpenAI
from pdf2image import convert_from_path
from dotenv import load_dotenv
from logger import custom_logger as llog # Use llog for logging

class PDFVisionProcessor:
    """
    A class for processing PDF documents using vision models to extract and translate text.
    Supports multi-language text extraction with translation to English and table preservation.
    """
    
    def __init__(self, model_name: str = "qwen/qwen2.5-vl-72b-instruct", log_save_file_name: str = "custom_logs"):
        """
        Initialize the PDFVisionProcessor with the specified model.
        
        Args:
            model_name: Name of the vision model to use (default: "qwen/qwen2.5-vl-72b-instruct")
            log_save_file_name: Name of the log file for this process.
            
        Raises:
            ValueError: If OPENROUTER_API_KEY is not found in environment variables
        """
        load_dotenv()
        self.log_save_file_name = log_save_file_name
        
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        if not openrouter_api_key:
            llog("PDFVisionProcessor", "OPENROUTER_API_KEY environment variable not found. Please set it.", self.log_save_file_name)
            raise ValueError(
                "OPENROUTER_API_KEY not found in environment variables. "
                "Please set it in a .env file in the same directory as this script. "
                "Example: OPENROUTER_API_KEY=\"your_openrouter_key_here\""
            )
        
        self.model_name = model_name
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=openrouter_api_key,
        )
        llog("PDFVisionProcessor", f"Initialized with model: {self.model_name}", self.log_save_file_name)

    def pdf_to_images(self, pdf_path: str):
        """Converts each page of a PDF to a list of PIL Image objects."""
        llog("PDFVisionProcessor", f"Converting PDF '{pdf_path}' to images...", self.log_save_file_name)
        try:
            images = convert_from_path(pdf_path)
            llog("PDFVisionProcessor", f"Successfully converted {len(images)} page(s) from PDF.", self.log_save_file_name)
            return images
        except Exception as e:
            llog("PDFVisionProcessor", f"Error converting PDF to images: {e}", self.log_save_file_name)
            llog("PDFVisionProcessor", "Please ensure 'poppler' is installed and in your system's PATH.", self.log_save_file_name)
            raise e

    def image_to_base64(self, image, format="PNG"):
        """Converts a PIL Image object to a base64 encoded string."""
        buffered = io.BytesIO()
        image.save(buffered, format=format)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_str

    def get_text_from_image_via_openrouter(self, image_base64: str):
        """
        Sends an image to the Qwen VL model via OpenRouter and returns the
        extracted/translated text.
        """
        llog("PDFVisionProcessor", f"Sending image to {self.model_name} for processing...", self.log_save_file_name)
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    """
                                    You are a precise document parser.

                                    Your task is to analyze the attached image and extract **only the core body content** (ignore headers, footers, logos, or page numbers), and convert it into clean English. Follow these detailed rules:

                                    ---

                                    ### 1. LANGUAGE RULES — ENGLISH ONLY OUTPUT:

                                    * Translate all non-English text (Arabic, Chinese, Korean, etc.) into **fluent, accurate English**.
                                    * Remove all non-English characters entirely. Output must be **100% English**.
                                    * If text is already in English, preserve it as-is.

                                    ---

                                    ### 2. FOCUS AREA — IGNORE PERIPHERALS:

                                    Only extract **main content** from the center of the page. Skip:

                                    * Document headers, footers, page numbers
                                    * Watermarks, logos, decorative elements
                                    * Titles that are not part of main text or tables

                                    ---

                                    ### 3. TABLE HANDLING RULES — MARKDOWN FORMAT:

                                    If any **tables** are found:

                                    * Extract all rows and columns **accurately** and present them in **markdown table format**.
                                    * Use pipe (`|`) separators for columns and maintain headers at the top.
                                    * Preserve row alignment and content fidelity.
                                    * If any **cell contains multi-line text** (like bullet points or multiple sentences), format it using:

                                    * `<br>` for line breaks **within the same cell**, or
                                    * Markdown bullet points if appropriate
                                    * **Do not break rows unintentionally** by misinterpreting multi-line cell content.

                                    If a table includes **right-aligned or right-to-left columns** (e.g., Arabic labels on the right), ensure the **first column is treated as the header** and the corresponding row data is placed in the second column, left to right.

                                    ---

                                    ### 4. VISUAL ELEMENTS:

                                    If the image contains **charts, graphs, or diagrams**, describe them with the following format:

                                    ```
                                    **[IMAGE DESCRIPTION: type of visual – detailed description of content, axes, values, and labels]**
                                    ```

                                    Place image descriptions at the appropriate location in the flow.

                                    ---

                                    ### 5. FINAL OUTPUT STRUCTURE:

                                    * Present clean **markdown-formatted tables** for any tabular content
                                    * Keep non-tabular content (if any) in clean English paragraphs
                                    * Do **not include any prompt text, commentary, or instructions**
                                    * Final output must contain only clean English — absolutely no Arabic, Chinese, Korean, Russian, or other scripts
                                    * Double-check for non-English remnants and remove if any

                                    ---

                                    ### 6. EXAMPLE MARKDOWN TABLE (for reference):

                                    ```
                                    | Field                      | Value                                                  |
                                    |---------------------------|--------------------------------------------------------|
                                    | Language                  | ArchiMate                                              |
                                    | Monitoring Tools          | Native cloud monitoring, proactive security auditing   |
                                    | SLA                       | 99.9% uptime, max downtime 45 minutes per year         |
                                    | Preferred Technologies    | Go, Node.js, Python, Java, .Net                        |
                                    | Deliverables              | - UAT Plan<br>- BRD<br>- Low-Level Design<br>...       |
                                    ```

                                    ---

                                    Now, perform the task accordingly and return **only the final cleaned result** (translated, formatted, and structured as per the instructions).
                                """
                                )
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=2000,
                temperature=0.05,
                top_p=0.25,
                seed=42
            )
            if response.choices and response.choices[0].message and response.choices[0].message.content:
                processed_text = response.choices[0].message.content.strip()
                llog("PDFVisionProcessor", "Successfully received response from model.", self.log_save_file_name)
                return processed_text
            else:
                llog("PDFVisionProcessor", "Warning: No content found in API response.", self.log_save_file_name)
                return "Error: No content in API response"
        except Exception as e:
            llog("PDFVisionProcessor", f"Error calling OpenRouter API with model {self.model_name}: {e}", self.log_save_file_name)
            return f"Error processing image via API: {e}"

    def pdf_vision_process(self, pdf_path: str):
        if not isinstance(pdf_path, str):
            llog("PDFVisionProcessor", f"Expected a single file path (str), but got: {type(pdf_path)}", self.log_save_file_name)
            raise ValueError(f"Expected a single file path (str), but got: {type(pdf_path)}")
            
        llog("PDFVisionProcessor", f"Starting to process PDF: {pdf_path}", self.log_save_file_name)
        
        pdf_path = os.path.abspath(pdf_path)
        
        if not os.path.isfile(pdf_path):
            llog("PDFVisionProcessor", f"PDF file not found at: {pdf_path}", self.log_save_file_name)
            raise FileNotFoundError(f"PDF file not found at: {pdf_path}")
            
        if not pdf_path.lower().endswith('.pdf'):
            llog("PDFVisionProcessor", f"File is not a PDF: {pdf_path}", self.log_save_file_name)
            raise ValueError(f"File is not a PDF: {pdf_path}")

        try:
            images = self.pdf_to_images(pdf_path)
            if not images:
                llog("PDFVisionProcessor", f"Warning: No pages found in PDF: {pdf_path}", self.log_save_file_name)
                return []
                
        except Exception as e:
            llog("PDFVisionProcessor", f"Error converting PDF to images: {str(e)}", self.log_save_file_name)
            return []

        all_page_texts = [None] * len(images)
        for idx, image in enumerate(images):
            try:
                text_content = self.get_text_from_image_via_openrouter(self.image_to_base64(image))
                all_page_texts[idx] = text_content
                llog("PDFVisionProcessor", f"Text from page {idx+1} extracted successfully.", self.log_save_file_name)
            except Exception as e:
                llog("PDFVisionProcessor", f"An unexpected error occurred while processing page {idx+1}: {e}", self.log_save_file_name)
                all_page_texts[idx] = f"Error processing page {idx+1}: {e}"
        llog("PDFVisionProcessor", "Finished processing all pages.", self.log_save_file_name)
        return all_page_texts

    def process_and_save(self, pdf_path, output_filename: str = "output/extracted_pdf_texts.txt", log_save_file_name: str = "custom_logs"):
        if isinstance(pdf_path, str):
            pdf_paths = [pdf_path]
        else:
            pdf_paths = pdf_path
            
        all_texts = []
        
        for path in pdf_paths:
            llog("PDFVisionProcessor", f"Processing PDF: {path}", log_save_file_name)
            try:
                texts = self.pdf_vision_process(path)
                all_texts.extend(texts)
            except Exception as e:
                llog("PDFVisionProcessor", f"Error processing {path}: {str(e)}", log_save_file_name)
                continue
                
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        
        with open(output_filename, 'w', encoding='utf-8') as f:
            for i, text in enumerate(all_texts, 1):
                f.write(f"--- Page {i} ---\n{text}\n\n")
        
        llog("PDFVisionProcessor", f"Successfully processed {len(all_texts)} pages. Results saved to {output_filename}", log_save_file_name)
        return all_texts

if __name__ == "__main__":
    pdf_processor = PDFVisionProcessor(log_save_file_name="test_vision_processor_log")
    pdf_file_path = "sample_data/03/RFP.pdf"
    
    extracted_texts = pdf_processor.process_and_save(pdf_file_path, log_save_file_name="test_vision_processor_log")
    llog("PDFVisionProcessor", f"Extracted texts: {extracted_texts}", "test_vision_processor_log")



#   def pdf_to_images(self, pdf_path, chunk_size=5):
        # """
        # Converts each page of a potentially large PDF to a list of PIL Image objects.
        # Handles large PDFs by processing in chunks to avoid memory issues.
        # Merges all images into a single list and returns it.
        # """
    

        # print(f"Converting PDF '{pdf_path}' to images (chunked processing)...")
        # all_images = []
        # try:
        #     # First, get the total number of pages
        #     total_pages = _page_count(pdf_path)
        #     print(f"PDF has {total_pages} page(s). Processing in chunks of {chunk_size}...")

        #     for start in range(1, total_pages + 1, chunk_size):
        #         end = min(start + chunk_size - 1, total_pages)
        #         print(f"Processing pages {start} to {end}...")
        #         images_chunk = convert_from_path(pdf_path, first_page=start, last_page=end)
        #         all_images.extend(images_chunk)
        #         print(f"Processed pages {start}-{end}. Total images so far: {len(all_images)}")

        #     print(f"Successfully converted {len(all_images)} page(s) from PDF.")
        #     return all_images
        # except PDFPageCountError as e:
        #     print(f"Error: Could not determine page count for PDF: {e}")
        #     raise e
        # except Exception as e:
        #     print(f"Error converting PDF to images: {e}")
        #     print("Please ensure 'poppler' is installed and in your system's PATH.")
        #     print("  - On Ubuntu/Debian: sudo apt-get install poppler-utils")
        #     raise e

