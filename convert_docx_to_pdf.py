import subprocess
import os
from pathlib import Path
from logger_config import get_logger # Import get_logger

# Module-level logger
module_logger = get_logger(__name__)

def convert_docx_to_pdf(docx_path, pdf_path=None, method='libreoffice', logger=None):
    """
    Convert DOCX file to PDF on Linux systems.
    
    Args:
        docx_path (str): Path to the input DOCX file
        pdf_path (str, optional): Path for output PDF file. If None, uses same name as input
        method (str): Conversion method ('libreoffice', 'pandoc', or 'python-docx2pdf')
        logger: Logger instance to use for logging
    
    Returns:
        str or None: Path to the converted PDF file if successful, or original PDF path if already a PDF, None otherwise.
    """
    if logger is None:
        logger = module_logger

    # Validate input file
    if not os.path.exists(docx_path):
        logger.error(f"Error: Input file '{docx_path}' not found")
        return None

    # If the input file is already a PDF, return its path and skip conversion
    if docx_path.lower().endswith('.pdf'):
        logger.info(f"Info: Input file '{docx_path}' is already a PDF. Skipping conversion.")
        return docx_path
    
    if not docx_path.lower().endswith('.docx'):
        logger.error("Error: Input file must be a .docx file")
        return None
    
    # Set output path if not provided
    if pdf_path is None:
        pdf_path = str(Path(docx_path).with_suffix('.pdf'))
    
    try:
        if method == 'libreoffice':
            return _convert_with_libreoffice(docx_path, pdf_path, logger)
        elif method == 'pandoc':
            return _convert_with_pandoc(docx_path, pdf_path, logger)
        elif method == 'python-docx2pdf':
            return _convert_with_docx2pdf(docx_path, pdf_path, logger)
        else:
            logger.error(f"Error: Unknown method '{method}'")
            return None
            
    except Exception as e:
        logger.error(f"Error during conversion: {str(e)}")
        return None

def _convert_with_libreoffice(docx_path, pdf_path, logger):
    """Convert using LibreOffice (most reliable method)"""
    try:
        # Get output directory
        output_dir = os.path.dirname(pdf_path) or '.'
        
        # Run LibreOffice conversion
        cmd = [
            'libreoffice',
            '--headless',
            '--convert-to', 'pdf',
            '--outdir', output_dir,
            docx_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            # LibreOffice creates PDF with same name as input file
            default_pdf = os.path.join(output_dir, Path(docx_path).stem + '.pdf')
            
            # Rename if custom output path specified
            if default_pdf != pdf_path:
                os.rename(default_pdf, pdf_path)
            
            logger.info(f"Successfully converted '{docx_path}' to '{pdf_path}'")
            return pdf_path
        else:
            logger.error(f"LibreOffice conversion failed: {result.stderr}")
            return None
            
    except subprocess.TimeoutExpired:
        logger.error("LibreOffice conversion timed out")
        return None
    except FileNotFoundError:
        logger.error("LibreOffice not found. Install with: sudo apt-get install libreoffice")
        return None

def _convert_with_pandoc(docx_path, pdf_path, logger):
    """Convert using Pandoc"""
    try:
        cmd = ['pandoc', docx_path, '-o', pdf_path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            logger.info(f"Successfully converted '{docx_path}' to '{pdf_path}'")
            return pdf_path
        else:
            logger.error(f"Pandoc conversion failed: {result.stderr}")
            return None
            
    except FileNotFoundError:
        logger.error("Pandoc not found. Install with: sudo apt-get install pandoc texlive-latex-recommended")
        return None

def _convert_with_docx2pdf(docx_path, pdf_path, logger):
    """Convert using python-docx2pdf library"""
    try:
        from docx2pdf import convert
        convert(docx_path, pdf_path)
        logger.info(f"Successfully converted '{docx_path}' to '{pdf_path}'")
        return pdf_path
    except ImportError:
        logger.error("docx2pdf not installed. Install with: pip install docx2pdf")
        return None
    except Exception as e:
        logger.error(f"docx2pdf conversion failed: {str(e)}")
        return None

def batch_convert_docx_to_pdf(input_directory, output_directory=None, method='libreoffice', logger=None):
    """
    Convert all DOCX files in a directory to PDF.
    
    Args:
        input_directory (str): Directory containing DOCX files
        output_directory (str, optional): Directory for PDF output. If None, uses input directory
        method (str): Conversion method
        logger: Logger instance to use for logging
    
    Returns:
        list: List of successfully converted files
    """
    if logger is None:
        logger = module_logger

    if output_directory is None:
        output_directory = input_directory
    
    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    
    converted_files = []
    # Combine DOCX and PDF files for processing
    files_to_process = list(Path(input_directory).glob("*.docx")) + list(Path(input_directory).glob("*.pdf"))
    
    if not files_to_process:
        logger.warning(f"No DOCX or PDF files found in '{input_directory}'")
        return converted_files
    
    for input_file in files_to_process:
        # Determine output PDF path based on input file type
        if input_file.suffix.lower() == '.pdf':
            # If already a PDF, use its own path as the "converted" path
            converted_pdf_path = str(input_file)
        else:
            # For DOCX, define the output PDF path in the same directory as the input DOCX
            pdf_file = input_file.parent / f"{input_file.stem}.pdf"
            converted_pdf_path = convert_docx_to_pdf(str(input_file), str(pdf_file), method, logger=logger) # Pass logger
            
        if converted_pdf_path:
            converted_files.append(converted_pdf_path)
    
    logger.info(f"Processed {len(converted_files)} out of {len(files_to_process)} files")
    return converted_files

# Example usage
if __name__ == "__main__":
    # Single file conversion: PDF will be saved in the same directory as the input DOCX
    success_single = convert_docx_to_pdf("/home/ubuntu/Tendor_POC/tendor_poc/input_data/6747f7dd5c51cgK33W1732769757.pdf", logger=module_logger) # Pass module logger
    if success_single:
        module_logger.info(f"Single file conversion successful. PDF saved at: {success_single}")
    else:
        module_logger.error("Single file conversion failed.")
    
    # Batch conversion
    # converted = batch_convert_docx_to_pdf("./documents", "./pdfs")
