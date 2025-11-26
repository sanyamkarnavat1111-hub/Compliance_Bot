# Tendor POC: AI-Powered RFP Evaluation and Compliance Analysis System

## Project Overview
Tendor POC is an advanced AI-driven system designed to automate and enhance the Request for Proposal (RFP) evaluation process. Leveraging state-of-the-art machine learning and natural language processing techniques, this project enables intelligent analysis of PDF documents, multilingual support, and comprehensive compliance assessment.

## Key Features
- 🔍 **Intelligent PDF Processing**
  - Extract text, tables, and images from PDF documents
  - Support for multiple PDF files and multilingual content
  - Advanced parsing using PyMuPDF and pdfplumber

- 🧠 **Advanced Embedding and Semantic Search**
  - Text embedding generation using state-of-the-art models (Nomic Embed)
  - Semantic similarity search with FAISS indexing
  - Normalized embedding generation for accurate matching

- 🌐 **Multilingual Support**
  - Arabic and English language detection
  - Cross-language document processing
  - Intelligent translation and interpretation capabilities

- 📊 **Comprehensive Evaluation Pipeline**
  - Automated RFP compliance assessment
  - Detailed HTML report generation
  - Customizable evaluation criteria

## Technical Architecture
### Components
1. **PDF Parsing** (`pdf_parsing.py`)
   - Extracts text, tables, and images from PDF documents
   - Handles multiple document types and formats

2. **Embedding Management** (`embedding_management.py`)
   - Manages text embeddings using FAISS
   - Supports saving and loading embeddings
   - Provides semantic search capabilities

3. **Embedding Generation** (`embedding_generation.py`)
   - Generates text embeddings using Sentence Transformers
   - Supports normalization and dimensionality reduction

4. **Pipeline** (`pipeline.py`)
   - Orchestrates the entire document processing workflow
   - Integrates parsing, embedding, and evaluation modules

5. **RFP Evaluation Crew** (`crew.py`)
   - AI-powered evaluation system
   - Generates compliance reports and recommendations

## Dependencies
- Python 3.8+
- PyTorch
- Sentence Transformers
- FAISS
- pdfplumber
- PyMuPDF
- python-dotenv

## Usage
```python
from pipeline import Pipeline

pipeline = Pipeline()
pipeline.process_pdfs(
    rfp_pdf_path=["rfp.pdf"],
    ea_standard_pdf_path="standard.pdf",
    entity_path="entity.txt",
    tmp_output_path="outputs/"
)
```

## Configuration
Configure API keys and settings in a `.env` file:
```
GOOGLE_API_KEY=your_google_api_key
OPENAI_API_KEY=your_openai_api_key
```

## Performance Metrics
- Average Processing Time: ~5-10 minutes per document
- Embedding Dimension: 768
- Top-K Semantic Search: 5 most relevant chunks

## Future Roadmap
- [ ] Enhanced multilingual support
- [ ] Improved AI evaluation models
- [ ] Real-time collaborative editing
- [ ] Advanced visualization of compliance metrics

## Contributions
Contributions are welcome! Please read our contributing guidelines before submitting a pull request.

## License
This project is licensed under the MIT License.