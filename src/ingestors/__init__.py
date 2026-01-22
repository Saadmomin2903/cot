"""
Ingestors Module - Multi-modal input processing.

Supports:
- PDF extraction (text + tables)
- Image description (via Vision LLM)
- Automatic file type routing
"""

from .pdf_ingestor import PDFIngestor, extract_pdf_text
from .image_ingestor import ImageIngestor, describe_image
from .router import InputRouter, route_input

__all__ = [
    "PDFIngestor",
    "ImageIngestor", 
    "InputRouter",
    "extract_pdf_text",
    "describe_image",
    "route_input",
]
