"""
PDF Ingestor - Extract text and tables from PDF documents.

Uses pypdf for basic text extraction with fallback options.
"""

import io
import re
from typing import Dict, Any, List, Optional
from pathlib import Path
from dataclasses import dataclass


@dataclass
class PDFPage:
    """Represents a single PDF page."""
    page_number: int
    text: str
    has_images: bool = False
    has_tables: bool = False


@dataclass 
class PDFResult:
    """Result of PDF extraction."""
    text: str
    page_count: int
    pages: List[PDFPage]
    metadata: Dict[str, Any]
    
    def to_markdown(self) -> str:
        """Convert to Markdown format."""
        parts = []
        if self.metadata.get("title"):
            parts.append(f"# {self.metadata['title']}\n")
        
        for page in self.pages:
            parts.append(f"## Page {page.page_number}\n")
            parts.append(page.text)
            parts.append("\n---\n")
        
        return "\n".join(parts)


class PDFIngestor:
    """
    PDF text extraction with multiple strategies.
    
    Features:
    - Basic text extraction via pypdf
    - Metadata extraction (title, author, etc.)
    - Page-by-page processing
    - Markdown output format
    
    Usage:
        ingestor = PDFIngestor()
        result = ingestor.extract("document.pdf")
        print(result.text)
        print(result.to_markdown())
    """
    
    def __init__(self, extract_images: bool = False):
        """
        Initialize PDF ingestor.
        
        Args:
            extract_images: Whether to note image locations (not extract content)
        """
        self.extract_images = extract_images
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check if required libraries are available."""
        try:
            import pypdf
            self._pypdf_available = True
        except ImportError:
            self._pypdf_available = False
    
    def extract(self, file_path: str) -> PDFResult:
        """
        Extract text from PDF file.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            PDFResult with extracted text and metadata
        """
        if not self._pypdf_available:
            raise ImportError("pypdf is required. Install with: pip install pypdf")
        
        import pypdf
        
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        
        pages = []
        all_text = []
        
        with open(file_path, "rb") as f:
            reader = pypdf.PdfReader(f)
            
            # Extract metadata
            metadata = self._extract_metadata(reader)
            
            # Process each page
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                
                # Clean the text
                page_text = self._clean_text(page_text)
                
                # Check for images
                has_images = len(page.images) > 0 if hasattr(page, 'images') else False
                
                # Simple table detection (heuristic)
                has_tables = self._detect_tables(page_text)
                
                pages.append(PDFPage(
                    page_number=i + 1,
                    text=page_text,
                    has_images=has_images,
                    has_tables=has_tables
                ))
                
                all_text.append(page_text)
        
        return PDFResult(
            text="\n\n".join(all_text),
            page_count=len(pages),
            pages=pages,
            metadata=metadata
        )
    
    def extract_from_bytes(self, pdf_bytes: bytes) -> PDFResult:
        """
        Extract text from PDF bytes (for file uploads).
        
        Args:
            pdf_bytes: PDF file content as bytes
            
        Returns:
            PDFResult with extracted text
        """
        if not self._pypdf_available:
            raise ImportError("pypdf is required. Install with: pip install pypdf")
        
        import pypdf
        
        pages = []
        all_text = []
        
        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
        metadata = self._extract_metadata(reader)
        
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            page_text = self._clean_text(page_text)
            
            has_images = len(page.images) > 0 if hasattr(page, 'images') else False
            has_tables = self._detect_tables(page_text)
            
            pages.append(PDFPage(
                page_number=i + 1,
                text=page_text,
                has_images=has_images,
                has_tables=has_tables
            ))
            
            all_text.append(page_text)
        
        return PDFResult(
            text="\n\n".join(all_text),
            page_count=len(pages),
            pages=pages,
            metadata=metadata
        )
    
    def _extract_metadata(self, reader) -> Dict[str, Any]:
        """Extract PDF metadata."""
        meta = reader.metadata or {}
        return {
            "title": meta.get("/Title", ""),
            "author": meta.get("/Author", ""),
            "subject": meta.get("/Subject", ""),
            "creator": meta.get("/Creator", ""),
            "producer": meta.get("/Producer", ""),
            "creation_date": str(meta.get("/CreationDate", "")),
        }
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text."""
        # Fix common extraction issues
        text = re.sub(r'\s+', ' ', text)  # Collapse whitespace
        text = re.sub(r'(\w)-\s+(\w)', r'\1\2', text)  # Fix hyphenation
        text = text.strip()
        return text
    
    def _detect_tables(self, text: str) -> bool:
        """
        Simple heuristic to detect if text might contain tables.
        """
        # Check for patterns like repeated tabs or multiple columns
        lines = text.split('\n')
        tab_lines = sum(1 for line in lines if '\t' in line or '  ' in line)
        return tab_lines > 3


# ============== Quick-use function ==============

def extract_pdf_text(file_path: str) -> str:
    """
    Quick function to extract text from PDF.
    
    Args:
        file_path: Path to PDF file
        
    Returns:
        Extracted text as string
    """
    ingestor = PDFIngestor()
    result = ingestor.extract(file_path)
    return result.text
