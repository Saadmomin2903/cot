"""
Input Router - Automatic file type detection and routing.

Detects the input type and routes to the appropriate ingestor.
"""

import mimetypes
from typing import Union, Optional
from pathlib import Path
from dataclasses import dataclass
from enum import Enum


class InputType(Enum):
    """Supported input types."""
    TEXT = "text"
    PDF = "pdf"
    IMAGE = "image"
    UNKNOWN = "unknown"


@dataclass
class RoutedInput:
    """Result of routing an input."""
    input_type: InputType
    text: str
    metadata: dict
    source: str  # Original file path or "direct_text"


class InputRouter:
    """
    Route inputs to appropriate ingestors.
    
    Automatically detects:
    - PDF files → PDFIngestor
    - Image files → ImageIngestor
    - Text/unknown → Pass through
    
    Usage:
        router = InputRouter()
        result = router.route("document.pdf")
        print(result.text)  # Extracted text
    """
    
    IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.tiff'}
    PDF_EXTENSIONS = {'.pdf'}
    TEXT_EXTENSIONS = {'.txt', '.md', '.csv', '.json', '.xml', '.html'}
    
    def __init__(self, api_key: str = None):
        """
        Initialize router.
        
        Args:
            api_key: API key for vision models (if needed)
        """
        self.api_key = api_key
    
    def detect_type(self, file_path: str) -> InputType:
        """
        Detect the type of input file.
        
        Args:
            file_path: Path to file
            
        Returns:
            InputType enum value
        """
        path = Path(file_path)
        ext = path.suffix.lower()
        
        if ext in self.PDF_EXTENSIONS:
            return InputType.PDF
        elif ext in self.IMAGE_EXTENSIONS:
            return InputType.IMAGE
        elif ext in self.TEXT_EXTENSIONS:
            return InputType.TEXT
        else:
            # Try mime type
            mime, _ = mimetypes.guess_type(file_path)
            if mime:
                if mime == 'application/pdf':
                    return InputType.PDF
                elif mime.startswith('image/'):
                    return InputType.IMAGE
                elif mime.startswith('text/'):
                    return InputType.TEXT
            
            return InputType.UNKNOWN
    
    def detect_type_from_bytes(self, content: bytes, filename: str) -> InputType:
        """
        Detect type from bytes content.
        
        Args:
            content: File bytes
            filename: Original filename
            
        Returns:
            InputType enum
        """
        # Check magic bytes
        if content[:4] == b'%PDF':
            return InputType.PDF
        elif content[:8] == b'\x89PNG\r\n\x1a\n':
            return InputType.IMAGE
        elif content[:2] == b'\xff\xd8':  # JPEG
            return InputType.IMAGE
        elif content[:4] == b'GIF8':
            return InputType.IMAGE
        else:
            # Fall back to extension
            return self.detect_type(filename)
    
    def route(self, file_path: str) -> RoutedInput:
        """
        Route a file to the appropriate ingestor.
        
        Args:
            file_path: Path to file
            
        Returns:
            RoutedInput with extracted text
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        input_type = self.detect_type(file_path)
        
        if input_type == InputType.PDF:
            return self._process_pdf(file_path)
        elif input_type == InputType.IMAGE:
            return self._process_image(file_path)
        elif input_type == InputType.TEXT:
            return self._process_text(file_path)
        else:
            # Try as text
            return self._process_text(file_path)
    
    def route_bytes(self, content: bytes, filename: str) -> RoutedInput:
        """
        Route bytes content to appropriate ingestor.
        
        Args:
            content: File bytes
            filename: Original filename
            
        Returns:
            RoutedInput with extracted text
        """
        input_type = self.detect_type_from_bytes(content, filename)
        
        if input_type == InputType.PDF:
            return self._process_pdf_bytes(content, filename)
        elif input_type == InputType.IMAGE:
            return self._process_image_bytes(content, filename)
        else:
            # Decode as text
            try:
                text = content.decode('utf-8')
            except UnicodeDecodeError:
                text = content.decode('latin-1')
            
            return RoutedInput(
                input_type=InputType.TEXT,
                text=text,
                metadata={"filename": filename},
                source=filename
            )
    
    def _process_pdf(self, file_path: str) -> RoutedInput:
        """Process PDF file."""
        from .pdf_ingestor import PDFIngestor
        
        ingestor = PDFIngestor()
        result = ingestor.extract(file_path)
        
        return RoutedInput(
            input_type=InputType.PDF,
            text=result.to_markdown(),
            metadata={
                "page_count": result.page_count,
                **result.metadata
            },
            source=file_path
        )
    
    def _process_pdf_bytes(self, content: bytes, filename: str) -> RoutedInput:
        """Process PDF bytes."""
        from .pdf_ingestor import PDFIngestor
        
        ingestor = PDFIngestor()
        result = ingestor.extract_from_bytes(content)
        
        return RoutedInput(
            input_type=InputType.PDF,
            text=result.to_markdown(),
            metadata={
                "page_count": result.page_count,
                "filename": filename,
                **result.metadata
            },
            source=filename
        )
    
    def _process_image(self, file_path: str) -> RoutedInput:
        """Process image file."""
        from .image_ingestor import ImageIngestor
        
        ingestor = ImageIngestor(api_key=self.api_key)
        result = ingestor.analyze(file_path)
        
        return RoutedInput(
            input_type=InputType.IMAGE,
            text=result.to_markdown(),
            metadata={
                "dimensions": result.dimensions,
                "format": result.format,
                **result.metadata
            },
            source=file_path
        )
    
    def _process_image_bytes(self, content: bytes, filename: str) -> RoutedInput:
        """Process image bytes."""
        from .image_ingestor import ImageIngestor
        
        ingestor = ImageIngestor(api_key=self.api_key)
        result = ingestor.analyze_from_bytes(content, filename)
        
        return RoutedInput(
            input_type=InputType.IMAGE,
            text=result.to_markdown(),
            metadata={
                "dimensions": result.dimensions,
                "format": result.format,
                "filename": filename
            },
            source=filename
        )
    
    def _process_text(self, file_path: str) -> RoutedInput:
        """Process text file."""
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            text = f.read()
        
        return RoutedInput(
            input_type=InputType.TEXT,
            text=text,
            metadata={"file_path": file_path},
            source=file_path
        )


# ============== Quick-use function ==============

def route_input(file_path: str, api_key: str = None) -> str:
    """
    Quick function to route and extract text from a file.
    
    Args:
        file_path: Path to file
        api_key: Optional API key for vision
        
    Returns:
        Extracted text as string
    """
    router = InputRouter(api_key=api_key)
    result = router.route(file_path)
    return result.text
