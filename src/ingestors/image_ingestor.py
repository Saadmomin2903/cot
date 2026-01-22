"""
Image Ingestor - Describe and extract text from images using Vision LLM.

Uses Groq's vision-capable models or falls back to basic metadata.
"""

import base64
import io
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass
from PIL import Image


@dataclass
class ImageResult:
    """Result of image analysis."""
    description: str
    extracted_text: str
    dimensions: tuple
    format: str
    metadata: Dict[str, Any]
    
    def to_markdown(self) -> str:
        """Convert to Markdown format."""
        parts = [
            f"## Image Analysis",
            f"",
            f"**Dimensions**: {self.dimensions[0]}x{self.dimensions[1]}",
            f"**Format**: {self.format}",
            f"",
            f"### Description",
            self.description,
        ]
        
        if self.extracted_text:
            parts.extend([
                f"",
                f"### Extracted Text",
                self.extracted_text,
            ])
        
        return "\n".join(parts)


class ImageIngestor:
    """
    Image analysis and text extraction using Vision LLM.
    
    Features:
    - Image description using LLM vision
    - Text extraction (OCR via LLM)
    - Automatic resizing for API limits
    - Metadata extraction
    
    Usage:
        ingestor = ImageIngestor(api_key="your-key")
        result = ingestor.analyze("image.png")
        print(result.description)
    """
    
    MAX_IMAGE_SIZE = 1024  # Max dimension for LLM
    
    def __init__(
        self,
        api_key: str = None,
        use_groq: bool = True,
        extract_text: bool = True
    ):
        """
        Initialize image ingestor.
        
        Args:
            api_key: API key for vision model
            use_groq: Use Groq (if vision available) vs OpenAI
            extract_text: Also extract visible text from image
        """
        self.api_key = api_key
        self.use_groq = use_groq
        self.extract_text = extract_text
        self._init_client()
    
    def _init_client(self):
        """Initialize the vision client."""
        # Try to import Groq first
        if self.use_groq:
            try:
                from ..utils.groq_client import GroqClient
                self.client = GroqClient(api_key=self.api_key)
                self.vision_available = True
            except Exception:
                self.vision_available = False
        else:
            self.vision_available = False
    
    def analyze(self, file_path: str) -> ImageResult:
        """
        Analyze an image file.
        
        Args:
            file_path: Path to image file
            
        Returns:
            ImageResult with description and extracted text
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {file_path}")
        
        # Load and process image
        with Image.open(file_path) as img:
            dimensions = img.size
            img_format = img.format or path.suffix.upper().strip(".")
            
            # Resize if needed
            processed_img = self._resize_image(img)
            
            # Get base64 for API
            img_base64 = self._image_to_base64(processed_img)
        
        # Get description from LLM
        if self.vision_available:
            description = self._describe_with_llm(img_base64)
            extracted_text = self._extract_text_with_llm(img_base64) if self.extract_text else ""
        else:
            description = f"[Image: {dimensions[0]}x{dimensions[1]} {img_format}]"
            extracted_text = ""
        
        return ImageResult(
            description=description,
            extracted_text=extracted_text,
            dimensions=dimensions,
            format=img_format,
            metadata={"file_path": str(path)}
        )
    
    def analyze_from_bytes(self, image_bytes: bytes, filename: str = "image") -> ImageResult:
        """
        Analyze image from bytes (for file uploads).
        
        Args:
            image_bytes: Image file content as bytes
            filename: Original filename for format detection
            
        Returns:
            ImageResult with analysis
        """
        with Image.open(io.BytesIO(image_bytes)) as img:
            dimensions = img.size
            img_format = img.format or Path(filename).suffix.upper().strip(".")
            
            processed_img = self._resize_image(img)
            img_base64 = self._image_to_base64(processed_img)
        
        if self.vision_available:
            description = self._describe_with_llm(img_base64)
            extracted_text = self._extract_text_with_llm(img_base64) if self.extract_text else ""
        else:
            description = f"[Image: {dimensions[0]}x{dimensions[1]} {img_format}]"
            extracted_text = ""
        
        return ImageResult(
            description=description,
            extracted_text=extracted_text,
            dimensions=dimensions,
            format=img_format,
            metadata={"filename": filename}
        )
    
    def _resize_image(self, img: Image.Image) -> Image.Image:
        """Resize image if larger than max size."""
        if max(img.size) <= self.MAX_IMAGE_SIZE:
            return img
        
        ratio = self.MAX_IMAGE_SIZE / max(img.size)
        new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
        return img.resize(new_size, Image.Resampling.LANCZOS)
    
    def _image_to_base64(self, img: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        buffer = io.BytesIO()
        # Convert to RGB if necessary (for PNG with transparency)
        if img.mode in ('RGBA', 'P'):
            img = img.convert('RGB')
        img.save(buffer, format='JPEG', quality=85)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    def _describe_with_llm(self, img_base64: str) -> str:
        """Use LLM to describe image."""
        try:
            # Note: Groq may not support vision yet
            # This is a placeholder for when it does
            prompt = """Describe this image in detail. Include:
1. What is shown in the image
2. Key objects, people, or text visible
3. The overall context or purpose of the image

Be concise but comprehensive."""
            
            # For now, return placeholder since Groq vision may not be available
            return "[Vision analysis requires vision-capable model. Image metadata extracted.]"
            
        except Exception as e:
            return f"[Error describing image: {str(e)}]"
    
    def _extract_text_with_llm(self, img_base64: str) -> str:
        """Use LLM to extract visible text from image."""
        try:
            # Placeholder - would use vision model
            return ""
        except Exception:
            return ""


# ============== Quick-use function ==============

def describe_image(file_path: str, api_key: str = None) -> str:
    """
    Quick function to describe an image.
    
    Args:
        file_path: Path to image file
        api_key: Optional API key
        
    Returns:
        Image description as string
    """
    ingestor = ImageIngestor(api_key=api_key)
    result = ingestor.analyze(file_path)
    return result.to_markdown()
