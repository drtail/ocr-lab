"""Abstract base class for OCR providers."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field


class OCRResult(BaseModel):
    """Standardized OCR output format."""

    provider: str = Field(..., description="Name of the OCR provider")
    raw_response: Dict[str, Any] = Field(..., description="Original API response")
    normalized_data: Dict[str, Any] = Field(..., description="Standardized receipt data")
    confidence_score: Optional[float] = Field(None, description="Overall confidence score (0-1)")
    processing_time: float = Field(..., description="Processing time in seconds")
    error: Optional[str] = Field(None, description="Error message if processing failed")
    timestamp: datetime = Field(default_factory=datetime.now, description="Processing timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "provider": "veryfi",
                "raw_response": {"id": "123", "vendor": {"name": "Test Store"}},
                "normalized_data": {"merchant_name": "Test Store", "total_amount": 25.99},
                "confidence_score": 0.95,
                "processing_time": 1.23,
                "error": None,
                "timestamp": "2026-01-08T19:40:00"
            }
        }


class OCRProvider(ABC):
    """Abstract base class for OCR providers.

    All provider implementations must inherit from this class and implement
    the required abstract methods.
    """

    def __init__(self):
        """Initialize the provider."""
        self.name = self.__class__.__name__
        self.config = {}
        self._configured = False

    @abstractmethod
    def configure(self, config: Dict[str, Any]) -> None:
        """Apply runtime configuration to the provider.

        Args:
            config: Configuration dictionary with provider-specific settings

        Raises:
            ValueError: If configuration is invalid
        """
        pass

    @abstractmethod
    def process_image(self, image_path: str) -> OCRResult:
        """Process receipt image and return standardized result.

        Args:
            image_path: Path to the image file

        Returns:
            OCRResult with standardized data

        Raises:
            ValueError: If image path is invalid
            RuntimeError: If API call fails
        """
        pass

    @abstractmethod
    def get_config_schema(self) -> Dict[str, Any]:
        """Return configuration schema for UI generation.

        The schema defines what configuration parameters this provider accepts,
        their types, default values, and whether they're required.

        Returns:
            Dictionary describing the configuration schema

        Example:
            {
                "api_key": {
                    "type": "string",
                    "required": True,
                    "secret": True,
                    "description": "API key for authentication"
                },
                "language": {
                    "type": "string",
                    "required": False,
                    "default": "en",
                    "options": ["en", "ko", "ja"],
                    "description": "Processing language"
                }
            }
        """
        pass

    @abstractmethod
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate configuration parameters.

        Args:
            config: Configuration dictionary to validate

        Returns:
            True if configuration is valid

        Raises:
            ValueError: If configuration is invalid with detailed error message
        """
        pass

    def is_configured(self) -> bool:
        """Check if provider is properly configured.

        Returns:
            True if provider is ready to process images
        """
        return self._configured

    def get_provider_info(self) -> Dict[str, Any]:
        """Get provider metadata.

        Returns:
            Dictionary with provider information
        """
        return {
            "name": self.name,
            "configured": self._configured,
            "config_schema": self.get_config_schema()
        }
