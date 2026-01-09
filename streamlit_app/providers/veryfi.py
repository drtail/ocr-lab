"""Veryfi OCR provider implementation."""

import time
from typing import Dict, Any
from pathlib import Path

try:
    from veryfi import Client as VeryfiClient
except ImportError:
    VeryfiClient = None

from .base import OCRProvider, OCRResult
from core.result_normalizer import ResultNormalizer


class VeryfiProvider(OCRProvider):
    """Veryfi receipt OCR provider."""

    def __init__(self):
        """Initialize Veryfi provider."""
        super().__init__()
        self.client = None
        self.name = "veryfi"

    def configure(self, config: Dict[str, Any]) -> None:
        """Apply runtime configuration.

        Args:
            config: Configuration dictionary with Veryfi credentials

        Raises:
            ValueError: If configuration is invalid
            ImportError: If veryfi SDK is not installed
        """
        if not self.validate_config(config):
            raise ValueError("Invalid Veryfi configuration")

        if VeryfiClient is None:
            raise ImportError(
                "Veryfi SDK not installed. Install with: pip install veryfi"
            )

        # Initialize Veryfi client
        self.client = VeryfiClient(
            client_id=config["client_id"],
            client_secret=config["client_secret"],
            username=config["username"],
            api_key=config["api_key"]
        )

        # Store additional config
        self.config = config
        self._configured = True

    def process_image(self, image_path: str) -> OCRResult:
        """Process receipt image with Veryfi.

        Args:
            image_path: Path to receipt image

        Returns:
            OCRResult with standardized data

        Raises:
            RuntimeError: If provider not configured or API call fails
        """
        if not self._configured or self.client is None:
            raise RuntimeError("Provider not configured. Call configure() first.")

        if not Path(image_path).exists():
            raise ValueError(f"Image file not found: {image_path}")

        start_time = time.time()
        error = None
        raw_response = {}
        normalized_data = {}
        confidence_score = None

        try:
            # Call Veryfi API
            categories = self.config.get("categories", ["Grocery"])
            boost_mode = self.config.get("boost_mode", False)

            raw_response = self.client.process_document(
                file_path=image_path,
                categories=categories,
                boost_mode=boost_mode
            )

            # Extract confidence score
            confidence_score = raw_response.get("confidence")

            # Normalize response
            receipt_data = ResultNormalizer.normalize_veryfi(raw_response)
            normalized_data = receipt_data.model_dump()

        except Exception as e:
            error = f"Veryfi API error: {str(e)}"
            # Return partial result with error
            pass

        processing_time = time.time() - start_time

        return OCRResult(
            provider=self.name,
            raw_response=raw_response,
            normalized_data=normalized_data,
            confidence_score=confidence_score,
            processing_time=processing_time,
            error=error
        )

    def get_config_schema(self) -> Dict[str, Any]:
        """Return configuration schema.

        Returns:
            Configuration schema dictionary
        """
        return {
            "client_id": {
                "type": "string",
                "required": True,
                "secret": True,
                "description": "Veryfi Client ID"
            },
            "client_secret": {
                "type": "string",
                "required": True,
                "secret": True,
                "description": "Veryfi Client Secret"
            },
            "username": {
                "type": "string",
                "required": True,
                "description": "Veryfi Username"
            },
            "api_key": {
                "type": "string",
                "required": True,
                "secret": True,
                "description": "Veryfi API Key"
            },
            "categories": {
                "type": "array",
                "required": False,
                "default": ["Grocery", "Meals & Entertainment"],
                "options": ["Grocery", "Meals & Entertainment", "Shopping", "Gas/Transport", "Other"],
                "description": "Receipt categories"
            },
            "boost_mode": {
                "type": "boolean",
                "required": False,
                "default": False,
                "description": "Enable boost mode for higher accuracy (slower)"
            }
        }

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate configuration.

        Args:
            config: Configuration to validate

        Returns:
            True if valid

        Raises:
            ValueError: If configuration is invalid
        """
        required_fields = ["client_id", "client_secret", "username", "api_key"]

        for field in required_fields:
            if field not in config or not config[field]:
                raise ValueError(f"Missing required field: {field}")

        return True
