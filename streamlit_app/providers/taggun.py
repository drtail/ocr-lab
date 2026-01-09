"""Taggun OCR provider implementation."""

import time
import requests
from typing import Dict, Any
from pathlib import Path

from .base import OCRProvider, OCRResult
from core.result_normalizer import ResultNormalizer


class TaggunProvider(OCRProvider):
    """Taggun receipt OCR provider."""

    API_URL = "https://api.taggun.io/api/receipt/v1/verbose/file"

    def __init__(self):
        """Initialize Taggun provider."""
        super().__init__()
        self.api_key = None
        self.name = "taggun"

    def configure(self, config: Dict[str, Any]) -> None:
        """Apply runtime configuration.

        Args:
            config: Configuration dictionary with Taggun credentials

        Raises:
            ValueError: If configuration is invalid
        """
        if not self.validate_config(config):
            raise ValueError("Invalid Taggun configuration")

        self.api_key = config["api_key"]
        self.config = config
        self._configured = True

    def process_image(self, image_path: str) -> OCRResult:
        """Process receipt image with Taggun.

        Args:
            image_path: Path to receipt image

        Returns:
            OCRResult with standardized data

        Raises:
            RuntimeError: If provider not configured or API call fails
        """
        if not self._configured or not self.api_key:
            raise RuntimeError("Provider not configured. Call configure() first.")

        if not Path(image_path).exists():
            raise ValueError(f"Image file not found: {image_path}")

        start_time = time.time()
        error = None
        raw_response = {}
        normalized_data = {}
        confidence_score = None

        try:
            # Prepare request
            headers = {
                "apikey": self.api_key,
                "Accept": "application/json"
            }

            with open(image_path, "rb") as image_file:
                files = {"file": image_file}

                # Optional parameters
                data = {}
                if "extract_line_items" in self.config:
                    data["extractLineItems"] = str(self.config["extract_line_items"]).lower()
                if "extract_payment_method" in self.config:
                    data["extractPaymentMethod"] = str(self.config["extract_payment_method"]).lower()
                if "language" in self.config:
                    data["language"] = self.config["language"]

                # Call Taggun API
                response = requests.post(
                    self.API_URL,
                    headers=headers,
                    files=files,
                    data=data,
                    timeout=30
                )

                response.raise_for_status()
                raw_response = response.json()

            # Calculate average confidence
            confidence_values = []
            merchant_name = raw_response.get("merchantName", {})
            if isinstance(merchant_name, dict) and "confidenceLevel" in merchant_name:
                confidence_values.append(merchant_name["confidenceLevel"])

            if confidence_values:
                confidence_score = sum(confidence_values) / len(confidence_values)

            # Normalize response
            receipt_data = ResultNormalizer.normalize_taggun(raw_response)
            normalized_data = receipt_data.model_dump()

        except requests.exceptions.RequestException as e:
            error = f"Taggun API error: {str(e)}"
        except Exception as e:
            error = f"Taggun processing error: {str(e)}"

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
            "api_key": {
                "type": "string",
                "required": True,
                "secret": True,
                "description": "Taggun API Key"
            },
            "extract_line_items": {
                "type": "boolean",
                "required": False,
                "default": True,
                "description": "Extract individual line items"
            },
            "extract_payment_method": {
                "type": "boolean",
                "required": False,
                "default": True,
                "description": "Extract payment method"
            },
            "language": {
                "type": "string",
                "required": False,
                "default": "en",
                "options": ["en", "ko", "ja", "zh", "es", "fr", "de"],
                "description": "Receipt language"
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
        if "api_key" not in config or not config["api_key"]:
            raise ValueError("Missing required field: api_key")

        return True
