"""Klippa OCR provider implementation."""

import time
import requests
from typing import Dict, Any
from pathlib import Path

from .base import OCRProvider, OCRResult
from core.result_normalizer import ResultNormalizer


class KlippaProvider(OCRProvider):
    """Klippa receipt OCR provider."""

    API_URL = "https://custom-ocr.klippa.com/api/v1/parseDocument"

    def __init__(self):
        """Initialize Klippa provider."""
        super().__init__()
        self.api_key = None
        self.name = "klippa"

    def configure(self, config: Dict[str, Any]) -> None:
        """Apply runtime configuration.

        Args:
            config: Configuration dictionary with Klippa credentials

        Raises:
            ValueError: If configuration is invalid
        """
        if not self.validate_config(config):
            raise ValueError("Invalid Klippa configuration")

        self.api_key = config["api_key"]
        self.config = config
        self._configured = True

    def process_image(self, image_path: str) -> OCRResult:
        """Process receipt image with Klippa.

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
                "X-Auth-Key": self.api_key
            }

            with open(image_path, "rb") as image_file:
                files = {"document": image_file}

                # Optional parameters
                data = {
                    "template": self.config.get("template", "receipts")
                }

                # Call Klippa API
                response = requests.post(
                    self.API_URL,
                    headers=headers,
                    files=files,
                    data=data,
                    timeout=30
                )

                response.raise_for_status()
                raw_response = response.json()

            # Extract confidence if available
            if "data" in raw_response and "confidence" in raw_response["data"]:
                confidence_score = raw_response["data"]["confidence"]

            # Normalize response
            receipt_data = ResultNormalizer.normalize_klippa(raw_response)
            normalized_data = receipt_data.model_dump()

        except requests.exceptions.RequestException as e:
            error = f"Klippa API error: {str(e)}"
        except Exception as e:
            error = f"Klippa processing error: {str(e)}"

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
                "description": "Klippa API Key"
            },
            "template": {
                "type": "string",
                "required": False,
                "default": "receipts",
                "options": ["receipts", "invoices", "general"],
                "description": "Document template"
            },
            "extract_merchant_details": {
                "type": "boolean",
                "required": False,
                "default": True,
                "description": "Extract merchant details"
            },
            "extract_line_items": {
                "type": "boolean",
                "required": False,
                "default": True,
                "description": "Extract line items"
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
