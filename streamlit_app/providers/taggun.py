"""Taggun OCR provider implementation."""

import time
import mimetypes
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
            # Prepare request - DO NOT set content-type for multipart/form-data
            # requests library handles this automatically with proper boundary
            headers = {
                "apikey": self.api_key,
                "accept": "application/json"
            }

            # Prepare form data
            data = {}

            # Extract line items (default: true)
            extract_line_items = self.config.get("extract_line_items", True)
            data["extractLineItems"] = "true" if extract_line_items else "false"

            # Extract payment method (default: true)
            extract_payment = self.config.get("extract_payment_method", True)
            data["extractPaymentMethod"] = "true" if extract_payment else "false"

            # Extract time (default: false)
            extract_time = self.config.get("extract_time", False)
            data["extractTime"] = "true" if extract_time else "false"

            # Incognito mode (default: false)
            incognito = self.config.get("incognito", False)
            data["incognito"] = "true" if incognito else "false"

            # Refresh (default: false)
            refresh = self.config.get("refresh", False)
            data["refresh"] = "true" if refresh else "false"

            # Language (optional)
            if "language" in self.config:
                data["language"] = self.config["language"]

            # Prepare file upload
            with open(image_path, "rb") as image_file:
                # Detect MIME type for proper file upload
                mime_type, _ = mimetypes.guess_type(image_path)
                if not mime_type or not mime_type.startswith("image/"):
                    mime_type = "image/png"  # Default fallback

                # Properly format file upload with content type
                files = {
                    "file": (Path(image_path).name, image_file, mime_type)
                }

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

        except requests.exceptions.HTTPError as e:
            # Include response body in error for debugging 400 errors
            try:
                error_detail = e.response.json() if e.response else {}
                error = f"Taggun API error: {e.response.status_code} - {error_detail}"
            except:
                error = f"Taggun API error: {str(e)}"
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
            "extract_time": {
                "type": "boolean",
                "required": False,
                "default": False,
                "description": "Extract time information"
            },
            "incognito": {
                "type": "boolean",
                "required": False,
                "default": False,
                "description": "Incognito mode (doesn't store receipt data)"
            },
            "refresh": {
                "type": "boolean",
                "required": False,
                "default": False,
                "description": "Force refresh cached results"
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
