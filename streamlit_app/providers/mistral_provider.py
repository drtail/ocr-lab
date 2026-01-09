"""Mistral AI Vision provider implementation."""

import time
import base64
from typing import Dict, Any
from pathlib import Path

try:
    from mistralai import Mistral
except ImportError:
    Mistral = None

from .base import OCRProvider, OCRResult
from core.result_normalizer import ResultNormalizer


class MistralProvider(OCRProvider):
    """Mistral Pixtral Vision receipt OCR provider."""

    def __init__(self):
        """Initialize Mistral provider."""
        super().__init__()
        self.client = None
        self.name = "mistral"

    def configure(self, config: Dict[str, Any]) -> None:
        """Apply runtime configuration.

        Args:
            config: Configuration dictionary with Mistral API key

        Raises:
            ValueError: If configuration is invalid
            ImportError: If mistralai SDK is not installed
        """
        if not self.validate_config(config):
            raise ValueError("Invalid Mistral configuration")

        if Mistral is None:
            raise ImportError(
                "Mistral SDK not installed. Install with: pip install mistralai"
            )

        # Initialize Mistral client
        self.client = Mistral(api_key=config["api_key"])

        # Store additional config
        self.config = config
        self._configured = True

    def process_image(self, image_path: str) -> OCRResult:
        """Process receipt image with Mistral Vision.

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
            # Get model parameters (using OCR-specific models)
            model = self.config.get("model", "mistral-ocr-2512")

            # Call Mistral OCR API with the image file
            # Using the dedicated OCR endpoint which is optimized for document processing
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode("utf-8")
                response = self.client.ocr.process(
                    model=model,
                    document={
                        "type": "image_url",
                        "image_url": f"data:image/jpeg;base64,{base64_image}"
                    }
                )

            # Extract OCR text from response
            # The OCR API returns structured text extraction
            ocr_text = response.text if hasattr(response, 'text') else str(response)

            # Store raw response
            raw_response = {
                "model": model,
                "text": ocr_text,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens if hasattr(response, 'usage') and response.usage else None,
                    "completion_tokens": response.usage.completion_tokens if hasattr(response, 'usage') and response.usage else None,
                    "total_tokens": response.usage.total_tokens if hasattr(response, 'usage') and response.usage else None
                }
            }

            # Now use a follow-up call to structure the OCR text into receipt data
            # This uses the chat API with the OCR text to extract structured information
            import json

            structure_prompt = f"""Given this receipt OCR text, extract structured information in JSON format:

OCR Text:
{ocr_text}

Extract and return ONLY a JSON object with this structure:
{{
    "merchant_name": "store name",
    "merchant_address": "full address if visible",
    "merchant_phone": "phone number if visible",
    "date": "transaction date in YYYY-MM-DD format",
    "time": "transaction time in HH:MM format",
    "total_amount": numeric value only,
    "subtotal": numeric value only,
    "tax": numeric value only,
    "tip": numeric value only (if present),
    "currency": "currency code (e.g., USD, EUR)",
    "payment_method": "cash, card, or other method",
    "line_items": [
        {{
            "description": "item name",
            "quantity": numeric value,
            "unit_price": numeric value,
            "total_price": numeric value
        }}
    ]
}}

Use null for missing fields. Return ONLY the JSON, no additional text."""

            structure_response = self.client.chat.complete(
                model=model,
                messages=[{"role": "user", "content": structure_prompt}],
                temperature=0.1
            )

            content = structure_response.choices[0].message.content

            # Parse JSON from response
            if "```json" in content:
                json_start = content.find("```json") + 7
                json_end = content.find("```", json_start)
                content = content[json_start:json_end].strip()
            elif "```" in content:
                json_start = content.find("```") + 3
                json_end = content.find("```", json_start)
                content = content[json_start:json_end].strip()

            parsed_data = json.loads(content)

            # Add structured response to raw_response
            raw_response["structured_content"] = content

            # Normalize response using standard format
            receipt_data = ResultNormalizer.normalize_mistral(parsed_data)
            normalized_data = receipt_data.model_dump()

            # Set confidence score (OCR is generally high confidence)
            confidence_score = 0.9

        except Exception as e:
            error = f"Mistral API error: {str(e)}"

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
                "description": "Mistral API Key"
            },
            "model": {
                "type": "string",
                "required": False,
                "default": "mistral-ocr-2512",
                "options": ["mistral-ocr-2512", "mistral-ocr-2505"],
                "description": "Model to use for OCR analysis"
            },
            "max_tokens": {
                "type": "integer",
                "required": False,
                "default": 1000,
                "description": "Maximum tokens in response"
            },
            "temperature": {
                "type": "number",
                "required": False,
                "default": 0.1,
                "description": "Temperature for generation (0-1)"
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
