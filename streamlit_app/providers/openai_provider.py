"""OpenAI (ChatGPT) Vision provider implementation."""

import time
import base64
from typing import Dict, Any
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from .base import OCRProvider, OCRResult
from core.result_normalizer import ResultNormalizer


class OpenAIProvider(OCRProvider):
    """OpenAI GPT-4 Vision receipt OCR provider."""

    def __init__(self):
        """Initialize OpenAI provider."""
        super().__init__()
        self.client = None
        self.name = "openai"

    def configure(self, config: Dict[str, Any]) -> None:
        """Apply runtime configuration.

        Args:
            config: Configuration dictionary with OpenAI API key

        Raises:
            ValueError: If configuration is invalid
            ImportError: If openai SDK is not installed
        """
        if not self.validate_config(config):
            raise ValueError("Invalid OpenAI configuration")

        if OpenAI is None:
            raise ImportError(
                "OpenAI SDK not installed. Install with: pip install openai"
            )

        # Initialize OpenAI client
        self.client = OpenAI(api_key=config["api_key"])

        # Store additional config
        self.config = config
        self._configured = True

    def process_image(self, image_path: str) -> OCRResult:
        """Process receipt image with OpenAI Vision.

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
            # Encode image to base64
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode("utf-8")
                image_ext = Path(image_path).suffix.lower().replace(".", "")
                mime_type = f"image/{image_ext}" if image_ext in ["jpg", "jpeg", "png", "gif", "webp"] else "image/jpeg"

            # Prepare the prompt for receipt extraction
            prompt = """Analyze this receipt image and extract the following information in JSON format:
{
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
        {
            "description": "item name",
            "quantity": numeric value,
            "unit_price": numeric value,
            "total_price": numeric value
        }
    ]
}

Extract all visible information. Use null for missing fields. Ensure amounts are numeric values without currency symbols."""

            # Get model parameters
            model = self.config.get("model", "gpt-4o")
            max_tokens = self.config.get("max_tokens", 1000)
            temperature = self.config.get("temperature", 0.1)

            # Call OpenAI Vision API
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{image_data}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )

            # Extract response
            content = response.choices[0].message.content
            raw_response = {
                "model": response.model,
                "content": content,
                "usage": response.usage.model_dump() if response.usage else None,
                "finish_reason": response.choices[0].finish_reason
            }

            # Parse JSON from response
            import json
            # Try to extract JSON from markdown code blocks
            if "```json" in content:
                json_start = content.find("```json") + 7
                json_end = content.find("```", json_start)
                content = content[json_start:json_end].strip()
            elif "```" in content:
                json_start = content.find("```") + 3
                json_end = content.find("```", json_start)
                content = content[json_start:json_end].strip()

            parsed_data = json.loads(content)

            # Normalize response using standard format
            receipt_data = ResultNormalizer.normalize_openai(parsed_data)
            normalized_data = receipt_data.model_dump()

            # Estimate confidence based on finish_reason
            if response.choices[0].finish_reason == "stop":
                confidence_score = 0.9
            else:
                confidence_score = 0.7

        except Exception as e:
            error = f"OpenAI API error: {str(e)}"

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
                "description": "OpenAI API Key"
            },
            "model": {
                "type": "string",
                "required": False,
                "default": "gpt-4o",
                "options": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
                "description": "Model to use for vision analysis"
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
                "description": "Temperature for generation (0-2)"
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
