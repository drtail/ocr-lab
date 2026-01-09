"""Google Gemini Vision provider implementation."""

import time
from typing import Dict, Any
from pathlib import Path

try:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
except ImportError:
    genai = None

from .base import OCRProvider, OCRResult
from core.result_normalizer import ResultNormalizer


class GeminiProvider(OCRProvider):
    """Google Gemini Vision receipt OCR provider."""

    def __init__(self):
        """Initialize Gemini provider."""
        super().__init__()
        self.model = None
        self.name = "gemini"

    def configure(self, config: Dict[str, Any]) -> None:
        """Apply runtime configuration.

        Args:
            config: Configuration dictionary with Gemini API key

        Raises:
            ValueError: If configuration is invalid
            ImportError: If google-generativeai SDK is not installed
        """
        if not self.validate_config(config):
            raise ValueError("Invalid Gemini configuration")

        if genai is None:
            raise ImportError(
                "Google Generative AI SDK not installed. Install with: pip install google-generativeai"
            )

        # Configure Gemini with API key
        genai.configure(api_key=config["api_key"])

        # Get model name
        model_name = config.get("model", "gemini-2.0-flash-exp")

        # Initialize model with safety settings
        generation_config = {
            "temperature": config.get("temperature", 0.1),
            "max_output_tokens": config.get("max_tokens", 1000),
        }

        # Disable safety filters for receipt text
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

        self.model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=generation_config,
            safety_settings=safety_settings
        )

        # Store additional config
        self.config = config
        self._configured = True

    def process_image(self, image_path: str) -> OCRResult:
        """Process receipt image with Gemini Vision.

        Args:
            image_path: Path to receipt image

        Returns:
            OCRResult with standardized data

        Raises:
            RuntimeError: If provider not configured or API call fails
        """
        if not self._configured or self.model is None:
            raise RuntimeError("Provider not configured. Call configure() first.")

        if not Path(image_path).exists():
            raise ValueError(f"Image file not found: {image_path}")

        start_time = time.time()
        error = None
        raw_response = {}
        normalized_data = {}
        confidence_score = None

        try:
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

            # Upload the image
            with open(image_path, "rb") as f:
                image_data = f.read()

            # Use PIL to load image
            from PIL import Image
            import io
            image = Image.open(io.BytesIO(image_data))

            # Generate content with image
            response = self.model.generate_content([prompt, image])

            # Extract response
            content = response.text
            raw_response = {
                "model": self.config.get("model", "gemini-2.0-flash-exp"),
                "content": content,
                "usage": {
                    "prompt_tokens": response.usage_metadata.prompt_token_count if hasattr(response, 'usage_metadata') else None,
                    "candidates_tokens": response.usage_metadata.candidates_token_count if hasattr(response, 'usage_metadata') else None,
                    "total_tokens": response.usage_metadata.total_token_count if hasattr(response, 'usage_metadata') else None
                },
                "finish_reason": response.candidates[0].finish_reason if response.candidates else None
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
            receipt_data = ResultNormalizer.normalize_gemini(parsed_data)
            normalized_data = receipt_data.model_dump()

            # Estimate confidence based on finish_reason
            if response.candidates and response.candidates[0].finish_reason == 1:  # STOP
                confidence_score = 0.9
            else:
                confidence_score = 0.7

        except Exception as e:
            error = f"Gemini API error: {str(e)}"

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
                "description": "Google AI API Key"
            },
            "model": {
                "type": "string",
                "required": False,
                "default": "gemini-2.0-flash-exp",
                "options": ["gemini-2.0-flash-exp", "gemini-1.5-pro", "gemini-1.5-flash"],
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
