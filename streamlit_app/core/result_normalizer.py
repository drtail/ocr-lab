"""Result normalization and standard data models for receipt data."""

from typing import List, Optional, Dict, Any
from decimal import Decimal
from datetime import datetime, time
from pydantic import BaseModel, Field, field_validator


class LineItem(BaseModel):
    """Standardized line item from receipt."""

    description: str = Field(..., description="Item description/name")
    quantity: Optional[Decimal] = Field(None, description="Item quantity")
    unit_price: Optional[Decimal] = Field(None, description="Price per unit")
    total_price: Decimal = Field(..., description="Total price for this item")
    category: Optional[str] = Field(None, description="Item category")
    sku: Optional[str] = Field(None, description="Stock keeping unit / product code")

    @field_validator("quantity", "unit_price", "total_price", mode="before")
    @classmethod
    def convert_to_decimal(cls, v):
        """Convert numeric values to Decimal."""
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return Decimal(str(v))
        if isinstance(v, str):
            try:
                return Decimal(v)
            except:
                return None
        return v


class ReceiptData(BaseModel):
    """Standardized receipt data structure."""

    # Merchant information
    merchant_name: Optional[str] = Field(None, description="Store/merchant name")
    merchant_address: Optional[str] = Field(None, description="Store address")
    merchant_phone: Optional[str] = Field(None, description="Store phone number")
    merchant_category: Optional[str] = Field(None, description="Business category")

    # Transaction information
    transaction_date: Optional[datetime] = Field(None, description="Transaction date")
    transaction_time: Optional[time] = Field(None, description="Transaction time")
    receipt_number: Optional[str] = Field(None, description="Receipt/transaction number")

    # Financial information
    total_amount: Decimal = Field(..., description="Total amount")
    currency: str = Field(default="USD", description="Currency code")
    subtotal: Optional[Decimal] = Field(None, description="Subtotal before tax")
    tax_amount: Optional[Decimal] = Field(None, description="Tax amount")
    tip_amount: Optional[Decimal] = Field(None, description="Tip amount")
    discount_amount: Optional[Decimal] = Field(None, description="Discount amount")

    # Line items
    line_items: List[LineItem] = Field(default_factory=list, description="Receipt line items")

    # Payment information
    payment_method: Optional[str] = Field(None, description="Payment method used")
    card_last_four: Optional[str] = Field(None, description="Last 4 digits of card")

    # Confidence scores
    confidence_scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Confidence scores for extracted fields"
    )

    @field_validator("total_amount", "subtotal", "tax_amount", "tip_amount", "discount_amount", mode="before")
    @classmethod
    def convert_to_decimal(cls, v):
        """Convert numeric values to Decimal."""
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return Decimal(str(v))
        if isinstance(v, str):
            try:
                # Remove currency symbols and spaces
                v = v.replace("$", "").replace(",", "").strip()
                return Decimal(v)
            except:
                return None
        return v

    @field_validator("transaction_date", mode="before")
    @classmethod
    def parse_date(cls, v):
        """Parse date from various formats."""
        if v is None:
            return None
        if isinstance(v, datetime):
            return v
        if isinstance(v, str):
            # Try common date formats
            for fmt in [
                "%Y-%m-%d",
                "%Y/%m/%d",
                "%m/%d/%Y",
                "%d/%m/%Y",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d %H:%M:%S"
            ]:
                try:
                    return datetime.strptime(v, fmt)
                except ValueError:
                    continue
        return None

    @field_validator("transaction_time", mode="before")
    @classmethod
    def parse_time(cls, v):
        """Parse time from various formats."""
        if v is None:
            return None
        if isinstance(v, time):
            return v
        if isinstance(v, str):
            # Try common time formats
            for fmt in ["%H:%M:%S", "%H:%M", "%I:%M %p", "%I:%M:%S %p"]:
                try:
                    return datetime.strptime(v, fmt).time()
                except ValueError:
                    continue
        return None


class ResultNormalizer:
    """Normalizes provider-specific responses to standard format."""

    @staticmethod
    def normalize_veryfi(raw_response: Dict[str, Any]) -> ReceiptData:
        """Normalize Veryfi response to standard format.

        Args:
            raw_response: Raw Veryfi API response

        Returns:
            Standardized ReceiptData
        """
        vendor = raw_response.get("vendor", {})
        line_items_raw = raw_response.get("line_items", [])

        # Extract line items
        line_items = [
            LineItem(
                description=item.get("description", ""),
                quantity=item.get("quantity"),
                unit_price=item.get("price"),
                total_price=item.get("total", item.get("price", 0)),
                category=item.get("category"),
                sku=item.get("sku")
            )
            for item in line_items_raw
            if item.get("description") or item.get("total")
        ]

        # Build confidence scores
        confidence_scores = {}
        if "confidence" in raw_response:
            confidence_scores["overall"] = raw_response["confidence"]

        return ReceiptData(
            merchant_name=vendor.get("name"),
            merchant_address=vendor.get("address"),
            merchant_phone=vendor.get("phone_number"),
            merchant_category=vendor.get("category"),
            transaction_date=raw_response.get("date"),
            receipt_number=raw_response.get("receipt_number"),
            total_amount=raw_response.get("total", 0),
            currency=raw_response.get("currency_code", "USD"),
            subtotal=raw_response.get("subtotal"),
            tax_amount=raw_response.get("tax"),
            tip_amount=raw_response.get("tip"),
            discount_amount=raw_response.get("discount"),
            line_items=line_items,
            payment_method=raw_response.get("payment", {}).get("type"),
            card_last_four=raw_response.get("payment", {}).get("card_number"),
            confidence_scores=confidence_scores
        )

    @staticmethod
    def normalize_taggun(raw_response: Dict[str, Any]) -> ReceiptData:
        """Normalize Taggun response to standard format."""
        merchant_data = raw_response.get("merchantName", {})
        amounts = raw_response.get("amounts", {})
        line_items_raw = raw_response.get("lineItems", [])

        # Extract line items
        line_items = [
            LineItem(
                description=item.get("text", ""),
                quantity=item.get("quantity"),
                unit_price=item.get("pricePerUnit"),
                total_price=item.get("price", 0),
            )
            for item in line_items_raw
            if item.get("text") or item.get("price")
        ]

        # Build confidence scores
        confidence_scores = {}
        if isinstance(merchant_data, dict) and "confidenceLevel" in merchant_data:
            confidence_scores["merchant"] = merchant_data["confidenceLevel"]

        return ReceiptData(
            merchant_name=merchant_data.get("data") if isinstance(merchant_data, dict) else str(merchant_data),
            transaction_date=raw_response.get("date", {}).get("data"),
            total_amount=amounts.get("total", {}).get("data", 0),
            tax_amount=amounts.get("tax", {}).get("data"),
            line_items=line_items,
            payment_method=raw_response.get("paymentMethod", {}).get("data"),
            confidence_scores=confidence_scores
        )

    @staticmethod
    def normalize_klippa(raw_response: Dict[str, Any]) -> ReceiptData:
        """Normalize Klippa response to standard format."""
        data = raw_response.get("data", {})
        merchant = data.get("merchant", {})
        line_items_raw = data.get("line_items", [])

        # Extract line items
        line_items = [
            LineItem(
                description=item.get("description", ""),
                quantity=item.get("quantity"),
                unit_price=item.get("unit_price"),
                total_price=item.get("amount", 0),
            )
            for item in line_items_raw
        ]

        return ReceiptData(
            merchant_name=merchant.get("name"),
            merchant_address=merchant.get("address"),
            transaction_date=data.get("date"),
            transaction_time=data.get("time"),
            receipt_number=data.get("receipt_number"),
            total_amount=data.get("amount", 0),
            currency=data.get("currency", "USD"),
            tax_amount=data.get("vat_amount"),
            line_items=line_items,
            payment_method=data.get("payment_method"),
            confidence_scores={}
        )

    @staticmethod
    def normalize_google_vision(raw_response: Dict[str, Any], parsed_data: Dict[str, Any]) -> ReceiptData:
        """Normalize Google Cloud Vision response.

        Note: Google Vision returns raw OCR text, so we need parsed_data
        from custom parsing logic.

        Args:
            raw_response: Raw Vision API response
            parsed_data: Custom parsed receipt data

        Returns:
            Standardized ReceiptData
        """
        return ReceiptData(
            merchant_name=parsed_data.get("merchant_name"),
            merchant_address=parsed_data.get("merchant_address"),
            transaction_date=parsed_data.get("date"),
            transaction_time=parsed_data.get("time"),
            total_amount=parsed_data.get("total", 0),
            currency=parsed_data.get("currency", "USD"),
            tax_amount=parsed_data.get("tax"),
            line_items=[
                LineItem(
                    description=item.get("description", ""),
                    total_price=item.get("price", 0)
                )
                for item in parsed_data.get("items", [])
            ],
            confidence_scores=parsed_data.get("confidence_scores", {})
        )

    @staticmethod
    def normalize_aws_textract(raw_response: Dict[str, Any]) -> ReceiptData:
        """Normalize AWS Textract AnalyzeExpense response."""
        expense_docs = raw_response.get("ExpenseDocuments", [])
        if not expense_docs:
            return ReceiptData(total_amount=0)

        doc = expense_docs[0]
        summary_fields = {
            field["Type"]["Text"]: field["ValueDetection"]["Text"]
            for field in doc.get("SummaryFields", [])
            if "ValueDetection" in field
        }

        # Extract line items
        line_items = []
        for item_group in doc.get("LineItemGroups", []):
            for item in item_group.get("LineItems", []):
                item_data = {
                    field["Type"]["Text"]: field["ValueDetection"]["Text"]
                    for field in item.get("LineItemExpenseFields", [])
                    if "ValueDetection" in field
                }
                if "ITEM" in item_data or "PRICE" in item_data:
                    line_items.append(
                        LineItem(
                            description=item_data.get("ITEM", ""),
                            quantity=item_data.get("QUANTITY"),
                            unit_price=item_data.get("UNIT_PRICE"),
                            total_price=item_data.get("PRICE", 0)
                        )
                    )

        return ReceiptData(
            merchant_name=summary_fields.get("VENDOR_NAME"),
            merchant_address=summary_fields.get("VENDOR_ADDRESS"),
            merchant_phone=summary_fields.get("VENDOR_PHONE"),
            transaction_date=summary_fields.get("INVOICE_RECEIPT_DATE"),
            receipt_number=summary_fields.get("INVOICE_RECEIPT_ID"),
            total_amount=summary_fields.get("TOTAL", 0),
            subtotal=summary_fields.get("SUBTOTAL"),
            tax_amount=summary_fields.get("TAX"),
            line_items=line_items,
            confidence_scores={}
        )

    @staticmethod
    def normalize_azure_vision(raw_response: Dict[str, Any], parsed_data: Dict[str, Any]) -> ReceiptData:
        """Normalize Azure Computer Vision response.

        Similar to Google Vision, Azure returns raw OCR text requiring custom parsing.
        """
        return ReceiptData(
            merchant_name=parsed_data.get("merchant_name"),
            merchant_address=parsed_data.get("merchant_address"),
            transaction_date=parsed_data.get("date"),
            transaction_time=parsed_data.get("time"),
            total_amount=parsed_data.get("total", 0),
            currency=parsed_data.get("currency", "USD"),
            tax_amount=parsed_data.get("tax"),
            line_items=[
                LineItem(
                    description=item.get("description", ""),
                    total_price=item.get("price", 0)
                )
                for item in parsed_data.get("items", [])
            ],
            confidence_scores=parsed_data.get("confidence_scores", {})
        )

    @staticmethod
    def normalize_openai(parsed_data: Dict[str, Any]) -> ReceiptData:
        """Normalize OpenAI GPT-4 Vision response.

        Args:
            parsed_data: Parsed JSON from GPT-4 Vision response

        Returns:
            Standardized ReceiptData
        """
        line_items_raw = parsed_data.get("line_items", [])

        # Extract line items
        line_items = [
            LineItem(
                description=item.get("description", ""),
                quantity=item.get("quantity"),
                unit_price=item.get("unit_price"),
                total_price=item.get("total_price", item.get("unit_price", 0)),
            )
            for item in line_items_raw
            if item.get("description") or item.get("total_price")
        ]

        return ReceiptData(
            merchant_name=parsed_data.get("merchant_name"),
            merchant_address=parsed_data.get("merchant_address"),
            merchant_phone=parsed_data.get("merchant_phone"),
            transaction_date=parsed_data.get("date"),
            transaction_time=parsed_data.get("time"),
            total_amount=parsed_data.get("total_amount", 0),
            currency=parsed_data.get("currency", "USD"),
            subtotal=parsed_data.get("subtotal"),
            tax_amount=parsed_data.get("tax"),
            tip_amount=parsed_data.get("tip"),
            line_items=line_items,
            payment_method=parsed_data.get("payment_method"),
            confidence_scores={}
        )

    @staticmethod
    def normalize_mistral(parsed_data: Dict[str, Any]) -> ReceiptData:
        """Normalize Mistral Pixtral Vision response.

        Args:
            parsed_data: Parsed JSON from Pixtral response

        Returns:
            Standardized ReceiptData
        """
        line_items_raw = parsed_data.get("line_items", [])

        # Extract line items
        line_items = [
            LineItem(
                description=item.get("description", ""),
                quantity=item.get("quantity"),
                unit_price=item.get("unit_price"),
                total_price=item.get("total_price", item.get("unit_price", 0)),
            )
            for item in line_items_raw
            if item.get("description") or item.get("total_price")
        ]

        return ReceiptData(
            merchant_name=parsed_data.get("merchant_name"),
            merchant_address=parsed_data.get("merchant_address"),
            merchant_phone=parsed_data.get("merchant_phone"),
            transaction_date=parsed_data.get("date"),
            transaction_time=parsed_data.get("time"),
            total_amount=parsed_data.get("total_amount", 0),
            currency=parsed_data.get("currency", "USD"),
            subtotal=parsed_data.get("subtotal"),
            tax_amount=parsed_data.get("tax"),
            tip_amount=parsed_data.get("tip"),
            line_items=line_items,
            payment_method=parsed_data.get("payment_method"),
            confidence_scores={}
        )

    @staticmethod
    def normalize_gemini(parsed_data: Dict[str, Any]) -> ReceiptData:
        """Normalize Google Gemini Vision response.

        Args:
            parsed_data: Parsed JSON from Gemini response

        Returns:
            Standardized ReceiptData
        """
        line_items_raw = parsed_data.get("line_items", [])

        # Extract line items
        line_items = [
            LineItem(
                description=item.get("description", ""),
                quantity=item.get("quantity"),
                unit_price=item.get("unit_price"),
                total_price=item.get("total_price", item.get("unit_price", 0)),
            )
            for item in line_items_raw
            if item.get("description") or item.get("total_price")
        ]

        return ReceiptData(
            merchant_name=parsed_data.get("merchant_name"),
            merchant_address=parsed_data.get("merchant_address"),
            merchant_phone=parsed_data.get("merchant_phone"),
            transaction_date=parsed_data.get("date"),
            transaction_time=parsed_data.get("time"),
            total_amount=parsed_data.get("total_amount", 0),
            currency=parsed_data.get("currency", "USD"),
            subtotal=parsed_data.get("subtotal"),
            tax_amount=parsed_data.get("tax"),
            tip_amount=parsed_data.get("tip"),
            line_items=line_items,
            payment_method=parsed_data.get("payment_method"),
            confidence_scores={}
        )
