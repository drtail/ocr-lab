"""Receipt text parser for extracting structured data from OCR text."""

import re
from typing import Dict, Any, List, Optional
from datetime import datetime
from decimal import Decimal


class ReceiptParser:
    """Parse raw OCR text into structured receipt data."""

    # Common patterns for receipt parsing
    DATE_PATTERNS = [
        r'\b(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b',
        r'\b(\d{4}[-/]\d{1,2}[-/]\d{1,2})\b',
        r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{2,4}\b'
    ]

    TIME_PATTERNS = [
        r'\b(\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AP]M)?)\b'
    ]

    MONEY_PATTERNS = [
        r'\$?\s*(\d+[,\d]*\.?\d{0,2})',
        r'(\d+[,\d]*\.?\d{0,2})\s*(?:USD|KRW|EUR|GBP)'
    ]

    TOTAL_KEYWORDS = [
        'total', 'amount due', 'balance', 'grand total', 'amount',
        '합계', '총액', 'sum', 'subtotal'
    ]

    TAX_KEYWORDS = ['tax', 'vat', 'gst', 'sales tax', '세금']

    def __init__(self):
        """Initialize receipt parser."""
        pass

    def parse_text(self, text: str) -> Dict[str, Any]:
        """Parse OCR text into structured receipt data.

        Args:
            text: Raw OCR text

        Returns:
            Dictionary with parsed receipt data
        """
        if not text:
            return {
                "merchant_name": None,
                "merchant_address": None,
                "date": None,
                "time": None,
                "total": 0,
                "tax": None,
                "currency": "USD",
                "items": []
            }

        lines = text.split("\n")

        result = {
            "merchant_name": self._extract_merchant_name(lines),
            "merchant_address": self._extract_address(lines),
            "date": self._extract_date(text),
            "time": self._extract_time(text),
            "total": self._extract_total(text),
            "tax": self._extract_tax(text),
            "currency": "USD",  # Default, can be enhanced
            "items": self._extract_line_items(lines)
        }

        return result

    def _extract_merchant_name(self, lines: List[str]) -> Optional[str]:
        """Extract merchant name from receipt (usually first few lines)."""
        # Merchant name is typically in the first 1-3 lines
        # Look for lines with all caps or title case
        for line in lines[:5]:
            line = line.strip()
            if len(line) > 3 and not any(keyword in line.lower() for keyword in ['tel', 'phone', 'address', 'street']):
                # Skip lines that look like phone numbers or addresses
                if not re.search(r'\d{3}[-\s]?\d{3}[-\s]?\d{4}', line):
                    if line.isupper() or line.istitle():
                        return line
                    # If first substantial line, use it
                    if len(line) > 5:
                        return line
        return None

    def _extract_address(self, lines: List[str]) -> Optional[str]:
        """Extract address from receipt."""
        # Look for lines with street/address keywords
        address_keywords = ['street', 'st', 'ave', 'avenue', 'rd', 'road', 'blvd', 'boulevard']

        for line in lines[:10]:
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in address_keywords):
                return line.strip()

        return None

    def _extract_date(self, text: str) -> Optional[str]:
        """Extract date from receipt text."""
        for pattern in self.DATE_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        return None

    def _extract_time(self, text: str) -> Optional[str]:
        """Extract time from receipt text."""
        for pattern in self.TIME_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        return None

    def _extract_total(self, text: str) -> Decimal:
        """Extract total amount from receipt."""
        # Look for total keywords followed by amount
        for keyword in self.TOTAL_KEYWORDS:
            # Search for keyword followed by amount on same or next line
            pattern = rf'{keyword}\s*:?\s*\$?\s*(\d+[,\d]*\.?\d{{0,2}})'
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    amount_str = match.group(1).replace(',', '')
                    return Decimal(amount_str)
                except:
                    pass

        # Fallback: find largest amount in text (likely the total)
        all_amounts = self._extract_all_amounts(text)
        if all_amounts:
            return max(all_amounts)

        return Decimal("0")

    def _extract_tax(self, text: str) -> Optional[Decimal]:
        """Extract tax amount from receipt."""
        for keyword in self.TAX_KEYWORDS:
            pattern = rf'{keyword}\s*:?\s*\$?\s*(\d+[,\d]*\.?\d{{0,2}})'
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    amount_str = match.group(1).replace(',', '')
                    return Decimal(amount_str)
                except:
                    pass
        return None

    def _extract_all_amounts(self, text: str) -> List[Decimal]:
        """Extract all monetary amounts from text."""
        amounts = []
        for pattern in self.MONEY_PATTERNS:
            for match in re.finditer(pattern, text):
                try:
                    amount_str = match.group(1).replace(',', '')
                    amount = Decimal(amount_str)
                    if amount > 0:
                        amounts.append(amount)
                except:
                    pass
        return amounts

    def _extract_line_items(self, lines: List[str]) -> List[Dict[str, Any]]:
        """Extract line items from receipt.

        This is a simplified implementation. More sophisticated parsing
        would use ML or more complex heuristics.
        """
        items = []

        # Look for lines that have both text and a price
        for line in lines:
            line = line.strip()
            if not line or len(line) < 3:
                continue

            # Look for pattern: description ... price
            # Price typically at end of line
            price_match = re.search(r'\$?\s*(\d+\.?\d{0,2})\s*$', line)
            if price_match:
                price_str = price_match.group(1)
                description = line[:price_match.start()].strip()

                # Filter out likely non-items (totals, subtotals, etc.)
                if description and not any(
                    keyword in description.lower()
                    for keyword in ['total', 'subtotal', 'tax', 'change', 'cash', 'card', 'credit']
                ):
                    try:
                        price = Decimal(price_str)
                        if price > 0 and price < 1000:  # Reasonable item price
                            items.append({
                                "description": description,
                                "price": float(price)
                            })
                    except:
                        pass

        return items
