"""Multi-provider comparison engine for OCR results."""

import concurrent.futures
from typing import List, Dict, Any
from decimal import Decimal

from providers.base import OCRProvider, OCRResult
from .result_normalizer import ReceiptData
from pydantic import BaseModel


class FieldComparison(BaseModel):
    """Comparison of a specific field across providers."""

    field_name: str
    values: Dict[str, Any]  # provider_name -> value
    consensus_value: Any = None
    agreement_count: int = 0
    total_providers: int = 0
    agreement_percentage: float = 0.0


class ComparisonReport(BaseModel):
    """Comprehensive comparison report across providers."""

    provider_results: List[OCRResult]
    field_comparisons: List[FieldComparison]
    consensus_data: Dict[str, Any]
    processing_times: Dict[str, float]
    confidence_scores: Dict[str, float]
    success_rate: float
    fastest_provider: str
    most_confident_provider: str


class ComparisonEngine:
    """Orchestrates multi-provider OCR processing and comparison."""

    def __init__(self, providers: List[OCRProvider]):
        """Initialize comparison engine.

        Args:
            providers: List of configured OCR providers
        """
        self.providers = providers

    def process_parallel(self, image_path: str, timeout: int = 60) -> List[OCRResult]:
        """Process image with all providers in parallel.

        Args:
            image_path: Path to receipt image
            timeout: Timeout in seconds for each provider

        Returns:
            List of OCR results from all providers
        """
        results = []

        # Use ThreadPoolExecutor for concurrent API calls
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.providers)) as executor:
            # Submit all provider tasks
            future_to_provider = {
                executor.submit(self._safe_process, provider, image_path): provider
                for provider in self.providers
            }

            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_provider, timeout=timeout):
                provider = future_to_provider[future]
                try:
                    result = future.result(timeout=timeout)
                    results.append(result)
                except concurrent.futures.TimeoutError:
                    # Provider timed out
                    results.append(OCRResult(
                        provider=provider.name,
                        raw_response={},
                        normalized_data={},
                        confidence_score=None,
                        processing_time=timeout,
                        error=f"Provider timed out after {timeout}s"
                    ))
                except Exception as e:
                    # Provider failed
                    results.append(OCRResult(
                        provider=provider.name,
                        raw_response={},
                        normalized_data={},
                        confidence_score=None,
                        processing_time=0,
                        error=f"Provider failed: {str(e)}"
                    ))

        return results

    def _safe_process(self, provider: OCRProvider, image_path: str) -> OCRResult:
        """Safely process image with error handling.

        Args:
            provider: OCR provider
            image_path: Path to image

        Returns:
            OCRResult (with error if processing fails)
        """
        try:
            return provider.process_image(image_path)
        except Exception as e:
            return OCRResult(
                provider=provider.name,
                raw_response={},
                normalized_data={},
                confidence_score=None,
                processing_time=0,
                error=f"Processing error: {str(e)}"
            )

    def compare_results(self, results: List[OCRResult]) -> ComparisonReport:
        """Generate comparison report across providers.

        Args:
            results: List of OCR results from multiple providers

        Returns:
            ComparisonReport with detailed comparison
        """
        # Filter successful results
        successful_results = [r for r in results if r.error is None]

        # Extract normalized data
        normalized_results = [
            (r.provider, r.normalized_data)
            for r in successful_results
        ]

        # Compare key fields
        field_comparisons = self._compare_fields(normalized_results)

        # Build consensus data
        consensus_data = self._build_consensus(field_comparisons)

        # Calculate metrics
        processing_times = {r.provider: r.processing_time for r in results}
        confidence_scores = {
            r.provider: r.confidence_score
            for r in results
            if r.confidence_score is not None
        }

        # Success rate
        success_rate = len(successful_results) / len(results) if results else 0.0

        # Find fastest and most confident
        fastest_provider = min(
            processing_times.items(),
            key=lambda x: x[1]
        )[0] if processing_times else "N/A"

        most_confident_provider = max(
            confidence_scores.items(),
            key=lambda x: x[1]
        )[0] if confidence_scores else "N/A"

        return ComparisonReport(
            provider_results=results,
            field_comparisons=field_comparisons,
            consensus_data=consensus_data,
            processing_times=processing_times,
            confidence_scores=confidence_scores,
            success_rate=success_rate,
            fastest_provider=fastest_provider,
            most_confident_provider=most_confident_provider
        )

    def _compare_fields(self, normalized_results: List[tuple]) -> List[FieldComparison]:
        """Compare specific fields across providers.

        Args:
            normalized_results: List of (provider_name, normalized_data) tuples

        Returns:
            List of field comparisons
        """
        # Fields to compare
        key_fields = [
            "merchant_name",
            "transaction_date",
            "total_amount",
            "tax_amount",
            "currency"
        ]

        comparisons = []

        for field in key_fields:
            values = {}
            for provider, data in normalized_results:
                value = data.get(field)
                if value is not None:
                    # Convert Decimal to float for comparison
                    if isinstance(value, Decimal):
                        value = float(value)
                    values[provider] = value

            # Find consensus (most common value)
            consensus_value, agreement_count = self._find_consensus(list(values.values()))

            comparisons.append(FieldComparison(
                field_name=field,
                values=values,
                consensus_value=consensus_value,
                agreement_count=agreement_count,
                total_providers=len(values),
                agreement_percentage=(agreement_count / len(values) * 100) if values else 0.0
            ))

        return comparisons

    def _find_consensus(self, values: List[Any]) -> tuple:
        """Find most common value (consensus).

        Args:
            values: List of values to compare

        Returns:
            Tuple of (consensus_value, count)
        """
        if not values:
            return None, 0

        # Count occurrences
        value_counts = {}
        for value in values:
            # For fuzzy matching of strings
            if isinstance(value, str):
                value = value.lower().strip()

            if value in value_counts:
                value_counts[value] += 1
            else:
                value_counts[value] = 1

        # Find most common
        if value_counts:
            consensus_value = max(value_counts.items(), key=lambda x: x[1])
            return consensus_value[0], consensus_value[1]

        return None, 0

    def _build_consensus(self, field_comparisons: List[FieldComparison]) -> Dict[str, Any]:
        """Build consensus receipt data from field comparisons.

        Args:
            field_comparisons: List of field comparisons

        Returns:
            Dictionary with consensus values
        """
        consensus = {}

        for comparison in field_comparisons:
            if comparison.consensus_value is not None:
                consensus[comparison.field_name] = comparison.consensus_value

        return consensus

    def get_outliers(self, comparison: FieldComparison) -> List[str]:
        """Identify providers with outlier values for a field.

        Args:
            comparison: Field comparison

        Returns:
            List of provider names with outlier values
        """
        outliers = []

        if comparison.consensus_value is None:
            return outliers

        for provider, value in comparison.values.items():
            # Normalize for comparison
            compare_value = value
            compare_consensus = comparison.consensus_value

            if isinstance(value, str):
                compare_value = value.lower().strip()
                compare_consensus = str(comparison.consensus_value).lower().strip()

            if compare_value != compare_consensus:
                outliers.append(provider)

        return outliers

    def calculate_provider_accuracy(self, comparison_report: ComparisonReport) -> Dict[str, float]:
        """Calculate accuracy score for each provider based on consensus.

        Args:
            comparison_report: Comparison report

        Returns:
            Dictionary of provider -> accuracy score (0-100)
        """
        provider_scores = {r.provider: 0.0 for r in comparison_report.provider_results}

        total_fields = len(comparison_report.field_comparisons)
        if total_fields == 0:
            return provider_scores

        for comparison in comparison_report.field_comparisons:
            if comparison.consensus_value is None:
                continue

            # Award points for matching consensus
            for provider, value in comparison.values.items():
                # Normalize for comparison
                compare_value = value
                compare_consensus = comparison.consensus_value

                if isinstance(value, str):
                    compare_value = value.lower().strip()
                    compare_consensus = str(comparison.consensus_value).lower().strip()

                if compare_value == compare_consensus:
                    provider_scores[provider] += 1

        # Convert to percentage
        for provider in provider_scores:
            provider_scores[provider] = (provider_scores[provider] / total_fields) * 100

        return provider_scores
