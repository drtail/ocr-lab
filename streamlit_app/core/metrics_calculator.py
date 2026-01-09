"""Metrics calculation for OCR results analysis."""

from typing import Dict, List, Any, Optional
from decimal import Decimal
from pydantic import BaseModel

from ..providers.base import OCRResult
from .comparison_engine import ComparisonReport


class ProviderMetrics(BaseModel):
    """Metrics for a single provider."""

    provider_name: str
    processing_time: float
    confidence_score: Optional[float]
    success: bool
    error_message: Optional[str]
    accuracy_score: Optional[float] = None  # Based on consensus
    completeness_score: Optional[float] = None  # % of fields extracted
    estimated_cost: Optional[float] = None


class AggregateMetrics(BaseModel):
    """Aggregate metrics across all providers."""

    total_providers: int
    successful_providers: int
    failed_providers: int
    success_rate: float
    avg_processing_time: float
    fastest_provider: str
    slowest_provider: str
    avg_confidence: float
    most_confident_provider: str
    least_confident_provider: str
    consensus_strength: float  # How much providers agree


class MetricsCalculator:
    """Calculate performance metrics for OCR results."""

    # Estimated costs per API call (in USD) - for reference
    PROVIDER_COSTS = {
        "veryfi": 0.50,  # Estimate
        "taggun": 0.05,  # Estimate
        "klippa": 0.10,  # Estimate
        "openai": 0.015,  # GPT-4o with vision (~1K tokens)
        "mistral": 0.006,  # Pixtral pricing estimate
        "gemini": 0.001,  # Gemini 2.0 Flash pricing
    }

    @staticmethod
    def calculate_provider_metrics(
        result: OCRResult,
        accuracy_score: Optional[float] = None
    ) -> ProviderMetrics:
        """Calculate metrics for a single provider result.

        Args:
            result: OCR result from provider
            accuracy_score: Optional accuracy score based on consensus

        Returns:
            ProviderMetrics for the provider
        """
        # Calculate completeness (% of key fields extracted)
        completeness = MetricsCalculator._calculate_completeness(result.normalized_data)

        # Get estimated cost
        estimated_cost = MetricsCalculator.PROVIDER_COSTS.get(result.provider)

        return ProviderMetrics(
            provider_name=result.provider,
            processing_time=result.processing_time,
            confidence_score=result.confidence_score,
            success=result.error is None,
            error_message=result.error,
            accuracy_score=accuracy_score,
            completeness_score=completeness,
            estimated_cost=estimated_cost
        )

    @staticmethod
    def _calculate_completeness(normalized_data: Dict[str, Any]) -> float:
        """Calculate completeness score (% of fields populated).

        Args:
            normalized_data: Normalized receipt data

        Returns:
            Completeness score (0-100)
        """
        key_fields = [
            "merchant_name",
            "transaction_date",
            "total_amount",
            "tax_amount",
            "line_items"
        ]

        populated_count = 0
        for field in key_fields:
            value = normalized_data.get(field)
            if value is not None and value != "" and value != [] and value != 0:
                populated_count += 1

        return (populated_count / len(key_fields)) * 100

    @staticmethod
    def calculate_aggregate_metrics(
        comparison_report: ComparisonReport,
        provider_metrics: List[ProviderMetrics]
    ) -> AggregateMetrics:
        """Calculate aggregate metrics across all providers.

        Args:
            comparison_report: Comparison report
            provider_metrics: List of provider metrics

        Returns:
            AggregateMetrics
        """
        total_providers = len(provider_metrics)
        successful_providers = sum(1 for m in provider_metrics if m.success)
        failed_providers = total_providers - successful_providers

        success_rate = (successful_providers / total_providers * 100) if total_providers > 0 else 0

        # Processing time metrics
        processing_times = [m.processing_time for m in provider_metrics if m.success]
        avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0

        fastest_provider = min(
            provider_metrics,
            key=lambda m: m.processing_time
        ).provider_name if provider_metrics else "N/A"

        slowest_provider = max(
            provider_metrics,
            key=lambda m: m.processing_time
        ).provider_name if provider_metrics else "N/A"

        # Confidence metrics
        confidence_scores = [m.confidence_score for m in provider_metrics if m.confidence_score is not None]
        avg_confidence = (sum(confidence_scores) / len(confidence_scores)) if confidence_scores else 0

        most_confident = max(
            [m for m in provider_metrics if m.confidence_score is not None],
            key=lambda m: m.confidence_score,
            default=None
        )
        most_confident_provider = most_confident.provider_name if most_confident else "N/A"

        least_confident = min(
            [m for m in provider_metrics if m.confidence_score is not None],
            key=lambda m: m.confidence_score,
            default=None
        )
        least_confident_provider = least_confident.provider_name if least_confident else "N/A"

        # Consensus strength (average agreement across fields)
        consensus_strength = MetricsCalculator._calculate_consensus_strength(comparison_report)

        return AggregateMetrics(
            total_providers=total_providers,
            successful_providers=successful_providers,
            failed_providers=failed_providers,
            success_rate=success_rate,
            avg_processing_time=avg_processing_time,
            fastest_provider=fastest_provider,
            slowest_provider=slowest_provider,
            avg_confidence=avg_confidence * 100 if avg_confidence else 0,  # Convert to percentage
            most_confident_provider=most_confident_provider,
            least_confident_provider=least_confident_provider,
            consensus_strength=consensus_strength
        )

    @staticmethod
    def _calculate_consensus_strength(comparison_report: ComparisonReport) -> float:
        """Calculate overall consensus strength.

        Args:
            comparison_report: Comparison report

        Returns:
            Consensus strength (0-100)
        """
        if not comparison_report.field_comparisons:
            return 0.0

        total_agreement = sum(
            comp.agreement_percentage
            for comp in comparison_report.field_comparisons
        )

        return total_agreement / len(comparison_report.field_comparisons)

    @staticmethod
    def estimate_total_cost(provider_metrics: List[ProviderMetrics]) -> float:
        """Estimate total cost for all API calls.

        Args:
            provider_metrics: List of provider metrics

        Returns:
            Total estimated cost in USD
        """
        total_cost = 0.0

        for metrics in provider_metrics:
            if metrics.success and metrics.estimated_cost:
                total_cost += metrics.estimated_cost

        return total_cost

    @staticmethod
    def rank_providers(
        provider_metrics: List[ProviderMetrics],
        criteria: str = "overall"
    ) -> List[ProviderMetrics]:
        """Rank providers by specified criteria.

        Args:
            provider_metrics: List of provider metrics
            criteria: Ranking criteria (overall, speed, accuracy, confidence, cost, completeness)

        Returns:
            Sorted list of provider metrics
        """
        successful_providers = [m for m in provider_metrics if m.success]

        if criteria == "speed":
            return sorted(successful_providers, key=lambda m: m.processing_time)
        elif criteria == "accuracy" and any(m.accuracy_score for m in successful_providers):
            return sorted(
                [m for m in successful_providers if m.accuracy_score is not None],
                key=lambda m: m.accuracy_score,
                reverse=True
            )
        elif criteria == "confidence" and any(m.confidence_score for m in successful_providers):
            return sorted(
                [m for m in successful_providers if m.confidence_score is not None],
                key=lambda m: m.confidence_score,
                reverse=True
            )
        elif criteria == "cost":
            return sorted(
                [m for m in successful_providers if m.estimated_cost is not None],
                key=lambda m: m.estimated_cost
            )
        elif criteria == "completeness":
            return sorted(
                [m for m in successful_providers if m.completeness_score is not None],
                key=lambda m: m.completeness_score,
                reverse=True
            )
        else:  # overall
            # Composite score: accuracy (40%) + completeness (30%) + speed (20%) + confidence (10%)
            def composite_score(m):
                score = 0.0

                if m.accuracy_score:
                    score += m.accuracy_score * 0.4

                if m.completeness_score:
                    score += m.completeness_score * 0.3

                # Normalize speed (faster is better)
                if m.processing_time > 0:
                    max_time = max(p.processing_time for p in successful_providers)
                    normalized_speed = (1 - m.processing_time / max_time) * 100
                    score += normalized_speed * 0.2

                if m.confidence_score:
                    score += m.confidence_score * 100 * 0.1

                return score

            return sorted(successful_providers, key=composite_score, reverse=True)
