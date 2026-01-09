"""OCR API Testing Lab - Main Streamlit Application."""

import streamlit as st
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from PIL import Image
import json
import pandas as pd
from decimal import Decimal

# Import core components
from core.config_manager import ConfigManager, ProviderRegistry
from core.comparison_engine import ComparisonEngine, ComparisonReport
from providers.base import OCRResult

# Import provider implementations
from providers.veryfi import VeryfiProvider
from providers.taggun import TaggunProvider
from providers.klippa import KlippaProvider
from providers.openai_provider import OpenAIProvider
from providers.mistral_provider import MistralProvider
from providers.gemini_provider import GeminiProvider


# Page configuration
st.set_page_config(
    page_title="OCR Testing Lab",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
if "config_manager" not in st.session_state:
    try:
        st.session_state.config_manager = ConfigManager()
    except FileNotFoundError as e:
        st.error(f"Configuration file not found: {e}")
        st.stop()

if "provider_registry" not in st.session_state:
    registry = ProviderRegistry()
    # Register all providers
    registry.register("veryfi", VeryfiProvider)
    registry.register("taggun", TaggunProvider)
    registry.register("klippa", KlippaProvider)
    registry.register("openai", OpenAIProvider)
    registry.register("mistral", MistralProvider)
    registry.register("gemini", GeminiProvider)
    st.session_state.provider_registry = registry

if "selected_providers" not in st.session_state:
    st.session_state.selected_providers = []

if "provider_configs" not in st.session_state:
    st.session_state.provider_configs = {}

if "config_version" not in st.session_state:
    # Counter to force form refresh when loading from environment
    st.session_state.config_version = 0

if "results" not in st.session_state:
    st.session_state.results = None

if "comparison_report" not in st.session_state:
    st.session_state.comparison_report = None


def render_sidebar():
    """Render sidebar with provider selection and configuration."""
    st.sidebar.title("🔍 OCR Testing Lab")
    st.sidebar.markdown("---")

    # Provider selection
    st.sidebar.subheader("Select Providers")

    config_manager = st.session_state.config_manager
    available_providers = config_manager.get_provider_names()

    # Group providers by type
    receipt_specific = config_manager.get_providers_by_type("receipt_specific")
    ai_vision = config_manager.get_providers_by_type("ai_vision")

    st.sidebar.markdown("**Receipt-Specific:**")
    for provider in receipt_specific:
        metadata = config_manager.get_provider_metadata(provider)
        selected = st.sidebar.checkbox(
            metadata["name"],
            key=f"select_{provider}",
            help=metadata["description"]
        )
        if selected and provider not in st.session_state.selected_providers:
            st.session_state.selected_providers.append(provider)
        elif not selected and provider in st.session_state.selected_providers:
            st.session_state.selected_providers.remove(provider)

    st.sidebar.markdown("**AI Vision Models:**")
    for provider in ai_vision:
        metadata = config_manager.get_provider_metadata(provider)
        selected = st.sidebar.checkbox(
            metadata["name"],
            key=f"select_{provider}",
            help=metadata["description"]
        )
        if selected and provider not in st.session_state.selected_providers:
            st.session_state.selected_providers.append(provider)
        elif not selected and provider in st.session_state.selected_providers:
            st.session_state.selected_providers.remove(provider)

    st.sidebar.markdown("---")

    # Quick actions
    if st.sidebar.button("🔄 Load from Environment", use_container_width=True):
        load_configs_from_env()

    if st.sidebar.button("🗑️ Clear All Configs", use_container_width=True):
        st.session_state.provider_configs = {}
        st.rerun()

    # Show selected count
    if st.session_state.selected_providers:
        st.sidebar.success(f"✅ {len(st.session_state.selected_providers)} provider(s) selected")
    else:
        st.sidebar.info("👆 Select at least one provider to begin")

    # Show debug info about .env file location
    with st.sidebar.expander("🔍 Debug Info"):
        import os
        from pathlib import Path

        # Show which .env file is being used
        env_path1 = Path(__file__).parent.parent / ".env"
        env_path2 = Path(__file__).parent / ".env"

        if env_path1.exists():
            st.write(f"📄 .env found at: `{env_path1}`")
        if env_path2.exists():
            st.write(f"📄 .env found at: `{env_path2}`")

        st.write(f"💻 Current working directory: `{os.getcwd()}`")

        # Show sample env vars (without revealing values)
        st.write("**Environment variables set:**")
        test_vars = ["OPENAI_API_KEY", "MISTRAL_API_KEY", "GEMINI_API_KEY", "VERYFI_API_KEY"]
        for var in test_vars:
            is_set = os.getenv(var) is not None
            st.write(f"- {var}: {'✅ Set' if is_set else '❌ Not set'}")


def load_configs_from_env():
    """Load provider configurations from environment variables."""
    config_manager = st.session_state.config_manager

    # Dynamically reload .env file to pick up any changes
    env_path, env_exists = config_manager.reload_env()

    if not env_exists:
        st.sidebar.error(
            f"❌ .env file not found at: {env_path}\n"
            f"Please create a .env file in the project root with your API keys."
        )
        return

    # Show which .env file was loaded
    st.sidebar.info(f"📄 Loading from: {env_path}")

    loaded_count = 0
    failed_count = 0

    for provider in st.session_state.selected_providers:
        try:
            # Get the schema to show what we're looking for
            schema = config_manager.get_config_schema(provider)
            env_vars_needed = [
                field_schema.get("env_var")
                for field_schema in schema.values()
                if field_schema.get("env_var")
            ]

            # Load the config
            env_config = config_manager.load_config_from_env(provider)
            st.session_state.provider_configs[provider] = env_config
            loaded_count += 1

            # Show what was loaded
            with st.sidebar.expander(f"✅ Loaded {provider} config"):
                st.write("**Loaded fields:**")
                for key, value in env_config.items():
                    # Don't show secret values
                    field_schema = schema.get(key, {})
                    if field_schema.get("secret"):
                        display_value = "***" if value else "Not set"
                    else:
                        display_value = str(value) if value is not None else "Not set"
                    st.write(f"- {key}: {display_value}")

        except ValueError as e:
            failed_count += 1
            st.sidebar.error(f"❌ {provider}: {str(e)}")

            # Show which env vars we were looking for
            schema = config_manager.get_config_schema(provider)
            required_env_vars = [
                field_schema.get("env_var")
                for field_schema in schema.values()
                if field_schema.get("env_var") and field_schema.get("required")
            ]
            if required_env_vars:
                with st.sidebar.expander("ℹ️ Required environment variables"):
                    import os
                    for env_var in required_env_vars:
                        is_set = os.getenv(env_var) is not None
                        status = "✅" if is_set else "❌"
                        st.write(f"{status} {env_var}")

        except Exception as e:
            failed_count += 1
            st.sidebar.error(f"❌ {provider}: Unexpected error: {str(e)}")
            import traceback
            with st.sidebar.expander("🔍 Error details"):
                st.code(traceback.format_exc())

    # Summary message
    if loaded_count > 0 and failed_count == 0:
        st.sidebar.success(f"✅ Successfully loaded {loaded_count} provider(s)")
    elif loaded_count > 0 and failed_count > 0:
        st.sidebar.warning(
            f"⚠️ Loaded {loaded_count} provider(s), {failed_count} failed. "
            f"Check your .env file."
        )
    elif failed_count > 0:
        st.sidebar.error(
            f"❌ Failed to load all {failed_count} provider(s). "
            f"Please check your .env file and ensure all required keys are set."
        )

    # Increment config version to force form widgets to refresh with new values
    if loaded_count > 0:
        st.session_state.config_version += 1
        st.rerun()


def render_provider_configuration():
    """Render configuration forms for selected providers."""
    if not st.session_state.selected_providers:
        st.info("👈 Please select at least one provider from the sidebar")
        return

    st.subheader("⚙️ Provider Configuration")

    config_manager = st.session_state.config_manager

    # Create tabs for each provider
    tabs = st.tabs([
        config_manager.get_provider_metadata(p)["name"]
        for p in st.session_state.selected_providers
    ])

    for idx, provider in enumerate(st.session_state.selected_providers):
        with tabs[idx]:
            render_provider_config_form(provider)


def render_provider_config_form(provider_name: str):
    """Render configuration form for a specific provider."""
    config_manager = st.session_state.config_manager
    schema = config_manager.get_config_schema(provider_name)
    metadata = config_manager.get_provider_metadata(provider_name)

    # Provider info
    st.markdown(f"**{metadata['description']}**")
    if metadata.get("documentation_url"):
        st.markdown(f"[📚 Documentation]({metadata['documentation_url']})")

    st.markdown("---")

    # Get existing config or initialize
    if provider_name not in st.session_state.provider_configs:
        st.session_state.provider_configs[provider_name] = {}

    config = st.session_state.provider_configs[provider_name]

    # Get config version for widget keys (forces refresh when loading from env)
    config_version = st.session_state.get("config_version", 0)

    # Render form fields based on schema
    for field_name, field_schema in schema.items():
        field_type = field_schema.get("type", "string")
        required = field_schema.get("required", False)
        description = field_schema.get("description", "")
        default = field_schema.get("default")
        is_secret = field_schema.get("secret", False)

        label = f"{field_name}{'*' if required else ''}"

        current_value = config.get(field_name, default)

        # Create unique key that changes when config is loaded from environment
        widget_key = f"{provider_name}_{field_name}_v{config_version}"

        if field_type == "boolean":
            value = st.checkbox(
                label,
                value=bool(current_value) if current_value is not None else False,
                help=description,
                key=widget_key
            )
        elif field_type == "array":
            options = field_schema.get("options", [])
            if options:
                value = st.multiselect(
                    label,
                    options=options,
                    default=current_value if current_value else default,
                    help=description,
                    key=widget_key
                )
            else:
                value = st.text_input(
                    label,
                    value=", ".join(current_value) if current_value else "",
                    help=f"{description} (comma-separated)",
                    key=widget_key
                )
                value = [v.strip() for v in value.split(",") if v.strip()]
        else:
            # String, number, integer
            options = field_schema.get("options")
            if options:
                value = st.selectbox(
                    label,
                    options=options,
                    index=options.index(current_value) if current_value in options else 0,
                    help=description,
                    key=widget_key
                )
            else:
                input_type = "password" if is_secret else "default"
                value = st.text_input(
                    label,
                    value=str(current_value) if current_value is not None else "",
                    type=input_type,
                    help=description,
                    key=widget_key
                )

                # Convert type
                if field_type == "number" and value:
                    try:
                        value = float(value)
                    except ValueError:
                        st.error(f"Invalid number for {field_name}")
                elif field_type == "integer" and value:
                    try:
                        value = int(value)
                    except ValueError:
                        st.error(f"Invalid integer for {field_name}")

        config[field_name] = value

    # Validate configuration
    is_valid, error = config_manager.validate_config(provider_name, config)

    if is_valid:
        st.success(f"✅ {metadata['name']} is configured correctly")
    else:
        st.error(f"❌ Configuration error: {error}")


def render_image_upload():
    """Render image upload and processing section."""
    st.subheader("📸 Upload Receipt")

    uploaded_file = st.file_uploader(
        "Choose a receipt image",
        type=["png", "jpg", "jpeg", "bmp", "tiff", "webp"],
        help="Supported formats: PNG, JPG, JPEG, BMP, TIFF, WEBP"
    )

    if uploaded_file is not None:
        col1, col2 = st.columns([1, 2])

        with col1:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Receipt", use_container_width=True)

            # Image info
            st.caption(f"Size: {image.size[0]}x{image.size[1]} pixels")
            st.caption(f"Format: {image.format}")

        with col2:
            # Processing options
            st.markdown("**Processing Options**")

            parallel_processing = st.checkbox(
                "Parallel Processing",
                value=True,
                help="Process with all providers simultaneously (faster)"
            )

            timeout = st.slider(
                "Timeout (seconds)",
                min_value=10,
                max_value=120,
                value=60,
                help="Maximum time to wait for each provider"
            )

            # Process button
            if st.button("🚀 Process Receipt", type="primary", use_container_width=True):
                process_receipt(uploaded_file, parallel_processing, timeout)


def process_receipt(uploaded_file, parallel: bool, timeout: int):
    """Process receipt with selected providers."""
    # Validate that providers are configured
    config_manager = st.session_state.config_manager
    registry = st.session_state.provider_registry

    configured_providers = []

    with st.spinner("Validating configurations..."):
        for provider_name in st.session_state.selected_providers:
            config = st.session_state.provider_configs.get(provider_name, {})
            is_valid, error = config_manager.validate_config(provider_name, config)

            if not is_valid:
                st.error(f"❌ {provider_name}: {error}")
                return

            try:
                provider = registry.get_provider(provider_name, config)
                configured_providers.append(provider)
            except Exception as e:
                st.error(f"❌ Failed to initialize {provider_name}: {str(e)}")
                return

    if not configured_providers:
        st.error("No providers configured. Please configure at least one provider.")
        return

    # Save uploaded file temporarily
    temp_path = Path("/tmp") / uploaded_file.name
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Process with providers
    with st.spinner(f"Processing with {len(configured_providers)} provider(s)..."):
        try:
            if parallel:
                # Use comparison engine for parallel processing
                engine = ComparisonEngine(configured_providers)
                results = engine.process_parallel(str(temp_path), timeout=timeout)
                comparison_report = engine.compare_results(results)
                st.session_state.comparison_report = comparison_report
            else:
                # Sequential processing
                results = []
                progress_bar = st.progress(0)
                for idx, provider in enumerate(configured_providers):
                    st.text(f"Processing with {provider.name}...")
                    result = provider.process_image(str(temp_path))
                    results.append(result)
                    progress_bar.progress((idx + 1) / len(configured_providers))

                # Generate comparison report
                engine = ComparisonEngine(configured_providers)
                comparison_report = engine.compare_results(results)
                st.session_state.comparison_report = comparison_report

            st.session_state.results = results
            st.success(f"✅ Processing complete! Processed by {len(results)} provider(s)")

        except Exception as e:
            st.error(f"❌ Processing failed: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
        finally:
            # Cleanup temp file
            if temp_path.exists():
                temp_path.unlink()


def render_results():
    """Render OCR results and comparison."""
    if st.session_state.results is None:
        return

    st.subheader("📊 Results")

    # Summary metrics
    render_summary_metrics()

    st.markdown("---")

    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Individual Results", "🔄 Comparison", "📈 Metrics", "💾 Export"])

    with tab1:
        render_individual_results()

    with tab2:
        render_comparison_view()

    with tab3:
        render_metrics_view()

    with tab4:
        render_export_options()


def render_summary_metrics():
    """Render summary metrics."""
    results = st.session_state.results
    comparison = st.session_state.comparison_report

    if not results or not comparison:
        return

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Providers Used",
            len(results),
            help="Number of OCR providers"
        )

    with col2:
        success_count = len([r for r in results if r.error is None])
        st.metric(
            "Success Rate",
            f"{comparison.success_rate * 100:.1f}%",
            f"{success_count}/{len(results)}",
            help="Percentage of successful processing"
        )

    with col3:
        avg_time = sum(comparison.processing_times.values()) / len(comparison.processing_times) if comparison.processing_times else 0
        st.metric(
            "Avg Processing Time",
            f"{avg_time:.2f}s",
            help="Average processing time across providers"
        )

    with col4:
        if comparison.fastest_provider:
            fastest_time = comparison.processing_times.get(comparison.fastest_provider, 0)
            st.metric(
                "Fastest Provider",
                comparison.fastest_provider,
                f"{fastest_time:.2f}s",
                help="Provider with shortest processing time"
            )


def render_individual_results():
    """Render individual provider results."""
    results = st.session_state.results

    for result in results:
        with st.expander(f"🔍 {result.provider} - {'✅ Success' if result.error is None else '❌ Failed'}", expanded=result.error is None):
            col1, col2 = st.columns([1, 1])

            with col1:
                st.markdown("**Processing Info**")
                st.write(f"⏱️ Time: {result.processing_time:.2f}s")
                if result.confidence_score:
                    st.write(f"📊 Confidence: {result.confidence_score * 100:.1f}%")
                st.write(f"🕐 Timestamp: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")

                if result.error:
                    st.error(f"Error: {result.error}")

            with col2:
                st.markdown("**Extracted Data**")
                if result.normalized_data:
                    # Display key fields
                    data = result.normalized_data
                    if data.get("merchant_name"):
                        st.write(f"🏪 Merchant: {data['merchant_name']}")
                    if data.get("total_amount"):
                        currency = data.get("currency", "USD")
                        st.write(f"💰 Total: {currency} {data['total_amount']}")
                    if data.get("transaction_date"):
                        st.write(f"📅 Date: {data['transaction_date']}")
                    if data.get("tax_amount"):
                        st.write(f"🧾 Tax: {data.get('currency', 'USD')} {data['tax_amount']}")

            # Raw response in collapsible section
            if result.raw_response:
                with st.expander("View Raw Response"):
                    st.json(result.raw_response)

            # Normalized data
            if result.normalized_data:
                with st.expander("View Normalized Data"):
                    st.json(convert_decimals_to_float(result.normalized_data))


def render_comparison_view():
    """Render side-by-side comparison of results."""
    comparison = st.session_state.comparison_report

    if not comparison:
        st.info("No comparison data available")
        return

    st.markdown("### Field-by-Field Comparison")

    # Create comparison table
    comparison_data = []

    for field_comp in comparison.field_comparisons:
        row = {
            "Field": field_comp.field_name.replace("_", " ").title(),
            "Agreement": f"{field_comp.agreement_percentage:.1f}%",
            "Consensus": str(field_comp.consensus_value) if field_comp.consensus_value else "N/A"
        }

        # Add provider values
        for provider, value in field_comp.values.items():
            row[provider] = str(value) if value is not None else "N/A"

        comparison_data.append(row)

    if comparison_data:
        df = pd.DataFrame(comparison_data)
        st.dataframe(df, use_container_width=True, hide_index=True)

    # Highlight disagreements
    st.markdown("### Disagreements")
    disagreements = [
        comp for comp in comparison.field_comparisons
        if comp.agreement_percentage < 100 and comp.total_providers > 1
    ]

    if disagreements:
        for comp in disagreements:
            with st.expander(f"⚠️ {comp.field_name} - {comp.agreement_percentage:.1f}% agreement"):
                st.write(f"**Consensus:** {comp.consensus_value}")
                st.write(f"**Agreement:** {comp.agreement_count}/{comp.total_providers} providers")
                st.write("**Values by provider:**")
                for provider, value in comp.values.items():
                    emoji = "✅" if str(value).lower().strip() == str(comp.consensus_value).lower().strip() else "❌"
                    st.write(f"{emoji} {provider}: {value}")
    else:
        st.success("✅ All providers agree on all fields!")


def render_metrics_view():
    """Render detailed metrics and analytics."""
    comparison = st.session_state.comparison_report

    if not comparison:
        return

    # Processing times comparison
    st.markdown("### ⏱️ Processing Time Comparison")
    times_df = pd.DataFrame([
        {"Provider": provider, "Time (seconds)": time}
        for provider, time in comparison.processing_times.items()
    ]).sort_values("Time (seconds)")

    st.bar_chart(times_df.set_index("Provider"))
    st.dataframe(times_df, use_container_width=True, hide_index=True)

    # Confidence scores
    if comparison.confidence_scores:
        st.markdown("### 📊 Confidence Scores")
        conf_df = pd.DataFrame([
            {"Provider": provider, "Confidence": score * 100}
            for provider, score in comparison.confidence_scores.items()
        ]).sort_values("Confidence", ascending=False)

        st.bar_chart(conf_df.set_index("Provider"))
        st.dataframe(conf_df, use_container_width=True, hide_index=True)

    # Accuracy based on consensus
    st.markdown("### 🎯 Accuracy (Based on Consensus)")

    from core.comparison_engine import ComparisonEngine
    engine = ComparisonEngine([])
    accuracy_scores = engine.calculate_provider_accuracy(comparison)

    acc_df = pd.DataFrame([
        {"Provider": provider, "Accuracy (%)": score}
        for provider, score in accuracy_scores.items()
    ]).sort_values("Accuracy (%)", ascending=False)

    st.bar_chart(acc_df.set_index("Provider"))
    st.dataframe(acc_df, use_container_width=True, hide_index=True)


def render_export_options():
    """Render export functionality."""
    results = st.session_state.results
    comparison = st.session_state.comparison_report

    if not results:
        return

    st.markdown("### 💾 Export Options")

    col1, col2, col3 = st.columns(3)

    with col1:
        # Export as JSON
        if st.button("📄 Export as JSON", use_container_width=True):
            export_data = {
                "results": [
                    {
                        "provider": r.provider,
                        "normalized_data": convert_decimals_to_float(r.normalized_data),
                        "confidence_score": r.confidence_score,
                        "processing_time": r.processing_time,
                        "error": r.error,
                        "timestamp": r.timestamp.isoformat()
                    }
                    for r in results
                ],
                "comparison": {
                    "consensus_data": comparison.consensus_data if comparison else {},
                    "success_rate": comparison.success_rate if comparison else 0,
                    "processing_times": comparison.processing_times if comparison else {}
                }
            }

            json_str = json.dumps(export_data, indent=2)
            st.download_button(
                label="Download JSON",
                data=json_str,
                file_name="ocr_results.json",
                mime="application/json"
            )

    with col2:
        # Export as CSV
        if st.button("📊 Export as CSV", use_container_width=True):
            csv_data = []
            for result in results:
                if result.normalized_data:
                    row = {
                        "provider": result.provider,
                        "processing_time": result.processing_time,
                        "confidence_score": result.confidence_score,
                        "error": result.error,
                        **convert_decimals_to_float(result.normalized_data)
                    }
                    csv_data.append(row)

            if csv_data:
                df = pd.DataFrame(csv_data)
                csv_str = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv_str,
                    file_name="ocr_results.csv",
                    mime="text/csv"
                )

    with col3:
        # Export comparison report
        if st.button("📋 Export Comparison", use_container_width=True) and comparison:
            comparison_data = {
                "field_comparisons": [
                    {
                        "field": comp.field_name,
                        "consensus": str(comp.consensus_value),
                        "agreement_percentage": comp.agreement_percentage,
                        "values": {k: str(v) for k, v in comp.values.items()}
                    }
                    for comp in comparison.field_comparisons
                ],
                "summary": {
                    "success_rate": comparison.success_rate,
                    "fastest_provider": comparison.fastest_provider,
                    "most_confident_provider": comparison.most_confident_provider
                }
            }

            json_str = json.dumps(comparison_data, indent=2)
            st.download_button(
                label="Download Comparison",
                data=json_str,
                file_name="comparison_report.json",
                mime="application/json"
            )


def convert_decimals_to_float(data: Any) -> Any:
    """Recursively convert Decimal objects to float for JSON serialization."""
    if isinstance(data, dict):
        return {k: convert_decimals_to_float(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_decimals_to_float(item) for item in data]
    elif isinstance(data, Decimal):
        return float(data)
    else:
        return data


def main():
    """Main application entry point."""
    # Render sidebar
    render_sidebar()

    # Main content
    st.title("🔍 OCR API Testing Lab")
    st.markdown("Interactive environment for testing and comparing OCR providers for receipt scanning")

    # Provider configuration
    if st.session_state.selected_providers:
        with st.expander("⚙️ Provider Configuration", expanded=not st.session_state.results):
            render_provider_configuration()

        st.markdown("---")

        # Image upload and processing
        render_image_upload()

        st.markdown("---")

        # Results
        render_results()
    else:
        st.info("👈 Please select at least one provider from the sidebar to get started")

        # Show available providers
        st.markdown("### Available Providers")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Receipt-Specific Providers:**")
            st.markdown("- 🎯 Veryfi - Specialized receipt parsing")
            st.markdown("- 🎯 Taggun - AI-powered receipt OCR")
            st.markdown("- 🎯 Klippa - Document recognition")

        with col2:
            st.markdown("**AI Vision Providers:**")
            st.markdown("- 🤖 OpenAI (ChatGPT)")
            st.markdown("- 🤖 Mistral AI")
            st.markdown("- 🤖 Google Gemini")


if __name__ == "__main__":
    main()
