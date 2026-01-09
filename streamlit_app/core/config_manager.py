"""Configuration management system for OCR providers."""

import os
import yaml
from typing import Dict, Any, List, Optional, Type
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from project root with fallback
# Use explicit path resolution to handle different working directories
env_path = Path(__file__).parent.parent.parent / ".env"
if not env_path.exists():
    # Fallback to streamlit_app/.env
    env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path, override=False)


class ConfigManager:
    """Manages provider configurations and registry."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration manager.

        Args:
            config_path: Path to providers.yaml config file
        """
        if config_path is None:
            # Default to config/providers.yaml relative to this file
            config_path = Path(__file__).parent.parent / "config" / "providers.yaml"

        self.config_path = Path(config_path)
        self.providers_config = self._load_config()

        # Store the env path for reload functionality
        self.env_path = self._get_env_path()

    def _get_env_path(self) -> Path:
        """Get the path to the .env file.

        Returns:
            Path to .env file (project root or streamlit_app fallback)
        """
        env_path = Path(__file__).parent.parent.parent / ".env"
        if not env_path.exists():
            # Fallback to streamlit_app/.env
            env_path = Path(__file__).parent.parent / ".env"
        return env_path

    def reload_env(self) -> tuple[Path, bool]:
        """Reload environment variables from .env file.

        This method forcefully reloads the .env file, overriding any existing
        environment variables. Use this when you want to pick up changes to
        the .env file without restarting the application.

        Returns:
            Tuple of (env_path, success) where env_path is the path that was
            loaded and success indicates if the file exists and was loaded.
        """
        env_path = self._get_env_path()
        exists = env_path.exists()

        if exists:
            # Use override=True to force reload and update existing env vars
            load_dotenv(dotenv_path=env_path, override=True)

        return env_path, exists

    def _load_config(self) -> Dict[str, Any]:
        """Load provider configurations from YAML file.

        Returns:
            Dictionary of provider configurations

        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is invalid
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, "r") as f:
            config = yaml.safe_load(f)

        if not config or "providers" not in config:
            raise ValueError("Invalid config file: missing 'providers' key")

        return config["providers"]

    def get_provider_names(self) -> List[str]:
        """Get list of all available provider names.

        Returns:
            List of provider names
        """
        return list(self.providers_config.keys())

    def get_provider_config(self, provider_name: str) -> Dict[str, Any]:
        """Get configuration metadata for a specific provider.

        Args:
            provider_name: Name of the provider

        Returns:
            Provider configuration dictionary

        Raises:
            ValueError: If provider not found
        """
        if provider_name not in self.providers_config:
            raise ValueError(f"Unknown provider: {provider_name}")

        return self.providers_config[provider_name]

    def get_config_schema(self, provider_name: str) -> Dict[str, Any]:
        """Get configuration schema for a provider.

        Args:
            provider_name: Name of the provider

        Returns:
            Configuration schema dictionary
        """
        provider_config = self.get_provider_config(provider_name)
        return provider_config.get("config_schema", {})

    def load_config_from_env(self, provider_name: str) -> Dict[str, Any]:
        """Load provider configuration from environment variables.

        Args:
            provider_name: Name of the provider

        Returns:
            Configuration dictionary with values from environment

        Raises:
            ValueError: If required environment variables are missing
        """
        schema = self.get_config_schema(provider_name)
        config = {}
        missing_required = []

        for key, field_schema in schema.items():
            # Try to load from environment variable
            env_var = field_schema.get("env_var")
            if env_var:
                value = os.getenv(env_var)
                if value:
                    # Convert type based on schema
                    field_type = field_schema.get("type", "string")
                    try:
                        config[key] = self._convert_type(value, field_type)
                    except (ValueError, TypeError) as e:
                        raise ValueError(
                            f"Invalid value for {key} from {env_var}: {str(e)}"
                        )
                elif field_schema.get("required", False):
                    # Required field missing from env
                    missing_required.append(env_var)
                else:
                    # Use default value if available
                    if "default" in field_schema:
                        config[key] = field_schema["default"]
            else:
                # No env var, use default if available
                if "default" in field_schema:
                    config[key] = field_schema["default"]

        # Report missing required environment variables
        if missing_required:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing_required)}\n"
                f"Please set these in your .env file."
            )

        return config

    def _convert_type(self, value: str, field_type: str) -> Any:
        """Convert string value to appropriate type.

        Args:
            value: String value from environment
            field_type: Target type from schema

        Returns:
            Converted value
        """
        if field_type == "boolean":
            return value.lower() in ("true", "1", "yes", "on")
        elif field_type == "number":
            return float(value)
        elif field_type == "integer":
            return int(value)
        elif field_type == "array":
            return [item.strip() for item in value.split(",")]
        else:  # string
            return value

    def validate_config(self, provider_name: str, config: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """Validate provider configuration against schema.

        Args:
            provider_name: Name of the provider
            config: Configuration dictionary to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        schema = self.get_config_schema(provider_name)

        # Check required fields
        for key, field_schema in schema.items():
            if field_schema.get("required", False):
                if key not in config or config[key] is None or config[key] == "":
                    return False, f"Required field missing: {key}"

        # Check field types and constraints
        for key, value in config.items():
            if key not in schema:
                continue  # Allow extra fields

            field_schema = schema[key]
            field_type = field_schema.get("type", "string")

            # Type validation
            if field_type == "boolean" and not isinstance(value, bool):
                return False, f"Field '{key}' must be boolean"
            elif field_type in ("number", "integer") and not isinstance(value, (int, float)):
                return False, f"Field '{key}' must be numeric"
            elif field_type == "array" and not isinstance(value, list):
                return False, f"Field '{key}' must be array"

            # Options validation
            if "options" in field_schema:
                if field_type == "array":
                    invalid_items = [item for item in value if item not in field_schema["options"]]
                    if invalid_items:
                        return False, f"Field '{key}' contains invalid items: {invalid_items}"
                else:
                    if value not in field_schema["options"]:
                        return False, f"Field '{key}' must be one of: {field_schema['options']}"

        return True, None

    def get_providers_by_type(self, provider_type: str) -> List[str]:
        """Get providers filtered by type.

        Args:
            provider_type: Type of provider (e.g., 'receipt_specific', 'cloud_ocr')

        Returns:
            List of provider names matching the type
        """
        return [
            name
            for name, config in self.providers_config.items()
            if config.get("type") == provider_type
        ]

    def get_provider_metadata(self, provider_name: str) -> Dict[str, Any]:
        """Get provider metadata (name, type, description, etc.).

        Args:
            provider_name: Name of the provider

        Returns:
            Metadata dictionary
        """
        config = self.get_provider_config(provider_name)
        return {
            "name": config.get("name", provider_name),
            "type": config.get("type", "unknown"),
            "description": config.get("description", ""),
            "documentation_url": config.get("documentation_url", "")
        }


class ProviderRegistry:
    """Registry for managing OCR provider instances."""

    def __init__(self):
        """Initialize provider registry."""
        self._providers: Dict[str, Type] = {}
        self._instances: Dict[str, Any] = {}

    def register(self, name: str, provider_class: Type) -> None:
        """Register a provider class.

        Args:
            name: Provider name (should match config key)
            provider_class: Provider class implementing OCRProvider interface
        """
        self._providers[name] = provider_class

    def get_provider(self, name: str, config: Optional[Dict[str, Any]] = None):
        """Get or create provider instance.

        Args:
            name: Provider name
            config: Optional configuration to apply

        Returns:
            Provider instance

        Raises:
            ValueError: If provider not registered
        """
        if name not in self._providers:
            raise ValueError(
                f"Provider '{name}' not registered. "
                f"Available providers: {list(self._providers.keys())}"
            )

        # Create new instance
        provider = self._providers[name]()

        # Apply configuration if provided
        if config:
            provider.configure(config)

        return provider

    def list_providers(self) -> List[str]:
        """Get list of registered provider names.

        Returns:
            List of provider names
        """
        return list(self._providers.keys())

    def is_registered(self, name: str) -> bool:
        """Check if provider is registered.

        Args:
            name: Provider name

        Returns:
            True if provider is registered
        """
        return name in self._providers
