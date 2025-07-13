"""OpenAPI Validator for specification validation."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    from ..cli.config import Config


@dataclass
class ValidationResult:
    """Result of OpenAPI validation."""

    is_valid: bool
    errors: List[str] = None
    warnings: List[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []


@dataclass
class ValidatorConfig:
    """Configuration for OpenAPI validation."""

    min_openapi_version: str = "3.0.0"
    require_info_section: bool = True
    require_paths_or_components: bool = True

    @classmethod
    def from_config(cls, config: "Config") -> "ValidatorConfig":
        """Create config from Config object."""
        return cls(
            min_openapi_version=config.min_openapi_version,
            require_info_section=config.require_info_section,
            require_paths_or_components=config.require_paths_or_components,
        )


class OpenAPIValidator:
    """Validates OpenAPI specification structure and requirements."""

    def __init__(self, config: ValidatorConfig = None):
        self.config = config or ValidatorConfig()

    def validate(self, spec_data: Dict[str, Any]) -> ValidationResult:
        """
        Validate an OpenAPI specification.

        Args:
            spec_data: Parsed OpenAPI specification data

        Returns:
            ValidationResult with validation status and any errors/warnings
        """
        if spec_data is None:
            return ValidationResult(is_valid=False, errors=["Specification data is None"])

        if not isinstance(spec_data, dict):
            return ValidationResult(is_valid=False, errors=["Specification data must be a dictionary"])

        errors = []
        warnings = []

        # Check for openapi field and version
        openapi_error = self._validate_openapi_version(spec_data)
        if openapi_error:
            errors.append(openapi_error)

        # Check for required info section
        if self.config.require_info_section:
            info_error = self._validate_info_section(spec_data)
            if info_error:
                errors.append(info_error)

        # Check for paths or components
        if self.config.require_paths_or_components:
            content_error = self._validate_content_sections(spec_data)
            if content_error:
                errors.append(content_error)

        # Additional structural validations
        structure_errors = self._validate_structure(spec_data)
        errors.extend(structure_errors)

        return ValidationResult(is_valid=len(errors) == 0, errors=errors, warnings=warnings)

    def _validate_openapi_version(self, spec_data: Dict[str, Any]) -> str:
        """Validate the openapi version field."""
        if "openapi" not in spec_data:
            return "Missing required 'openapi' field"

        openapi_version = spec_data["openapi"]

        if not isinstance(openapi_version, str):
            return "OpenAPI version must be a string"

        # Simple version comparison for OpenAPI versions (e.g., "3.0.0", "3.1.0")
        if not self._is_version_supported(openapi_version):
            return (
                f"OpenAPI version {openapi_version} is below minimum required version {self.config.min_openapi_version}"
            )

        return None

    def _is_version_supported(self, spec_version: str) -> bool:
        """Simple version comparison for OpenAPI versions."""
        try:
            # Parse version strings like "3.0.0" or "3.1.0"
            spec_parts = [int(x) for x in spec_version.split(".")]
            min_parts = [int(x) for x in self.config.min_openapi_version.split(".")]

            # Pad shorter version with zeros
            max_len = max(len(spec_parts), len(min_parts))
            spec_parts.extend([0] * (max_len - len(spec_parts)))
            min_parts.extend([0] * (max_len - len(min_parts)))

            return spec_parts >= min_parts
        except (ValueError, AttributeError):
            return False

    def _validate_info_section(self, spec_data: Dict[str, Any]) -> str:
        """Validate the info section."""
        if "info" not in spec_data:
            return "Missing required 'info' section"

        info = spec_data["info"]
        if not isinstance(info, dict):
            return "Info section must be an object"

        # Check for required info fields
        if "title" not in info:
            return "Info section missing required 'title' field"

        if "version" not in info:
            return "Info section missing required 'version' field"

        return None

    def _validate_content_sections(self, spec_data: Dict[str, Any]) -> str:
        """Validate that either paths or components section exists."""
        has_paths = "paths" in spec_data
        has_components = "components" in spec_data

        if not has_paths and not has_components:
            return "Specification must have either 'paths' or 'components' section"

        return None

    def _validate_structure(self, spec_data: Dict[str, Any]) -> List[str]:
        """Validate overall structure and common issues."""
        errors = []

        # Validate paths section if present
        if "paths" in spec_data:
            paths_errors = self._validate_paths_section(spec_data["paths"])
            errors.extend(paths_errors)

        # Validate components section if present
        if "components" in spec_data:
            components_errors = self._validate_components_section(spec_data["components"])
            errors.extend(components_errors)

        return errors

    def _validate_paths_section(self, paths: Any) -> List[str]:
        """Validate the paths section structure."""
        errors = []

        if not isinstance(paths, dict):
            errors.append("Paths section must be an object")
            return errors

        for path, path_item in paths.items():
            if not isinstance(path, str):
                errors.append(f"Path key must be a string, got {type(path).__name__}")
                continue

            if not path.startswith("/"):
                errors.append(f"Path '{path}' must start with '/'")

            if not isinstance(path_item, dict):
                errors.append(f"Path item for '{path}' must be an object")
                continue

        return errors

    def _validate_components_section(self, components: Any) -> List[str]:
        """Validate the components section structure."""
        errors = []

        if not isinstance(components, dict):
            errors.append("Components section must be an object")
            return errors

        # Validate known component types
        valid_component_types = {
            "schemas",
            "responses",
            "parameters",
            "examples",
            "requestBodies",
            "headers",
            "securitySchemes",
            "links",
            "callbacks",
        }

        for component_type, component_items in components.items():
            if component_type not in valid_component_types:
                # This is a warning, not an error (extensions allowed)
                continue

            if not isinstance(component_items, dict):
                errors.append(f"Component type '{component_type}' must be an object")
                continue

        return errors
