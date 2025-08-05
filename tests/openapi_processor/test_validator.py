"""Tests for OpenAPI Validator."""

from src.openapi_processor.validator import OpenAPIValidator, ValidatorConfig


class TestOpenAPIValidator:
    """Test cases for OpenAPIValidator."""

    def test_validate_valid_minimal_spec(self):
        """Test validation of a minimal valid OpenAPI spec."""
        spec_data = {
            "openapi": "3.0.0",
            "info": {"title": "Test API", "version": "1.0.0"},
            "paths": {},
        }

        validator = OpenAPIValidator()
        result = validator.validate(spec_data)

        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_validate_spec_with_components_only(self):
        """Test validation of spec with components but no paths."""
        spec_data = {
            "openapi": "3.0.0",
            "info": {"title": "Test API", "version": "1.0.0"},
            "components": {"schemas": {"Pet": {"type": "object"}}},
        }

        validator = OpenAPIValidator()
        result = validator.validate(spec_data)

        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_validate_missing_openapi_field(self):
        """Test validation failure when openapi field is missing."""
        spec_data = {"info": {"title": "Test API", "version": "1.0.0"}, "paths": {}}

        validator = OpenAPIValidator()
        result = validator.validate(spec_data)

        assert result.is_valid is False
        assert any(
            "Missing required 'openapi' field" in error for error in result.errors
        )

    def test_validate_missing_info_section(self):
        """Test validation failure when info section is missing."""
        spec_data = {"openapi": "3.0.0", "paths": {}}

        validator = OpenAPIValidator()
        result = validator.validate(spec_data)

        assert result.is_valid is False
        assert any(
            "Missing required 'info' section" in error for error in result.errors
        )

    def test_validate_missing_title_in_info(self):
        """Test validation failure when info.title is missing."""
        spec_data = {"openapi": "3.0.0", "info": {"version": "1.0.0"}, "paths": {}}

        validator = OpenAPIValidator()
        result = validator.validate(spec_data)

        assert result.is_valid is False
        assert any("missing required 'title' field" in error for error in result.errors)

    def test_validate_missing_version_in_info(self):
        """Test validation failure when info.version is missing."""
        spec_data = {"openapi": "3.0.0", "info": {"title": "Test API"}, "paths": {}}

        validator = OpenAPIValidator()
        result = validator.validate(spec_data)

        assert result.is_valid is False
        assert any(
            "missing required 'version' field" in error for error in result.errors
        )

    def test_validate_missing_paths_and_components(self):
        """Test validation failure when both paths and components are missing."""
        spec_data = {
            "openapi": "3.0.0",
            "info": {"title": "Test API", "version": "1.0.0"},
        }

        validator = OpenAPIValidator()
        result = validator.validate(spec_data)

        assert result.is_valid is False
        assert any(
            "must have either 'paths' or 'components'" in error
            for error in result.errors
        )

    def test_validate_version_comparison(self):
        """Test OpenAPI version comparison logic."""
        validator = OpenAPIValidator(ValidatorConfig(min_openapi_version="3.0.0"))

        # Test valid versions
        assert validator._is_version_supported("3.0.0") is True
        assert validator._is_version_supported("3.0.1") is True
        assert validator._is_version_supported("3.1.0") is True
        assert validator._is_version_supported("4.0.0") is True

        # Test invalid versions
        assert validator._is_version_supported("2.0.0") is False
        assert validator._is_version_supported("invalid") is False

    def test_validate_old_openapi_version(self):
        """Test validation failure for old OpenAPI version."""
        spec_data = {
            "openapi": "2.0.0",
            "info": {"title": "Test API", "version": "1.0.0"},
            "paths": {},
        }

        validator = OpenAPIValidator()
        result = validator.validate(spec_data)

        assert result.is_valid is False
        assert any("below minimum required version" in error for error in result.errors)

    def test_validate_invalid_paths_structure(self):
        """Test validation of invalid paths structure."""
        spec_data = {
            "openapi": "3.0.0",
            "info": {"title": "Test API", "version": "1.0.0"},
            "paths": {"invalid-path": {}},  # Should start with /
        }

        validator = OpenAPIValidator()
        result = validator.validate(spec_data)

        assert result.is_valid is False
        assert any("must start with '/'" in error for error in result.errors)

    def test_validate_none_input(self):
        """Test validation of None input."""
        validator = OpenAPIValidator()
        result = validator.validate(None)

        assert result.is_valid is False
        assert any("Specification data is None" in error for error in result.errors)

    def test_validate_non_dict_input(self):
        """Test validation of non-dictionary input."""
        validator = OpenAPIValidator()
        result = validator.validate("not a dict")

        assert result.is_valid is False
        assert any("must be a dictionary" in error for error in result.errors)

    def test_custom_config(self):
        """Test validator with custom configuration."""
        config = ValidatorConfig(
            min_openapi_version="3.1.0",
            require_info_section=False,
            require_paths_or_components=False,
        )

        spec_data = {"openapi": "3.1.0"}

        validator = OpenAPIValidator(config)
        result = validator.validate(spec_data)

        assert result.is_valid is True
        assert len(result.errors) == 0
