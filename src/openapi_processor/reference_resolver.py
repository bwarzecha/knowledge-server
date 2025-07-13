"""Reference Resolver for converting $ref strings to chunk IDs."""

from typing import Optional


class ReferenceResolver:
    """Resolves $ref strings to chunk IDs."""

    def resolve_ref_to_chunk_id(self, ref: str, spec_name: str) -> Optional[str]:
        """
        Convert $ref to chunk ID format.

        Args:
            ref: Reference string (e.g., "#/components/schemas/Pet")
            spec_name: Current spec name for context

        Returns:
            Chunk ID string or None if ref cannot be resolved
        """
        if not ref or not isinstance(ref, str):
            return None

        # Handle internal references (start with #/)
        if ref.startswith("#/"):
            return self._resolve_internal_ref(ref, spec_name)

        # Handle external file references (future enhancement)
        # For now, only support internal references
        return None

    def _resolve_internal_ref(self, ref: str, spec_name: str) -> Optional[str]:
        """Resolve internal reference to chunk ID."""
        # Remove leading #/
        path = ref[2:]

        # Convert to chunk ID format: {spec-name}:{element-path}
        chunk_id = f"{spec_name}:{path}"

        return chunk_id
