"""Reference Scanner for finding $ref dependencies in OpenAPI content."""

from typing import Any, Dict, List, Union


class ReferenceScanner:
    """Scans OpenAPI content for $ref references."""

    def find_references(self, content: Union[Dict, List, Any]) -> List[str]:
        """
        Find all $ref strings in content.

        Args:
            content: OpenAPI content to scan (dict, list, or other)

        Returns:
            List of unique $ref strings found
        """
        refs = set()
        self._scan_recursive(content, refs)
        return sorted(list(refs))

    def _scan_recursive(self, obj: Any, refs: set) -> None:
        """Recursively scan object for $ref occurrences."""
        if isinstance(obj, dict):
            self._scan_dict(obj, refs)
        elif isinstance(obj, list):
            self._scan_list(obj, refs)

    def _scan_dict(self, obj: Dict[str, Any], refs: set) -> None:
        """Scan dictionary for $ref keys and recurse into values."""
        for key, value in obj.items():
            if key == "$ref" and isinstance(value, str):
                refs.add(value)
            else:
                self._scan_recursive(value, refs)

    def _scan_list(self, obj: List[Any], refs: set) -> None:
        """Scan list items recursively."""
        for item in obj:
            self._scan_recursive(item, refs)
