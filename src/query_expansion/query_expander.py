#!/usr/bin/env python3
"""Query expander using LLM with API index."""

import json
import logging
from typing import Any, Dict

from src.llm.provider import create_llm_client

logger = logging.getLogger(__name__)


class QueryExpander:
    """Expand queries using pre-built API index and LLM."""

    def __init__(
        self,
        index_path: str = "data/api_index.json",
        llm_provider: str = "local",
        max_context_chars: int = 1200,
        **llm_config,
    ):
        self.index_path = index_path
        self.max_context_chars = max_context_chars
        self._index_cache = None
        self._compact_cache = None

        # Create LLM client
        try:
            self.llm = create_llm_client(llm_provider, **llm_config)
        except Exception as e:
            logger.warning(f"Warning: Failed to create LLM client: {e}")
            self.llm = None

    def expand_query(self, query: str) -> str:
        """Expand query using LLM and API index."""
        if not self.llm or not self.llm.is_available():
            logger.warning("LLM not available, returning original query")
            return query

        try:
            # Get compact API context
            api_context = self._get_compact_context()

            # Build prompt
            prompt = self._build_prompt(query, api_context)

            # Generate expansion
            response = self.llm.generate(prompt, temperature=0.2, max_tokens=150)

            # Parse and merge
            parsed = self._parse_llm_output(response)
            merged = self._merge_expansion(parsed)

            return merged if merged.strip() else query

        except Exception as e:
            logger.warning(f"Query expansion failed: {e}")
            return query

    def _load_index(self) -> Dict[str, Any]:
        """Load API index from file."""
        if self._index_cache is None:
            try:
                with open(self.index_path, "r") as f:
                    self._index_cache = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load index: {e}")
                self._index_cache = {"entries": []}
        return self._index_cache

    def _get_compact_context(self) -> str:
        """Get compact API context for LLM prompt."""
        if self._compact_cache is None:
            index_data = self._load_index()
            file_entries = index_data.get("files", [])

            # For text-based format, directly combine file texts within limit
            combined_texts = []
            char_count = 0

            for file_entry in file_entries:
                file_text = file_entry.get("text", "")

                # Check if adding this file would exceed the limit
                test_length = char_count + len(file_text) + 2  # +2 for newlines

                if test_length > self.max_context_chars:
                    # If we haven't added any content yet, take partial content from first file
                    if not combined_texts:
                        # Take as much as we can from the first file
                        available_chars = (
                            self.max_context_chars - 10
                        )  # Leave some buffer
                        truncated_text = file_text[:available_chars]
                        # Try to cut at a line boundary
                        if "\n" in truncated_text:
                            lines = truncated_text.split("\n")
                            truncated_text = "\n".join(
                                lines[:-1]
                            )  # Remove partial last line
                        combined_texts.append(truncated_text)
                    break

                combined_texts.append(file_text)
                char_count = test_length

            self._compact_cache = "\n\n".join(combined_texts)

        return self._compact_cache

    def _build_prompt(self, query: str, api_context: str) -> str:
        """Build the minimal prompt template."""
        return f"""You are an API-search assistant.
Expand the user's request so it matches *operationId*, HTTP method, path segments, tags,
and typical parameter names that appear in the **API index** below.
**Output** exactly 3 lines:

1. `keywords:` comma-separated lexical keywords & synonyms
2. `paths:` space-separated full or partial endpoint paths you judge relevant
3. `ops:` comma-separated operationIds (if any).
   Do **not** explain or add extra text.

**API INDEX**
{api_context}

**USER QUERY**
{query}"""

    def _parse_llm_output(self, llm_response: str) -> Dict[str, str]:
        """Parse the 3-line LLM output."""
        lines = llm_response.strip().split("\n")
        result = {"keywords": "", "paths": "", "ops": ""}

        for line in lines:
            line = line.strip()
            # Handle both with and without backticks
            if "keywords:" in line.lower():
                # Extract everything after "keywords:"
                idx = line.lower().find("keywords:")
                result["keywords"] = line[idx + 9 :].strip().strip("`")
            elif "paths:" in line.lower():
                idx = line.lower().find("paths:")
                result["paths"] = line[idx + 6 :].strip().strip("`")
            elif "ops:" in line.lower():
                idx = line.lower().find("ops:")
                result["ops"] = line[idx + 4 :].strip().strip("`")

        return result

    def _merge_expansion(self, parsed: Dict[str, str]) -> str:
        """Merge the 3 fields into single search string."""
        parts = []

        if parsed["keywords"]:
            parts.append(parsed["keywords"].replace(",", " "))
        if parsed["paths"]:
            parts.append(parsed["paths"])
        if parsed["ops"]:
            parts.append(parsed["ops"].replace(",", " "))

        return " ".join(parts)
