"""Simple response assembler for formatting chunks with length limits."""

from typing import List

import tiktoken

from src.retriever.data_classes import Chunk, KnowledgeContext


class ResponseAssembler:
    """Assembles chunks into formatted responses with length constraints."""

    def __init__(self, encoding_name: str = "cl100k_base"):
        """Initialize with token encoding for length calculation."""
        self.encoding = tiktoken.get_encoding(encoding_name)

    def assemble_response(
        self,
        context: KnowledgeContext,
        max_response_length: int = 4000,
        format_style: str = "detailed",
        chunk_limit_mode: str = "token_based",  # "token_based" or "count_based"
    ) -> str:
        """
        Assemble chunks into formatted response within length limit.

        Args:
            context: KnowledgeContext with priority-ordered chunks
            max_response_length: Maximum response length in tokens
            format_style: Response format ("detailed", "summary", "raw")
            chunk_limit_mode: "token_based" respects token limits, "count_based" includes all chunks

        Returns:
            Formatted response trimmed to fit within max_response_length
        """
        if context.total_chunks == 0:
            return self._format_no_results(context.query)

        # Combine chunks in priority order (primary first, then referenced)
        all_chunks = list(context.primary_chunks) + list(context.referenced_chunks)

        # Build response incrementally, checking length
        if format_style == "detailed":
            return self._assemble_detailed(
                context, all_chunks, max_response_length, chunk_limit_mode
            )
        elif format_style == "summary":
            return self._assemble_summary(
                context, all_chunks, max_response_length, chunk_limit_mode
            )
        else:  # raw
            return self._assemble_raw(all_chunks, max_response_length, chunk_limit_mode)

    def _assemble_detailed(
        self,
        context: KnowledgeContext,
        chunks: List[Chunk],
        max_length: int,
        chunk_limit_mode: str = "token_based",
    ) -> str:
        """Assemble detailed response with metadata."""
        response_parts = [
            f'Based on your query: "{context.query}"\n',
            "I found the following relevant API documentation:\n",
        ]

        # Reserve space for footer
        footer = self._build_footer(context)
        footer_tokens = self._count_tokens("\n".join(response_parts) + footer)
        available_tokens = max_length - footer_tokens

        # Add chunks until we hit the limit
        included_chunks = 0
        for i, chunk in enumerate(chunks, 1):
            chunk_section = self._format_chunk_detailed(i, chunk)

            if chunk_limit_mode == "count_based":
                # Include all chunks regardless of token count
                response_parts.append(chunk_section)
                response_parts.append("")  # Empty line
                included_chunks += 1
            else:
                # Token-based limiting (original behavior)
                current_content = "\n".join(response_parts + [chunk_section])
                if self._count_tokens(current_content) <= available_tokens:
                    response_parts.append(chunk_section)
                    response_parts.append("")  # Empty line
                    included_chunks += 1
                else:
                    break

        # Add truncation notice if needed
        if included_chunks < len(chunks):
            truncated = len(chunks) - included_chunks
            response_parts.append(
                f"... (truncated {truncated} additional chunks to fit response limit)"
            )
            response_parts.append("")

        response_parts.append(footer)
        return "\n".join(response_parts)

    def _assemble_summary(
        self,
        context: KnowledgeContext,
        chunks: List[Chunk],
        max_length: int,
        chunk_limit_mode: str = "token_based",
    ) -> str:
        """Assemble concise summary response."""
        response_parts = [
            f'Found {len(chunks)} relevant chunks for: "{context.query}"\n'
        ]

        # Reserve space for footer
        footer = self._build_footer(context)
        footer_tokens = self._count_tokens("\n".join(response_parts) + footer)
        available_tokens = max_length - footer_tokens

        # Add chunks in summary format
        included_chunks = 0
        for chunk in chunks:
            chunk_summary = f"• {chunk.id}: {chunk.document[:100]}..."

            if chunk_limit_mode == "count_based":
                response_parts.append(chunk_summary)
                included_chunks += 1
            else:
                current_content = "\n".join(response_parts + [chunk_summary])
                if self._count_tokens(current_content) <= available_tokens:
                    response_parts.append(chunk_summary)
                    included_chunks += 1
                else:
                    break

        if included_chunks < len(chunks):
            response_parts.append(
                f"... (showing {included_chunks}/{len(chunks)} chunks)"
            )

        response_parts.append("")
        response_parts.append(footer)
        return "\n".join(response_parts)

    def _assemble_raw(
        self,
        chunks: List[Chunk],
        max_length: int,
        chunk_limit_mode: str = "token_based",
    ) -> str:
        """Assemble raw chunk content only."""
        response_parts = []

        for chunk in chunks:
            chunk_content = f"=== {chunk.id} ===\n{chunk.document}\n"

            if chunk_limit_mode == "count_based":
                response_parts.append(chunk_content)
            else:
                current_content = "\n".join(response_parts + [chunk_content])
                if self._count_tokens(current_content) <= max_length:
                    response_parts.append(chunk_content)
                else:
                    break

        return "\n".join(response_parts)

    def _format_chunk_detailed(self, index: int, chunk: Chunk) -> str:
        """Format individual chunk with metadata."""
        chunk_info = f"{index}. {chunk.id}"
        if chunk.metadata.get("type"):
            chunk_info += f" ({chunk.metadata['type']})"

        # Include relevance score if available
        if hasattr(chunk, "relevance_score") and chunk.relevance_score is not None:
            chunk_info += f" [relevance: {chunk.relevance_score:.2f}]"

        # Show more content, let the main assembly logic handle truncation
        return f"{chunk_info}\n{chunk.document}"

    def _build_footer(self, context: KnowledgeContext) -> str:
        """Build response footer with metadata."""
        return f"""---
Confidence: high (from pre-built index)
Sources: {context.total_chunks} chunks
Response time: {context.retrieval_stats.total_time_ms:.0f}ms retrieval"""

    def _format_no_results(self, query: str) -> str:
        """Format response when no chunks found."""
        return f"""No relevant API documentation found for: "{query}"

This might mean:
1. The query is too specific or uses terminology not in the knowledge base
2. Try rephrasing with more general API terms
3. The indexed documentation might not cover this topic

---
Confidence: low (no matches)
Sources: 0 chunks"""

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken."""
        return len(self.encoding.encode(text))
