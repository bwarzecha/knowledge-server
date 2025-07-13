#!/usr/bin/env python3
"""
Quick prototype for OpenAPI chunking and graph construction.
Tests the file path prefix ID strategy and reference resolution.
"""

import yaml
import json
from pathlib import Path
from typing import Dict, List, Any, Set
from dataclasses import dataclass
import re


@dataclass
class Chunk:
    id: str
    document: str
    metadata: Dict[str, Any]


class OpenAPIChunker:
    def __init__(self, openapi_file_path: str, knowledge_root: str = None):
        self.file_path = Path(openapi_file_path)
        self.knowledge_root = Path(knowledge_root) if knowledge_root else self.file_path.parent.parent
        self.filename = self._generate_relative_prefix()
        self.spec = self._load_spec()
        self.chunks: List[Chunk] = []
        self.graph: Dict[str, List[str]] = {}
    
    def _generate_relative_prefix(self) -> str:
        """Generate prefix using relative path from knowledge root."""
        try:
            # Get relative path from knowledge root, INCLUDING extension for uniqueness
            relative_path = self.file_path.relative_to(self.knowledge_root)
            return str(relative_path).replace('/', '_').replace('.', '_')
        except ValueError:
            # Fallback if file is not under knowledge_root
            return str(self.file_path).replace('/', '_').replace('.', '_')
        
    def _load_spec(self) -> Dict[str, Any]:
        """Load OpenAPI specification from YAML or JSON."""
        with open(self.file_path, 'r') as f:
            if self.file_path.suffix.lower() in ['.yaml', '.yml']:
                return yaml.safe_load(f)
            else:
                return json.load(f)
    
    def _extract_refs(self, obj: Any, refs: Set[str] = None) -> Set[str]:
        """Recursively extract all $ref schema names from an object."""
        if refs is None:
            refs = set()
            
        if isinstance(obj, dict):
            if '$ref' in obj:
                # Extract schema name from #/components/schemas/SchemaName
                ref_path = obj['$ref']
                if ref_path.startswith('#/components/schemas/'):
                    schema_name = ref_path.split('/')[-1]
                    refs.add(schema_name)
            
            for value in obj.values():
                self._extract_refs(value, refs)
                
        elif isinstance(obj, list):
            for item in obj:
                self._extract_refs(item, refs)
                
        return refs
    
    def build_graph(self) -> Dict[str, List[str]]:
        """Build dependency graph using natural names from OpenAPI spec."""
        print(f"Building graph for {self.filename}...")
        
        # Process endpoints
        for path, methods in self.spec.get('paths', {}).items():
            for method, operation in methods.items():
                if not isinstance(operation, dict):
                    continue
                    
                # Get operation identifier
                operation_id = operation.get('operationId') or f"{method}_{path}"
                
                # Extract all schema references from this operation
                refs = self._extract_refs(operation)
                
                self.graph[operation_id] = list(refs)
                print(f"  {operation_id} -> {list(refs)}")
        
        # Process schemas (they may reference other schemas)
        for schema_name, schema_def in self.spec.get('components', {}).get('schemas', {}).items():
            refs = self._extract_refs(schema_def)
            self.graph[schema_name] = list(refs)
            if refs:
                print(f"  {schema_name} -> {list(refs)}")
        
        return self.graph
    
    def generate_ids(self) -> Dict[str, str]:
        """Generate file-prefixed IDs for all natural names."""
        id_mapping = {}
        
        # Endpoints
        for path, methods in self.spec.get('paths', {}).items():
            for method, operation in methods.items():
                if not isinstance(operation, dict):
                    continue
                    
                natural_name = operation.get('operationId') or f"{method}_{path}"
                id_mapping[natural_name] = f"{self.filename}:{natural_name}"
        
        # Schemas
        for schema_name in self.spec.get('components', {}).get('schemas', {}):
            id_mapping[schema_name] = f"{self.filename}:{schema_name}"
        
        return id_mapping
    
    def create_endpoint_chunk(self, path: str, method: str, operation: Dict[str, Any], 
                             id_mapping: Dict[str, str]) -> Chunk:
        """Create a chunk for an endpoint with inlined basic schema info."""
        operation_id = operation.get('operationId') or f"{method}_{path}"
        chunk_id = id_mapping[operation_id]
        
        # Build document content
        summary = operation.get('summary', '')
        description = operation.get('description', '')
        
        doc_lines = [
            f"{method.upper()} {path}",
            f"Operation: {operation_id}",
            ""
        ]
        
        if summary:
            doc_lines.append(f"Summary: {summary}")
        if description:
            doc_lines.append(f"Description: {description[:200]}...")
        
        doc_lines.append("")
        
        # Add parameters info
        parameters = operation.get('parameters', [])
        if parameters:
            doc_lines.append("Parameters:")
            for param in parameters[:3]:  # Limit for brevity
                if isinstance(param, dict) and '$ref' not in param:
                    name = param.get('name', 'unknown')
                    location = param.get('in', 'unknown')
                    required = param.get('required', False)
                    doc_lines.append(f"  - {name} ({location}) {'[required]' if required else '[optional]'}")
        
        # Add response info
        responses = operation.get('responses', {})
        if responses:
            doc_lines.append("Responses:")
            for status, response in list(responses.items())[:3]:  # Limit for brevity
                if isinstance(response, dict):
                    desc = response.get('description', '')
                    doc_lines.append(f"  - {status}: {desc}")
        
        document = "\n".join(doc_lines)
        
        # Get referenced schema IDs
        natural_refs = self.graph.get(operation_id, [])
        ref_ids = [id_mapping.get(ref, ref) for ref in natural_refs if ref in id_mapping]
        
        metadata = {
            "type": "endpoint",
            "source_file": str(self.file_path),
            "path": path,
            "method": method.upper(),
            "operationId": operation_id,
            "ref_ids": ref_ids,
            "tags": operation.get('tags', []),
            "chunk_strategy": "endpoint_with_inlined_schemas"
        }
        
        return Chunk(id=chunk_id, document=document, metadata=metadata)
    
    def create_schema_chunk(self, schema_name: str, schema_def: Dict[str, Any],
                           id_mapping: Dict[str, str]) -> Chunk:
        """Create a chunk for a schema definition."""
        chunk_id = id_mapping[schema_name]
        
        # Build document content
        doc_lines = [
            f"Schema: {schema_name}",
            ""
        ]
        
        # Add description
        description = schema_def.get('description', '')
        if description:
            doc_lines.append(f"Description: {description}")
            doc_lines.append("")
        
        # Add type info
        schema_type = schema_def.get('type', 'unknown')
        doc_lines.append(f"Type: {schema_type}")
        
        # Add properties (for object types)
        properties = schema_def.get('properties', {})
        if properties:
            doc_lines.append("Properties:")
            required = schema_def.get('required', [])
            for prop_name, prop_def in list(properties.items())[:10]:  # Limit for brevity
                prop_type = prop_def.get('type', 'unknown') if isinstance(prop_def, dict) else 'unknown'
                required_marker = " [required]" if prop_name in required else ""
                doc_lines.append(f"  - {prop_name}: {prop_type}{required_marker}")
        
        document = "\n".join(doc_lines)
        
        # Get referenced schema IDs
        natural_refs = self.graph.get(schema_name, [])
        ref_ids = [id_mapping.get(ref, ref) for ref in natural_refs if ref in id_mapping]
        
        metadata = {
            "type": "schema",
            "name": schema_name,
            "source_file": str(self.file_path),
            "ref_ids": ref_ids,
            "properties": list(properties.keys()) if properties else [],
            "schema_type": schema_type
        }
        
        return Chunk(id=chunk_id, document=document, metadata=metadata)
    
    def create_chunks(self) -> List[Chunk]:
        """Create all chunks for the OpenAPI specification."""
        print(f"\nCreating chunks for {self.filename}...")
        
        # Build graph first
        self.build_graph()
        
        # Generate ID mapping
        id_mapping = self.generate_ids()
        print(f"\nGenerated {len(id_mapping)} IDs:")
        for natural, prefixed in list(id_mapping.items())[:5]:
            print(f"  {natural} -> {prefixed}")
        if len(id_mapping) > 5:
            print(f"  ... and {len(id_mapping) - 5} more")
        
        chunks = []
        
        # Create endpoint chunks
        for path, methods in self.spec.get('paths', {}).items():
            for method, operation in methods.items():
                if isinstance(operation, dict):
                    chunk = self.create_endpoint_chunk(path, method, operation, id_mapping)
                    chunks.append(chunk)
        
        # Create schema chunks
        for schema_name, schema_def in self.spec.get('components', {}).get('schemas', {}).items():
            chunk = self.create_schema_chunk(schema_name, schema_def, id_mapping)
            chunks.append(chunk)
        
        self.chunks = chunks
        print(f"\nCreated {len(chunks)} chunks")
        
        return chunks
    
    def print_sample_chunks(self, limit: int = 3):
        """Print sample chunks for inspection."""
        print(f"\n=== Sample Chunks (showing {limit} of {len(self.chunks)}) ===")
        
        for i, chunk in enumerate(self.chunks[:limit]):
            print(f"\n--- Chunk {i+1}: {chunk.id} ---")
            print("Document:")
            print(chunk.document[:300] + "..." if len(chunk.document) > 300 else chunk.document)
            print("\nMetadata:")
            for key, value in chunk.metadata.items():
                if isinstance(value, list) and len(value) > 3:
                    print(f"  {key}: {value[:3]} ... (+{len(value)-3} more)")
                else:
                    print(f"  {key}: {value}")
    
    def test_reference_resolution(self):
        """Test that reference resolution works correctly."""
        print(f"\n=== Testing Reference Resolution ===")
        
        # Find an endpoint with references
        endpoint_chunks = [c for c in self.chunks if c.metadata["type"] == "endpoint"]
        if not endpoint_chunks:
            print("No endpoint chunks found!")
            return
        
        test_chunk = None
        for chunk in endpoint_chunks:
            if chunk.metadata.get("ref_ids"):
                test_chunk = chunk
                break
        
        if not test_chunk:
            print("No endpoint chunks with references found!")
            return
        
        print(f"Testing with endpoint: {test_chunk.id}")
        print(f"References: {test_chunk.metadata['ref_ids']}")
        
        # Check if referenced chunks exist
        chunk_ids = {c.id for c in self.chunks}
        for ref_id in test_chunk.metadata["ref_ids"]:
            if ref_id in chunk_ids:
                print(f"  ✅ {ref_id} - chunk exists")
            else:
                print(f"  ❌ {ref_id} - chunk missing!")


def export_chunks_to_file(chunks: List[Chunk], output_file: str):
    """Export chunks to JSON file for analysis."""
    import json
    
    chunk_data = []
    for chunk in chunks:
        chunk_data.append({
            "id": chunk.id,
            "document": chunk.document,
            "metadata": chunk.metadata
        })
    
    with open(output_file, 'w') as f:
        json.dump(chunk_data, f, indent=2)
    
    print(f"📄 Exported {len(chunks)} chunks to {output_file}")


def main():
    # Test with multiple OpenAPI specs using knowledge root
    knowledge_root = "/Users/bartosz/dev/knowledge-server"
    test_files = [
        "/Users/bartosz/dev/knowledge-server/samples/openapi.yaml",
        "/Users/bartosz/dev/knowledge-server/samples/openapi.json", 
        "/Users/bartosz/dev/knowledge-server/samples/SponsoredProducts_prod_3p.json"
    ]
    
    print("OpenAPI Chunking Prototype - Export for Analysis")
    print("=" * 60)
    
    all_chunks = []
    
    for openapi_file in test_files:
        print(f"\n📁 Processing {Path(openapi_file).name}...")
        
        try:
            chunker = OpenAPIChunker(openapi_file, knowledge_root)
            chunks = chunker.create_chunks()
            all_chunks.extend(chunks)
            
            print(f"   Prefix: '{chunker.filename}'")
            print(f"   Created {len(chunks)} chunks")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print(f"\n📊 Total: {len(all_chunks)} chunks from {len(test_files)} files")
    
    # Export chunks for analysis
    output_file = "/Users/bartosz/dev/knowledge-server/chunks_export.json"
    export_chunks_to_file(all_chunks, output_file)
    
    print(f"\n✅ Success! Chunks exported for analysis.")


if __name__ == "__main__":
    main()