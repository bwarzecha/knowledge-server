# OpenAPI Processor Component Specification

## Component Purpose

Transform OpenAPI specifications (JSON/YAML) into searchable chunks with metadata and reference tracking. This component implements the validated chunking strategy that achieved 80% query completeness in prototype testing.

## Core Responsibilities

1. **Parse OpenAPI Specifications**: Handle both JSON and YAML formats for OpenAPI 3.0+
2. **Generate File-Path-Based IDs**: Create collision-free identifiers using `{filename}:{natural_name}` format
3. **Create Intelligent Chunks**: Implement hybrid chunking (endpoint chunks + complex schema chunks)
4. **Build Reference Metadata**: Extract and map `$ref` dependencies for later resolution
5. **Generate Clean Documents**: Create human-readable text for each chunk suitable for embedding

## Input/Output Contracts

### Input
- **Specs Directory**: Configurable path containing OpenAPI files (.json, .yaml)
- **Configuration**: .env settings for chunking behavior and output format

### Output  
- **Chunk Objects**: List of dictionaries with this exact structure:
```python
{
    "id": str,           # Format: "{spec-name}:{element-path}"
    "document": str,     # YAML-formatted OpenAPI content (exact, no transformations)
    "metadata": {
        "type": str,                    # "info", "tags", "operation", "component"
        "source_file": str,             # Full relative path with extension
        "api_info_ref": str,            # Reference to info chunk: "{spec-name}:info"
        "api_tags_ref": str,            # Reference to tags chunk: "{spec-name}:tags" (if exists)
        "ref_ids": Dict[str, List[str]], # Hierarchical references
        "referenced_by": List[str],      # What chunks reference this one
        "natural_name": str,            # Original name from spec
        
        # Operation-specific metadata (type="operation")
        "path": str,                    # HTTP path
        "method": str,                  # HTTP method
        "operation_id": str,            # operationId
        "tags": List[str],              # OpenAPI tags
        
        # Component-specific metadata (type="component")
        "component_type": str,          # "schemas", "parameters", "responses", etc.
        "component_name": str           # Name of the component
    }
}
```

## Complete Processing Algorithm

### Overview
Transform OpenAPI specifications into chunks following a clear 5-phase process. Each OpenAPI element becomes a separate chunk with hierarchical reference tracking and unified YAML formatting.

### File-to-Chunk Mapping Strategy

**Core Principle**: One chunk per OpenAPI element (no inlining decisions needed).

**Mapping Rules**:
1. **`info` section** → **Info Chunk**  
2. **`tags` section** → **Tags Chunk** (if exists)
3. **Each operation in `paths`** → **Operation Chunk** (one per HTTP method)
4. **Each item in `components`** → **Component Chunk** (schemas, parameters, responses, etc.)

**Key Benefits**:
- No complex categorization logic needed
- Every element preserved exactly as written
- Complete dependency graph via hierarchical `ref_ids`
- Bidirectional references via `referenced_by`

### Content Preservation Requirements
**ABSOLUTE RULE**: Chunk documents must contain complete, unmodified OpenAPI content from the original specification.

**Prohibited Transformations**:
- ❌ No summarization or paraphrasing of descriptions
- ❌ No omission of fields, properties, or constraints  
- ❌ No reformatting of examples or enum values
- ❌ No simplification of complex schema structures
- ❌ No removal of "optional" or "advanced" fields

**Required Preservation**:
- ✅ Exact field names, descriptions, and examples as written
- ✅ All validation constraints (min/max, patterns, formats)
- ✅ Complete enum value lists and their descriptions
- ✅ Original OpenAPI structure and hierarchy
- ✅ All metadata (deprecated flags, extensions, etc.)

### Phase-by-Phase Processing Flow

**Phase 1: Directory Scanning**
1. Start at configured `OPENAPI_SPECS_DIR` path
2. Recursively traverse all subdirectories  
3. For each file discovered:
   - Check extension: `.json`, `.yaml`, `.yml`
   - Skip hidden files/directories (starting with `.`)
   - Skip non-OpenAPI files (package.json, etc.)
   - Process immediately when found (no queue needed)

**Phase 2: File Processing and Validation**
For each discovered file:
1. Read file content into memory
2. Store relative path with extension as `spec-name`
   - Example: `apis/payment/v2/openapi.yaml`
3. Parse content as JSON or YAML into Python objects
4. Validate OpenAPI structure:
   - Check for "openapi" field with version 3.0+
   - Verify "info" section exists
   - Must have "paths" or "components" sections
5. If validation fails: log error and skip to next file

**Phase 3: Element Extraction and Chunk Generation**
For each valid OpenAPI specification:
1. **Extract info section** → Create info chunk
2. **Extract tags section** (if exists) → Create tags chunk  
3. **Extract operations from paths**:
   - For each path (e.g., "/products"):
     - For each HTTP method (GET, POST, etc.):
       - Create operation chunk with ID: `{spec-name}:paths{path}/{method}`
4. **Extract components**:
   - For each component type (schemas, parameters, responses, etc.):
     - For each component:
       - Create component chunk with ID: `{spec-name}:components/{type}/{name}`
5. Convert all extracted content to YAML format using `yaml.dump()`

**Phase 4: Reference Resolution and Graph Building**  
For each generated chunk:
1. **Scan for references**: Recursively find all `$ref` occurrences in chunk content
2. **Convert references to chunk IDs**:
   - `#/components/schemas/Product` → `{spec-name}:components/schemas/Product`
   - Handle external file references if needed
3. **Build hierarchical ref_ids**: Recursively follow references to build complete dependency tree
4. **Build referenced_by lists**: Track which chunks reference this chunk
5. **Handle circular references**: Detect and prevent infinite loops during traversal

**Phase 5: Final Assembly and Output**
1. Accumulate all chunks in result list
2. Each chunk contains:
   - **id**: Unique identifier using spec-name + element path
   - **document**: YAML-formatted OpenAPI content (exact preservation)
   - **metadata**: Complete metadata including references and context
3. Return complete chunk list for all processed specifications

### Detailed Examples

**Example: Processing Single OpenAPI File**

*Input File: `apis/payment/v2/openapi.yaml`*
```yaml
openapi: 3.0.0
info:
  title: Payment API
  version: 2.0.0
tags:
  - name: Payments
paths:
  /payments:
    post:
      operationId: createPayment
      requestBody:
        $ref: '#/components/requestBodies/PaymentRequest'
components:
  requestBodies:
    PaymentRequest:
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/Payment'
  schemas:
    Payment:
      type: object
      properties:
        amount:
          $ref: '#/components/schemas/Money'
    Money:
      type: object
      properties:
        value: {type: number}
        currency: {type: string}
```

*Generated Chunks:*

**1. Info Chunk**
```python
{
    "id": "apis/payment/v2/openapi.yaml:info",
    "document": """title: Payment API
version: 2.0.0""",
    "metadata": {
        "type": "info",
        "source_file": "apis/payment/v2/openapi.yaml",
        "ref_ids": {},
        "referenced_by": []  # Will be populated by other chunks
    }
}
```

**2. Tags Chunk**  
```python
{
    "id": "apis/payment/v2/openapi.yaml:tags",
    "document": """- name: Payments""",
    "metadata": {
        "type": "tags", 
        "source_file": "apis/payment/v2/openapi.yaml",
        "ref_ids": {},
        "referenced_by": []
    }
}
```

**3. Operation Chunk**
```python
{
    "id": "apis/payment/v2/openapi.yaml:paths/payments/post",
    "document": """post:
  operationId: createPayment
  requestBody:
    $ref: '#/components/requestBodies/PaymentRequest'""",
    "metadata": {
        "type": "operation",
        "source_file": "apis/payment/v2/openapi.yaml",
        "api_info_ref": "apis/payment/v2/openapi.yaml:info",
        "api_tags_ref": "apis/payment/v2/openapi.yaml:tags", 
        "ref_ids": {
            "apis/payment/v2/openapi.yaml:components/requestBodies/PaymentRequest": [
                "apis/payment/v2/openapi.yaml:components/schemas/Payment",
                "apis/payment/v2/openapi.yaml:components/schemas/Money"
            ]
        },
        "referenced_by": [],
        "path": "/payments",
        "method": "post",
        "operation_id": "createPayment"
    }
}
```

**4. Component Chunks**
```python
{
    "id": "apis/payment/v2/openapi.yaml:components/schemas/Payment",
    "document": """Payment:
  type: object
  properties:
    amount:
      $ref: '#/components/schemas/Money'""",
    "metadata": {
        "type": "component",
        "source_file": "apis/payment/v2/openapi.yaml", 
        "api_info_ref": "apis/payment/v2/openapi.yaml:info",
        "api_tags_ref": "apis/payment/v2/openapi.yaml:tags",
        "ref_ids": {
            "apis/payment/v2/openapi.yaml:components/schemas/Money": []
        },
        "referenced_by": [
            "apis/payment/v2/openapi.yaml:paths/payments/post",
            "apis/payment/v2/openapi.yaml:components/requestBodies/PaymentRequest"
        ],
        "component_type": "schemas",
        "component_name": "Payment"
    }
}
```

## Subcomponent Architecture

The OpenAPI Processor is built from 8 distinct, testable subcomponents:

### 1. Directory Scanner
- **Input**: Root directory path
- **Output**: List of OpenAPI file paths (`.json`, `.yaml`, `.yml`)
- **Algorithm**: Recursive traversal, filter by extension, skip hidden files
- **Test**: Given test directory structure, returns correct file list

### 2. OpenAPI Validator  
- **Input**: Parsed JSON/YAML data structure
- **Output**: Boolean (valid/invalid) + error details
- **Algorithm**: Check "openapi" version ≥3.0, verify "info" exists, require "paths" or "components"
- **Test**: Various valid/invalid OpenAPI structures

### 3. OpenAPI Parser
- **Input**: File path 
- **Output**: Parsed Python data structure (dict/list)
- **Algorithm**: Read file, parse JSON/YAML using standard libraries
- **Test**: Parse both JSON and YAML formats successfully

### 4. Element Extractor
- **Input**: Parsed OpenAPI structure + spec-name
- **Output**: List of elements (info, tags, operations, components) with IDs
- **Algorithm**: Extract each section, generate IDs using `{spec-name}:{element-path}` format
- **Test**: Complete extraction of all elements from sample specs

### 5. Reference Scanner
- **Input**: Element content (Python dict/list)
- **Output**: List of `$ref` strings found in content
- **Algorithm**: Recursive traversal of data structure, collect all `$ref` values
- **Test**: Various nested structures with multiple reference types

### 6. Reference Resolver
- **Input**: `$ref` string + current spec context
- **Output**: Resolved chunk ID
- **Algorithm**: Parse reference format, convert to chunk ID pattern
- **Test**: Internal refs, external file refs, fragment refs

### 7. Graph Builder
- **Input**: List of chunks with references 
- **Output**: Chunks with `ref_ids` hierarchy and `referenced_by` lists
- **Algorithm**: Build bidirectional reference graph, handle circular references
- **Test**: Circular refs, deep nesting, complex dependency trees

### 8. Chunk Assembler
- **Input**: Element content + metadata + references
- **Output**: Complete chunk with YAML document and full metadata
- **Algorithm**: Convert Python objects to YAML using `yaml.dump()`, assemble final structure
- **Test**: Various chunk types with correct formatting

### Key Implementation Details

**Spec-Name Generation**:
- Use full relative path with extension: `apis/payment/v2/openapi.yaml`
- Guarantees uniqueness across all directory structures and file formats

**Chunk ID Format**:
- `{spec-name}:{element-path}`
- Examples:
  - `apis/v1/openapi.yaml:info`
  - `apis/v1/openapi.yaml:paths/payments/post`
  - `apis/v1/openapi.yaml:components/schemas/Payment`

**Reference Conversion**:
- `#/components/schemas/Product` → `{spec-name}:components/schemas/Product`
- External file references supported for split OpenAPI structures

**YAML Formatting**:
- All content unified to YAML format for consistency and token efficiency
- Use `yaml.dump(data, default_flow_style=False)` for readable output
- 20-30% token savings vs formatted JSON

## Configuration (.env Variables)

```bash
# Required
OPENAPI_SPECS_DIR=/path/to/openapi/specs

# Processing Configuration
SKIP_HIDDEN_FILES=true             # Skip files/directories starting with '.'
SUPPORTED_EXTENSIONS=.json,.yaml,.yml  # File extensions to process
LOG_PROCESSING_PROGRESS=true       # Log progress during processing

# Validation Configuration  
MIN_OPENAPI_VERSION=3.0.0          # Minimum OpenAPI version to accept
REQUIRE_INFO_SECTION=true          # Require info section in specs
REQUIRE_PATHS_OR_COMPONENTS=true   # Require either paths or components

# Output Configuration
YAML_FORMATTING_STYLE=readable     # 'readable' or 'compact' YAML output
PRESERVE_YAML_FORMATTING=true      # Maintain original YAML structure when possible
INCLUDE_EMPTY_SECTIONS=false       # Include sections that exist but are empty
```

## Definition of Done

### Functional Requirements
1. **Process All Sample Specs**: Successfully process all 3 sample OpenAPI files without errors
2. **Generate Valid Chunks**: Produce chunks matching the exact contract structure above
3. **Unique Chunk IDs**: No duplicate chunk IDs across all processed specifications  
4. **Complete Reference Resolution**: All `$ref` dependencies correctly identified and mapped to chunk IDs
5. **Content Preservation**: Generated chunks contain complete, unmodified OpenAPI content in YAML format
6. **Bidirectional References**: Both `ref_ids` hierarchy and `referenced_by` lists correctly populated

### Measurable Success Criteria
1. **Complete Coverage**: Process 100% of elements from sample specs:
   - All info sections, tags, operations, and components become chunks
   - Expected ~250-300 total chunks across all 3 sample files
2. **Reference Accuracy**: 100% of extracted `ref_ids` point to existing chunk IDs in the same batch
3. **Graph Completeness**: Every chunk that contains a `$ref` has corresponding entries in other chunks' `referenced_by` lists
4. **No ID Collisions**: Zero duplicate chunk IDs across all processed files  
5. **YAML Validity**: All generated YAML documents parse correctly and preserve original structure

### Integration Test Scenarios
1. **Batch Processing**: Process all sample specs together, verify no ID collisions or reference conflicts
2. **Reference Graph Validation**: For each `ref_id`, verify corresponding chunk exists and has correct `referenced_by` entry
3. **Content Fidelity**: Compare original OpenAPI sections with generated YAML chunks for exact preservation
4. **Circular Reference Handling**: Verify system handles circular references without infinite loops
5. **API Context Integration**: Verify all chunks correctly reference their API's info and tags
6. **File Format Consistency**: Process both JSON and YAML files, verify identical logical output structure

## Implementation Guidelines

### Code Structure
```python
# Modular subcomponent organization
openapi_processor/
├── __init__.py
├── scanner.py         # Directory scanning and file discovery
├── parser.py          # JSON/YAML parsing and validation  
├── extractor.py       # Element extraction from parsed specs
├── reference_scanner.py   # $ref discovery in content
├── reference_resolver.py  # $ref to chunk ID conversion
├── graph_builder.py   # Bidirectional reference graph construction
├── chunk_assembler.py # Final chunk assembly with YAML formatting
└── processor.py       # Main orchestration (5-phase pipeline)
```

### Key Classes
```python
class OpenAPIProcessor:
    def process_directory(self, specs_dir: str) -> List[Dict]:
        """Main entry point - orchestrates 5-phase pipeline"""
        
    def _scan_directory(self, specs_dir: str) -> Iterator[str]:
        """Phase 1: Directory scanning"""
        
    def _process_file(self, file_path: str) -> List[Dict]:
        """Phase 2-5: File processing through chunk generation"""

class DirectoryScanner:
    def scan_for_openapi_files(self, root_dir: str) -> Iterator[str]:
        """Recursive scan for .json/.yaml/.yml files"""

class ElementExtractor:
    def extract_elements(self, spec: dict, spec_name: str) -> List[Tuple[str, dict]]:
        """Extract info, tags, operations, components with IDs"""

class ReferenceResolver:
    def resolve_ref_to_chunk_id(self, ref: str, spec_name: str) -> str:
        """Convert $ref to chunk ID format"""

class GraphBuilder:
    def build_reference_graph(self, chunks: List[Dict]) -> List[Dict]:
        """Build hierarchical ref_ids and referenced_by lists"""

class ChunkAssembler:
    def assemble_chunk(self, element_id: str, content: dict, metadata: dict) -> Dict:
        """Create final chunk with YAML document and complete metadata"""
```

### Error Handling Strategy
- **File Access Errors**: Log and skip inaccessible files
- **Parse Errors**: Log parsing failures, continue with remaining files
- **Invalid OpenAPI**: Log validation errors, skip invalid specs
- **Broken References**: Log warnings for invalid `$ref`, include chunk anyway
- **Circular References**: Detect cycles, limit traversal depth, continue processing
- **Memory Limits**: Process files individually, release memory after each file

### Performance Optimizations
- **Single-Pass Processing**: Process each file once through complete pipeline
- **Memory Management**: Release file content after chunk generation
- **Progress Logging**: Track and report processing progress for large directories
- **Efficient Reference Resolution**: Build reference maps once, reuse for graph construction
- **YAML Generation**: Use efficient YAML formatting only in final assembly step

## Integration Points

### Upstream Dependencies
- **OpenAPI Specs**: Reads from configurable directory
- **Configuration**: Uses .env settings for behavior control

### Downstream Dependencies  
- **Vector Store Manager**: Provides chunks for embedding and storage
- **Evaluation Framework**: Uses sample chunks for testing

### Data Contract Validation
```python
# Validate output chunks match expected contract
def validate_chunk(chunk: Dict) -> bool:
    required_fields = ["id", "document", "metadata"]
    metadata_required = ["type", "source_file", "ref_ids", "natural_name"]
    
    # Validate structure
    # Validate ID format: "{filename}:{name}"
    # Validate ref_ids point to valid chunk IDs
    # Validate metadata completeness

def validate_content_fidelity(chunk: Dict, original_spec: Dict) -> bool:
    """Validate chunk preserves original OpenAPI content exactly"""
    
    # For endpoint chunks: verify operation content is complete
    if chunk["metadata"]["type"] == "endpoint":
        # Check that all operation fields are present
        # Verify parameter descriptions match exactly
        # Confirm response definitions are complete
        # Validate examples are preserved verbatim
        
    # For schema chunks: verify schema content is complete  
    elif chunk["metadata"]["type"] == "schema":
        # Check that all schema properties are present
        # Verify constraints (min/max, patterns) are preserved
        # Confirm enum values and descriptions match exactly
        # Validate nested structure is maintained
        
    # Critical validations:
    # - No text summarization or paraphrasing
    # - All constraint values preserved (numbers, strings, patterns)
    # - Complete enum lists with exact values
    # - All required/optional designations preserved
    # - Examples included verbatim from original spec
```

### Content Fidelity Requirements

**Validation Checklist for Each Chunk**:

1. **Complete Field Coverage**: Every field from original OpenAPI section included
2. **Exact Text Preservation**: Descriptions, examples, and constraints copied verbatim  
3. **Structure Integrity**: Nested objects and arrays maintain original hierarchy
4. **Constraint Accuracy**: Min/max values, patterns, formats exactly as specified
5. **Enum Completeness**: All enumeration values present with original descriptions
6. **Metadata Preservation**: Required/optional flags, deprecated status, custom extensions
7. **Example Fidelity**: Code examples and sample values included unchanged
8. **Reference Accuracy**: All `$ref` pointers correctly identified and mapped to chunk IDs

**Quality Gates**:
- Random sample validation: 10% of chunks manually verified for content fidelity
- Automated checks: Validate chunk token count matches expected complexity
- Round-trip testing: Ensure chunk content enables complete API understanding
- No information loss: Every piece of OpenAPI data preserved somewhere in chunk set

## Testing Requirements

### Unit Tests
- Test ID generation with various filename formats
- Test reference extraction from sample operation definitions
- Test document generation for different chunk types
- Test error handling for malformed specs

### Integration Tests  
- Process complete sample OpenAPI files
- Verify chunk count and coverage
- Validate reference resolution accuracy
- Test with mixed JSON/YAML directory

### Performance Tests
- Measure processing time for large specs (>1MB)
- Memory usage during batch processing
- Validate output chunk token sizes

This specification provides everything needed to implement the OpenAPI Processor component using the validated chunking strategy and design decisions from the prototype work.