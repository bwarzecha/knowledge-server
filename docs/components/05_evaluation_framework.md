# Evaluation Framework Component Specification

## Component Purpose

Provide automated testing and quality measurement for the knowledge retrieval system. This component implements rigorous evaluation methodology to ensure system reliability, track performance improvements, and prevent regressions during development.

## Core Responsibilities

1. **Component Testing**: Independent validation of each system component
2. **Integration Testing**: End-to-end workflow validation with real data
3. **Performance Measurement**: Track response times, accuracy, and resource usage
4. **Regression Detection**: Automated testing to prevent quality degradation
5. **Quality Metrics**: Measurable assessment of retrieval completeness and accuracy
6. **Test Data Management**: Maintain and organize test cases and expected results

## Input/Output Contracts

### Test Configuration Interface
```python
@dataclass
class EvaluationConfig:
    test_specs_dir: str              # Sample OpenAPI specs for testing
    test_queries_file: str           # Predefined test queries
    output_dir: str                  # Results and reports directory
    
    # Component test settings
    run_component_tests: bool = True
    run_integration_tests: bool = True
    run_performance_tests: bool = True
    
    # Performance targets
    max_retrieval_time_ms: float = 200
    max_total_response_time_ms: float = 5000
    min_retrieval_completeness: float = 0.90
    target_context_tokens: int = 1500
```

### Test Results Structure
```python
@dataclass
class EvaluationResults:
    test_run_id: str                 # Unique identifier for test run
    timestamp: datetime              # When tests were executed
    overall_status: str              # "PASS", "FAIL", "PARTIAL"
    
    component_results: Dict[str, ComponentTestResult]
    integration_results: IntegrationTestResult
    performance_results: PerformanceTestResult
    
    summary: TestSummary

@dataclass
class ComponentTestResult:
    component_name: str
    tests_run: int
    tests_passed: int
    tests_failed: int
    failure_details: List[str]
    execution_time_ms: float

@dataclass
class IntegrationTestResult:
    queries_tested: int
    successful_responses: int
    retrieval_completeness_avg: float
    response_quality_score: float
    failed_queries: List[FailedQuery]

@dataclass
class PerformanceTestResult:
    avg_retrieval_time_ms: float
    avg_total_response_time_ms: float
    memory_usage_mb: float
    throughput_queries_per_second: float
    performance_regression: bool
```

## Key Implementation Details

### Test Query Management
**Comprehensive Coverage**: Test queries covering all API usage patterns from sample specs.

```python
class TestQueryManager:
    def __init__(self, specs_dir: str):
        self.specs_dir = specs_dir
        self.test_queries = self._load_test_queries()
    
    def _load_test_queries(self) -> List[TestQuery]:
        """Load predefined test queries with expected results"""
        
        # Load from JSON file or generate programmatically
        queries = [
            TestQuery(
                id="endpoint_discovery_001",
                query="How do I create a new campaign?",
                expected_chunks=["openapi:createCampaign", "openapi:Campaign"],
                query_type="endpoint_discovery",
                difficulty="basic"
            ),
            TestQuery(
                id="schema_inspection_001", 
                query="What fields are required in the Campaign schema?",
                expected_chunks=["openapi:Campaign"],
                query_type="schema_inspection",
                difficulty="basic"
            ),
            TestQuery(
                id="complex_workflow_001",
                query="How do I create a campaign with custom targeting and budget rules?",
                expected_chunks=["openapi:createCampaign", "openapi:Campaign", "openapi:TargetingCriteria", "openapi:Budget"],
                query_type="complex_workflow", 
                difficulty="advanced"
            )
        ]
        
        return queries

@dataclass
class TestQuery:
    id: str
    query: str
    expected_chunks: List[str]        # Chunk IDs that should be retrieved
    query_type: str                   # Category for analysis
    difficulty: str                   # Complexity level
    min_confidence: str = "medium"    # Minimum expected confidence
    max_response_time_ms: float = 5000
```

### Component Testing Framework
**Independent Validation**: Test each component in isolation with controlled inputs.

```python
class ComponentTester:
    def test_openapi_processor(self, specs_dir: str) -> ComponentTestResult:
        """Test OpenAPI processing with sample specifications"""
        
        start_time = time.time()
        failures = []
        
        try:
            # Test 1: Parse all sample specs without errors
            processor = OpenAPIProcessor()
            chunks = processor.process_directory(specs_dir)
            
            # Test 2: Verify chunk structure and IDs
            self._validate_chunk_structure(chunks, failures)
            
            # Test 3: Check reference resolution
            self._validate_reference_integrity(chunks, failures)
            
            # Test 4: Verify no duplicate IDs
            self._validate_unique_ids(chunks, failures)
            
            # Test 5: Check token distribution
            self._validate_chunk_sizes(chunks, failures)
            
        except Exception as e:
            failures.append(f"OpenAPI Processor failed: {str(e)}")
        
        execution_time = (time.time() - start_time) * 1000
        
        return ComponentTestResult(
            component_name="OpenAPI Processor",
            tests_run=5,
            tests_passed=5 - len(failures),
            tests_failed=len(failures),
            failure_details=failures,
            execution_time_ms=execution_time
        )
    
    def _validate_chunk_structure(self, chunks: List[Dict], failures: List[str]):
        """Validate chunk structure matches specification"""
        required_fields = ["id", "document", "metadata"]
        
        for chunk in chunks:
            if not all(field in chunk for field in required_fields):
                failures.append(f"Chunk {chunk.get('id', 'unknown')} missing required fields")
                
            # Validate ID format
            if ":" not in chunk.get("id", ""):
                failures.append(f"Invalid ID format: {chunk.get('id')}")
                
            # Validate metadata structure
            metadata = chunk.get("metadata", {})
            if "type" not in metadata or metadata["type"] not in ["endpoint", "schema"]:
                failures.append(f"Invalid chunk type: {metadata.get('type')}")
```

### Integration Testing Pipeline
**End-to-End Validation**: Test complete workflow with realistic scenarios.

```python
class IntegrationTester:
    def __init__(self, knowledge_retriever: KnowledgeRetriever, mcp_server: KnowledgeServerMCP):
        self.retriever = knowledge_retriever
        self.mcp_server = mcp_server
    
    async def run_integration_tests(self, test_queries: List[TestQuery]) -> IntegrationTestResult:
        """Run complete workflow tests with predefined queries"""
        
        successful_responses = 0
        completeness_scores = []
        failed_queries = []
        
        for test_query in test_queries:
            try:
                # Test retrieval completeness
                context = await self.retriever.retrieve_knowledge(test_query.query)
                completeness_score = self._calculate_completeness(
                    context, test_query.expected_chunks
                )
                completeness_scores.append(completeness_score)
                
                # Test end-to-end response
                response = await self.mcp_server._handle_api_query(test_query.query)
                
                # Validate response quality
                if self._validate_response_quality(response, test_query):
                    successful_responses += 1
                else:
                    failed_queries.append(FailedQuery(
                        query_id=test_query.id,
                        query=test_query.query,
                        failure_reason="Poor response quality",
                        completeness_score=completeness_score
                    ))
                    
            except Exception as e:
                failed_queries.append(FailedQuery(
                    query_id=test_query.id,
                    query=test_query.query,
                    failure_reason=str(e),
                    completeness_score=0.0
                ))
        
        return IntegrationTestResult(
            queries_tested=len(test_queries),
            successful_responses=successful_responses,
            retrieval_completeness_avg=sum(completeness_scores) / len(completeness_scores),
            response_quality_score=successful_responses / len(test_queries),
            failed_queries=failed_queries
        )
    
    def _calculate_completeness(
        self, 
        context: KnowledgeContext, 
        expected_chunks: List[str]
    ) -> float:
        """Calculate retrieval completeness score"""
        
        retrieved_ids = {chunk.id for chunk in context.primary_chunks + context.referenced_chunks}
        expected_ids = set(expected_chunks)
        
        if not expected_ids:
            return 1.0  # Perfect score if no expectations
            
        found_expected = len(retrieved_ids.intersection(expected_ids))
        return found_expected / len(expected_ids)
```

### Performance Benchmarking
**Measurable Targets**: Track performance against validated prototype benchmarks.

```python
class PerformanceTester:
    def __init__(self):
        self.baseline_metrics = self._load_baseline_metrics()
    
    async def run_performance_tests(
        self, 
        retriever: KnowledgeRetriever,
        test_queries: List[TestQuery]
    ) -> PerformanceTestResult:
        """Comprehensive performance testing"""
        
        retrieval_times = []
        memory_usage = []
        
        # Test retrieval performance
        for query in test_queries:
            start_time = time.time()
            start_memory = self._get_memory_usage()
            
            context = await retriever.retrieve_knowledge(query.query)
            
            retrieval_time = (time.time() - start_time) * 1000
            end_memory = self._get_memory_usage()
            
            retrieval_times.append(retrieval_time)
            memory_usage.append(end_memory - start_memory)
        
        # Calculate metrics
        avg_retrieval_time = sum(retrieval_times) / len(retrieval_times)
        avg_memory_usage = sum(memory_usage) / len(memory_usage)
        
        # Check for regression
        regression = self._detect_performance_regression(
            avg_retrieval_time, 
            self.baseline_metrics.get("avg_retrieval_time_ms", 200)
        )
        
        return PerformanceTestResult(
            avg_retrieval_time_ms=avg_retrieval_time,
            avg_total_response_time_ms=avg_retrieval_time * 1.5,  # Estimate with LLM
            memory_usage_mb=avg_memory_usage,
            throughput_queries_per_second=1000 / avg_retrieval_time,
            performance_regression=regression
        )
    
    def _detect_performance_regression(
        self, 
        current_time: float, 
        baseline_time: float,
        threshold: float = 1.2
    ) -> bool:
        """Detect if performance has regressed significantly"""
        return current_time > baseline_time * threshold
```

### Automated Test Execution
**CI/CD Integration**: Run tests automatically and generate reports.

```python
class EvaluationFramework:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.test_queries = TestQueryManager(config.test_specs_dir).test_queries
    
    async def run_full_evaluation(self) -> EvaluationResults:
        """Execute complete evaluation suite"""
        
        test_run_id = f"eval_{int(time.time())}"
        timestamp = datetime.now()
        
        results = EvaluationResults(
            test_run_id=test_run_id,
            timestamp=timestamp,
            overall_status="RUNNING",
            component_results={},
            integration_results=None,
            performance_results=None,
            summary=None
        )
        
        try:
            # Run component tests
            if self.config.run_component_tests:
                results.component_results = await self._run_component_tests()
            
            # Run integration tests
            if self.config.run_integration_tests:
                results.integration_results = await self._run_integration_tests()
            
            # Run performance tests
            if self.config.run_performance_tests:
                results.performance_results = await self._run_performance_tests()
            
            # Generate summary
            results.summary = self._generate_summary(results)
            results.overall_status = self._determine_overall_status(results)
            
        except Exception as e:
            results.overall_status = "FAIL"
            logger.error(f"Evaluation failed: {str(e)}")
        
        # Save results
        self._save_results(results)
        
        return results
    
    def _determine_overall_status(self, results: EvaluationResults) -> str:
        """Determine overall test status based on all results"""
        
        # Check component tests
        component_failures = sum(
            r.tests_failed for r in results.component_results.values()
        )
        
        # Check integration targets
        integration_success = (
            results.integration_results and
            results.integration_results.retrieval_completeness_avg >= self.config.min_retrieval_completeness
        )
        
        # Check performance targets
        performance_success = (
            results.performance_results and
            results.performance_results.avg_retrieval_time_ms <= self.config.max_retrieval_time_ms and
            not results.performance_results.performance_regression
        )
        
        if component_failures == 0 and integration_success and performance_success:
            return "PASS"
        elif component_failures == 0 and (integration_success or performance_success):
            return "PARTIAL"
        else:
            return "FAIL"
```

## Configuration (.env Variables)

```bash
# Test Configuration
TEST_SPECS_DIR=./samples                    # Directory with sample OpenAPI specs
TEST_QUERIES_FILE=./tests/test_queries.json # Predefined test queries
TEST_OUTPUT_DIR=./test_results              # Results and reports

# Test Execution Control
RUN_COMPONENT_TESTS=true
RUN_INTEGRATION_TESTS=true  
RUN_PERFORMANCE_TESTS=true
PARALLEL_TEST_EXECUTION=false              # Run tests sequentially by default

# Performance Targets (from prototype validation)
MAX_RETRIEVAL_TIME_MS=200                  # Retrieval performance target
MAX_TOTAL_RESPONSE_TIME_MS=5000            # End-to-end response target
MIN_RETRIEVAL_COMPLETENESS=0.90            # Minimum completeness score
TARGET_CONTEXT_TOKENS=1500                 # Expected context size

# Quality Thresholds
MIN_RESPONSE_QUALITY_SCORE=0.80            # Minimum acceptable response quality
PERFORMANCE_REGRESSION_THRESHOLD=1.2       # 20% performance degradation threshold
MEMORY_USAGE_LIMIT_MB=1000                 # Maximum memory usage during tests

# Reporting
GENERATE_HTML_REPORTS=true                 # Generate visual test reports
SAVE_DETAILED_LOGS=true                    # Save detailed execution logs
ENABLE_PERFORMANCE_HISTORY=true           # Track performance over time
```

## Definition of Done

### Functional Requirements
1. **Component Testing**: Independent tests for all 4 core components
2. **Integration Testing**: End-to-end workflow validation with sample data
3. **Performance Testing**: Automated benchmarking against prototype targets
4. **Test Data Management**: Comprehensive test queries covering all use cases
5. **Automated Execution**: CI/CD compatible test runner with clear reporting
6. **Regression Detection**: Automated detection of quality or performance degradation

### Measurable Success Criteria
1. **Test Coverage**: 100% component test coverage with passing tests
2. **Integration Success**: >90% of test queries successfully processed
3. **Performance Compliance**: Meet all performance targets from prototype validation
4. **Regression Detection**: Accurately detect >20% performance degradation
5. **Report Quality**: Clear, actionable test reports with failure details
6. **Automation**: Full test suite runs without manual intervention

### Integration Test Scenarios
1. **Full System**: Test complete pipeline from OpenAPI specs to final responses
2. **Component Isolation**: Verify each component works independently
3. **Error Scenarios**: Test system behavior with invalid inputs and edge cases
4. **Performance Load**: Test system under concurrent load and stress conditions
5. **Data Validation**: Verify test results match expected outcomes
6. **Regression Testing**: Compare results with previous test runs

## Implementation Guidelines

### Code Structure
```python
# Suggested file organization
evaluation/
├── __init__.py
├── framework.py           # Main evaluation orchestration
├── component_tests.py     # Individual component testing
├── integration_tests.py   # End-to-end workflow testing
├── performance_tests.py   # Performance benchmarking
├── test_data/
│   ├── test_queries.json  # Predefined test queries
│   └── baseline_metrics.json # Performance baselines
└── reports/
    ├── report_generator.py # Test report generation
    └── templates/         # HTML report templates
```

### Key Classes
```python
class EvaluationFramework:
    def __init__(self, config: EvaluationConfig):
        """Initialize evaluation framework with configuration"""
        
    async def run_full_evaluation(self) -> EvaluationResults:
        """Execute complete evaluation suite"""

class TestQueryManager:
    def load_test_queries(self) -> List[TestQuery]:
        """Load and manage test queries"""
        
    def generate_queries_from_specs(self, specs_dir: str) -> List[TestQuery]:
        """Generate test queries from OpenAPI specifications"""

class PerformanceBenchmark:
    def run_benchmark(self, component: Any, test_data: Any) -> PerformanceResult:
        """Run performance benchmark for component"""
```

### Testing Strategy
- **Deterministic Tests**: Use fixed test data for reproducible results
- **Realistic Data**: Test with actual sample OpenAPI specifications
- **Edge Case Coverage**: Include error conditions and boundary cases
- **Performance Regression**: Track performance over time
- **Automated Validation**: Minimal manual verification required

### Error Handling
- **Test Failures**: Continue testing even when individual tests fail
- **Missing Data**: Graceful handling of missing test files or queries
- **Resource Limits**: Monitor and limit resource usage during testing
- **Timeout Handling**: Respect test timeout limits
- **Partial Results**: Provide meaningful results even when some tests fail

## Integration Points

### Upstream Dependencies
- **All System Components**: Tests all implemented components
- **Sample Data**: Uses OpenAPI specs from `/samples/` directory
- **Configuration**: Uses .env settings for test behavior

### Downstream Dependencies
- **CI/CD Systems**: Provides test results for automated pipelines
- **Development Tools**: Supports developer workflow and debugging
- **Monitoring**: Feeds test results to monitoring and alerting systems

### External Integrations
```python
# CI/CD integration example
def main():
    config = EvaluationConfig.from_env()
    framework = EvaluationFramework(config)
    results = await framework.run_full_evaluation()
    
    # Exit with appropriate code for CI/CD
    exit_code = 0 if results.overall_status == "PASS" else 1
    sys.exit(exit_code)
```

## Testing Requirements

### Self-Testing
- Test the evaluation framework itself with known good/bad inputs
- Validate test result accuracy and completeness
- Verify report generation and formatting
- Test error handling and edge cases

### Integration Validation
- Verify tests accurately reflect system behavior
- Validate test queries cover all major use cases
- Ensure performance tests correlate with real usage
- Check regression detection sensitivity and accuracy

### Documentation Testing
- Verify all components can be tested according to specifications
- Validate test procedures and expected outcomes
- Ensure test results are interpretable and actionable
- Check that tests support development workflow

This specification provides complete guidance for implementing a comprehensive evaluation framework that ensures system quality and prevents regressions during development.