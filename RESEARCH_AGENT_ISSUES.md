# Research Agent Issues

## Overview

The research agent (`@src/research_agent/`) has critical error handling and context management issues that cause complete failures during normal operation.

## Problems Identified

### 1. Bedrock Timeout Handling

**Issue**: The research agent fails completely when AWS Bedrock calls timeout, providing no graceful degradation or user feedback.

**Current Behavior**:
- Agent hangs indefinitely on timeout
- No timeout configuration available
- No error recovery mechanism
- Complete failure with no useful error message

**Impact**: 
- Complete system unavailability during network issues
- Poor user experience with no feedback
- Unpredictable behavior in production environments

### 2. Context Explosion

**Issue**: The agent's context grows uncontrollably during operation, leading to performance degradation and potential failures.

**Current Behavior**:
- Context accumulates through multiple tool calls
- No context size limits or management
- Performance degrades as context grows
- May exceed model token limits

**Impact**:
- Degraded response quality
- Increased latency
- Potential token limit exceeded errors
- Higher operational costs

## Technical Analysis

### Code Locations

#### Main Research Function
- **File**: `src/research_agent/agent.py:114-136`
- **Function**: `research_api_question()`
- **Issues**: 
  - No timeout handling around `agent.ainvoke()`
  - No error recovery mechanism
  - No context size monitoring

#### Bedrock Model Configuration
- **File**: `src/research_agent/agent.py:22-27`
- **Function**: `create_research_agent()`
- **Issues**:
  - No timeout configuration in `ChatBedrockConverse`
  - No request-level timeout settings
  - No retry mechanism

#### Reranker LLM Calls
- **File**: `src/research_agent/tools.py:485-486`
- **Function**: `_filter_and_expand_chunks()`
- **Issues**:
  - Basic exception handling but no specific timeout handling
  - Falls back without proper error classification

#### Tool Integration Points
- **MCP Server**: `src/mcp_server/server.py:8` - Exposes `research_api_question`
- **CLI Commands**: `src/cli/commands/research.py` - Direct CLI usage
- **Module Imports**: `src/research_agent/__init__.py:3` - Public API

### Current Error Handling

```python
# Current implementation in tools.py:127-142
try:
    filtered_chunks, filtering_stats = await _filter_and_expand_chunks(...)
    chunks = filtered_chunks
    logger.info(f"Applied LLM filtering: {filtering_stats['original_count']} -> {filtering_stats['filtered_count']} chunks")
except Exception as e:
    logger.error(f"LLM filtering failed: {e}. Continuing with original chunks.")
    chunks = chunks[:max_chunks]  # Fallback: just trim to max_chunks
```

**Problems with current approach**:
- Generic exception handling doesn't distinguish timeout vs other errors
- No timeout configuration
- No user feedback about what went wrong
- No retry mechanism

## Impact Assessment

### User Experience
- **Severity**: Critical
- **Frequency**: Intermittent (depends on network conditions)
- **Effect**: Complete feature unavailability

### System Reliability
- **Severity**: High
- **Effect**: Unpredictable behavior in production
- **Cascading**: May affect MCP server and CLI operations

### Operational Costs
- **Severity**: Medium
- **Effect**: Wasted compute resources on hung operations
- **Monitoring**: Difficult to detect and diagnose

## Root Cause Analysis

### 1. Bedrock Timeout Issues
- **Root Cause**: No timeout configuration in AWS Bedrock client
- **Contributing Factors**:
  - No async timeout wrapper around agent invocation
  - No request-level timeout settings
  - No circuit breaker pattern implementation

### 2. Context Explosion
- **Root Cause**: Uncontrolled context accumulation in LangGraph agent
- **Contributing Factors**:
  - No context size limits in agent configuration
  - No context pruning mechanism
  - Verbose prompts with extensive instructions (110 lines)
  - Multiple tool calls with full context retention

## Proposed Solutions

### Option A: Comprehensive Timeout and Error Handling

**Approach**: Add robust timeout handling with graceful degradation

**Implementation**:
1. Add timeout configuration to config system
2. Implement async timeout wrapper around agent calls
3. Add proper error classification and user feedback
4. Implement retry mechanism with exponential backoff

**Pros**:
- Addresses root cause of timeout issues
- Provides better user experience
- Maintains full functionality when possible

**Cons**:
- More complex implementation
- Requires configuration management
- May need testing infrastructure changes

### Option B: Context Size Management

**Approach**: Implement context size limits and pruning

**Implementation**:
1. Add context size monitoring
2. Implement context pruning strategies
3. Add context size limits to agent configuration
4. Optimize prompt structure

**Pros**:
- Prevents context explosion
- Improves performance
- Reduces operational costs

**Cons**:
- May affect response quality
- Requires careful tuning
- Complex context management logic

### Option C: Hybrid Approach

**Approach**: Combine timeout handling with context management

**Implementation**:
1. Implement both timeout and context management
2. Add comprehensive error handling
3. Implement fallback mechanisms
4. Add monitoring and observability

**Pros**:
- Addresses both issues comprehensively
- Most robust solution
- Better long-term maintainability

**Cons**:
- Most complex implementation
- Requires significant testing
- Higher initial development cost

## Implementation Roadmap

### Phase 1: Critical Timeout Handling (Priority: High)
1. Add timeout configuration to `Config` class
2. Implement async timeout wrapper in `research_api_question()`
3. Add proper error messages and user feedback
4. Update MCP server error handling

### Phase 2: Context Management (Priority: Medium)
1. Implement context size monitoring
2. Add context pruning mechanism
3. Optimize prompt structure
4. Add context size limits

### Phase 3: Enhanced Error Handling (Priority: Medium)
1. Implement retry mechanism
2. Add circuit breaker pattern
3. Improve error classification
4. Add comprehensive logging

### Phase 4: Testing and Monitoring (Priority: Low)
1. Add timeout simulation tests
2. Implement context size tests
3. Add performance benchmarks
4. Implement monitoring dashboards

## Configuration Requirements

### New Environment Variables Needed
```bash
# Timeout configuration
RESEARCH_AGENT_LLM_TIMEOUT_MS=60000    # 60 seconds
CHUNK_FILTERING_LLM_TIMEOUT_MS=10000   # 10 seconds

# Context management
RESEARCH_AGENT_MAX_CONTEXT_SIZE=8000   # tokens
RESEARCH_AGENT_CONTEXT_PRUNING_RATIO=0.7

# Error handling
RESEARCH_AGENT_RETRY_ATTEMPTS=3
RESEARCH_AGENT_RETRY_DELAY_MS=1000
```

## Testing Strategy

### Unit Tests
- Mock timeout scenarios
- Test error handling paths
- Verify context size limits
- Test retry mechanisms

### Integration Tests
- Test with real Bedrock timeouts
- Test context accumulation scenarios
- Test MCP server error propagation
- Test CLI error handling

### Performance Tests
- Measure response times with different context sizes
- Test timeout thresholds
- Benchmark context pruning performance
- Load test with multiple concurrent requests

## Monitoring and Observability

### Metrics to Track
- Request timeout frequency
- Context size distribution
- Error type classification
- Response time percentiles
- Retry success rates

### Alerting
- High timeout rate alerts
- Context size threshold alerts
- Error rate spike alerts
- Performance degradation alerts

## Conclusion

The research agent requires immediate attention to address critical timeout and context management issues. The hybrid approach (Option C) is recommended for comprehensive resolution, implemented in phases to balance urgency with thoroughness.

The timeout handling should be prioritized as it directly affects system availability, while context management can be implemented in parallel to address performance and cost concerns.