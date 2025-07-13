# Evaluation Framework Plan for Knowledge Server

## Executive Summary

The current evaluation method using 5 hand-picked queries with arbitrary "expected" results lacks statistical rigor. This plan outlines a comprehensive evaluation framework using modern LLM-assisted evaluation techniques combined with standard Information Retrieval (IR) metrics to provide meaningful, reproducible performance measurements.

## Current State Analysis

### Problems with Existing Evaluation
- **Sample Size**: Only 5 queries (need minimum 30-50 for statistical power)
- **Arbitrary Expected Results**: Hand-picked without systematic methodology
- **Binary Scoring**: Simple hit/miss doesn't capture relevance nuances
- **No Statistical Testing**: Cannot determine if improvements are significant
- **No Ground Truth Process**: No validation of what constitutes "correct" results
- **Meaningless Percentages**: 60%/70% numbers have no statistical basis

### Impact
- Cannot reliably compare system improvements
- No confidence in performance claims
- Difficult to prioritize optimization efforts
- Risk of overfitting to small test set

## LLM-Powered Evaluation Strategy

### Why Large Context LLMs?

**Gemini 2.5 Flash Advantages:**
- **2M Token Context**: Can ingest entire API documentation at once
- **Superior Understanding**: Better context awareness than local models (Gemma)
- **Consistency**: More reliable than human annotators for large-scale evaluation
- **Cost-Effective**: Cheaper than extensive human annotation
- **Scalable**: Can evaluate thousands of query-document pairs

### Two-Stage LLM Approach

#### Stage 1: Query Generation
Use Gemini 2.5 Flash to generate comprehensive test queries by:
1. **Document Analysis**: Ingest entire OpenAPI documentation
2. **Persona-Based Generation**: Simulate different developer types
   - Beginner developers
   - Expert API users
   - Mobile developers
   - Backend developers
   - DevOps engineers
3. **Task-Based Coverage**: Generate queries for all API usage patterns
   - CRUD operations
   - Authentication/authorization
   - Error handling
   - Schema validation
   - Integration patterns

#### Stage 2: Automated Judging
Use Gemini 2.5 Flash as relevance judge with:
1. **Structured Prompts**: Clear criteria for 0-3 relevance scoring
2. **Full Context**: Entire API documentation for accurate judgment
3. **Consistency Validation**: Multiple evaluation runs for reliability
4. **Human Spot-Checking**: Quality assurance on 10-20% of judgments

## Standard IR Metrics Implementation

### Primary Metrics
1. **NDCG@10** (Normalized Discounted Cumulative Gain)
   - Best metric for ranking quality
   - Accounts for position of relevant documents
   - Industry standard for search evaluation

2. **MAP@10** (Mean Average Precision)
   - Balances precision and recall
   - Popular for recommendation systems
   - Good overall system performance indicator

3. **Precision@5**
   - Practical metric for user experience
   - Measures quality of top results
   - Easy to interpret and communicate

4. **MRR** (Mean Reciprocal Rank)
   - Focuses on first relevant result
   - Important for "find specific answer" queries
   - Complements other ranking metrics

### Statistical Validation
1. **Paired t-tests**: Compare system variants statistically
2. **Effect Size (Cohen's d)**: Measure practical significance
3. **Bootstrap Confidence Intervals**: Robust uncertainty estimates
4. **Cross-Validation**: Test stability across query subsets

## Test Dataset Generation

### Target Specifications
- **Size**: 50-100 diverse queries (statistical power + practical constraints)
- **Coverage**: All API endpoints, schemas, and error types
- **Realism**: Queries developers would actually search for
- **Difficulty Range**: Mix of simple and complex information needs

### Generation Process
1. **Endpoint Coverage**: Generate 2-3 queries per API endpoint
2. **Schema Coverage**: Generate queries for each data model
3. **Error Coverage**: Generate queries for each error type
4. **Cross-Cutting Concerns**: Authentication, pagination, rate limits
5. **Human Validation**: Review generated queries for realism

### Query Categories
```
Category                | Count | Examples
------------------------|-------|------------------------------------------
Endpoint Discovery      | 15-20 | "How to create a campaign?"
Schema Information      | 15-20 | "Campaign object properties"
Error Resolution        | 10-15 | "ACCESS_DENIED error meaning"
Authentication          | 5-10  | "API key authentication"
Integration Patterns    | 10-15 | "Bulk operations", "Pagination"
```

## Automated Judge System

### Relevance Scoring Criteria (0-3 Scale)

**Score 3 (Highly Relevant):**
- Directly answers the developer's question
- Provides complete information needed for implementation
- Includes relevant code examples or usage patterns
- Technically accurate and up-to-date

**Score 2 (Relevant):**
- Addresses the developer's question
- Provides most information needed
- May lack some implementation details
- Generally accurate

**Score 1 (Marginally Relevant):**
- Partially addresses the question
- Provides some useful information
- Requires additional research for implementation
- Mostly accurate but incomplete

**Score 0 (Irrelevant):**
- Does not address the developer's question
- Provides incorrect or misleading information
- Not useful for the stated information need

### Judge Prompt Template
```
You are evaluating the relevance of API documentation chunks for developer queries.

CONTEXT: [Full OpenAPI Documentation]

QUERY: [Developer Query]

CHUNK: [Documentation Chunk to Evaluate]

Evaluate on 0-3 scale based on:
1. How completely this chunk answers the developer's question
2. Technical accuracy of the information
3. Practical utility for implementation
4. Presence of relevant examples or usage patterns

Score: [0-3]
Reasoning: [Brief explanation]
```

## Implementation Timeline

### Phase 1: Foundation (Week 1)
- [ ] Set up Gemini 2.5 Flash API integration
- [ ] Implement query generation pipeline
- [ ] Create structured judge prompts
- [ ] Generate initial 50-query test set

### Phase 2: Metrics Implementation (Week 2)
- [ ] Implement NDCG@10, MAP@10, Precision@5, MRR calculations
- [ ] Build statistical testing framework (paired t-tests)
- [ ] Create bootstrap confidence interval estimation
- [ ] Implement evaluation pipeline automation

### Phase 3: Validation (Week 3)
- [ ] Run multi-judge consistency validation
- [ ] Conduct human spot-checking on 20% of judgments
- [ ] Measure inter-judge agreement (Kappa statistic)
- [ ] Refine judge prompts based on validation results

### Phase 4: Production Integration (Week 4)
- [ ] Integrate evaluation into search system testing
- [ ] Create automated A/B testing framework
- [ ] Build evaluation reporting dashboard
- [ ] Document evaluation procedures for future use

## Cost-Benefit Analysis

### LLM Evaluation Costs
**Gemini 2.5 Flash Pricing (estimated):**
- Query Generation: 50 queries × 2M tokens = ~$20
- Judge Evaluation: 50 queries × 10 results × 100K tokens = ~$100
- **Total per evaluation run: ~$120**

### Comparison to Human Annotation
- Human expert: $50-100/hour
- Time per query evaluation: 15-30 minutes
- 50 queries × 10 results × 20 minutes = ~167 hours
- **Human cost: $8,000-16,000**

### ROI Calculation
- **LLM approach**: 99% cost reduction vs human annotation
- **Scalability**: Can evaluate multiple system variants easily
- **Consistency**: No inter-annotator variability
- **Speed**: Complete evaluation in hours vs weeks

## Quality Assurance

### Human Validation Process
1. **Stratified Sampling**: Select 20% of judgments across different query types
2. **Expert Review**: API documentation experts validate LLM judgments
3. **Agreement Measurement**: Calculate human-LLM agreement using Kappa
4. **Threshold**: Require >0.7 agreement for acceptable quality
5. **Prompt Refinement**: Iterate on judge prompts if agreement is low

### Continuous Monitoring
- Track evaluation consistency across runs
- Monitor for prompt degradation over time
- Update evaluation criteria as API documentation evolves
- Maintain human validation pipeline for quality control

## Expected Outcomes

### Immediate Benefits
1. **Statistical Rigor**: Replace arbitrary percentages with meaningful metrics
2. **Comparative Analysis**: Reliably compare system improvements
3. **Optimization Guidance**: Identify specific areas for improvement
4. **Confidence**: Statistical significance testing for all claims

### Long-term Benefits
1. **Automated Testing**: Continuous evaluation of search quality
2. **A/B Testing**: Reliable framework for testing new features
3. **Performance Tracking**: Monitor search quality over time
4. **Research Foundation**: Basis for publication and knowledge sharing

## Success Criteria

### Technical Metrics
- **Inter-judge Agreement**: Kappa > 0.7 between LLM and human judges
- **Evaluation Coverage**: 100% of API endpoints covered in test queries
- **Statistical Power**: Minimum 50 diverse queries for reliable testing
- **Reproducibility**: <5% variance in metrics across evaluation runs

### Business Metrics
- **Cost Efficiency**: >90% cost reduction vs human annotation
- **Speed**: Complete evaluation in <4 hours vs days/weeks
- **Scalability**: Ability to evaluate 10+ system variants monthly
- **Quality**: Human validation confirms LLM judgment accuracy

## Conclusion

This evaluation framework transforms the knowledge server from having ad-hoc performance measurement to industry-standard, statistically rigorous evaluation. By leveraging Gemini 2.5 Flash's large context capabilities, we achieve both scale and accuracy while maintaining cost-effectiveness.

The framework enables:
- **Confident performance claims** backed by statistical testing
- **Reliable system comparison** for optimization decisions  
- **Automated quality monitoring** for continuous improvement
- **Research-grade evaluation** for knowledge sharing and publication

Implementation of this framework is essential for the knowledge server to achieve its goal of 70%+ search relevance and demonstrate measurable improvements over baseline performance.