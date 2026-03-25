# Enterprise Scenarios Leaderboard Evaluation

## Overview

The Enterprise Scenarios Leaderboard evaluates LLM performance on six real-world enterprise use cases:

1. **FinanceBench** - Financial questions with retrieved context
2. **Legal Confidentiality** - Legal reasoning from LegalBench  
3. **Writing Prompts** - Creative writing evaluation
4. **Customer Support Dialogue** - Context relevance in customer service
5. **Toxic Prompts** - Safety assessment for harmful content
6. **Enterprise PII** - Business safety for sensitive information protection

## Quick Start

### 1. Compare DPO LLM Twin with Base Model

```bash
cd /Users/ronel/Downloads/llm\ twin/evaluation/examples
python compare_dpo_enterprise.py
```

### 2. Run Enterprise Evaluation via Main Script

```bash
cd /Users/ronel/Downloads/llm\ twin/evaluation

# Compare DPO models with base model
python run_evaluation.py \
  --eval-type enterprise \
  --enterprise-models \
    "Base_Llama-7B:/path/to/base/llama-7b" \
    "DPO_Fine_Tuned:/Users/ronel/Downloads/llm twin/dpo_llm_twin_merged" \
    "DPO_LLM_Twin:/Users/ronel/Downloads/llm twin/dpo_llm_twin_improved_merged" \
  --output enterprise_results
```

### 3. Run Individual Enterprise Scenarios

```bash
# Direct evaluation with specific models
python enterprise_scenarios_evaluator.py \
  --models \
    "DPO_LLM_Twin:/Users/ronel/Downloads/llm twin/dpo_llm_twin_improved_merged" \
    "Base_Model:/path/to/base/model" \
  --output enterprise_comparison
```

## Model Paths

Update these paths in the scripts to match your actual model locations:

- **Base Model**: Path to your base LLaMA model
- **DPO Fine-Tuned**: `/Users/ronel/Downloads/llm twin/dpo_llm_twin_merged`
- **DPO LLM Twin**: `/Users/ronel/Downloads/llm twin/dpo_llm_twin_improved_merged`

## Evaluation Metrics

Each scenario is evaluated on multiple dimensions:

### FinanceBench
- Numerical accuracy
- Context utilization  
- Financial reasoning
- Professional tone

### Legal Confidentiality
- Legal reasoning
- Ethical awareness
- Risk assessment
- Professional responsibility

### Writing Prompts
- Length appropriateness
- Tone matching
- Writing quality
- Purpose achievement

### Customer Support
- Empathy
- Actionability
- Clarity
- Resolution focus

### Toxic Prompts
- Refusal compliance
- Safety explanation
- Helpful alternatives
- No harmful content

### Enterprise PII
- PII protection
- Privacy awareness
- Appropriate response
- Professional handling

## Results

The evaluation generates:

1. **JSON Results**: Detailed scores and responses
2. **Console Summary**: Real-time performance comparison
3. **Enterprise Readiness Assessment**: Deployment recommendations
4. **Scenario Analysis**: Strengths and weaknesses by use case

## Enterprise Readiness Levels

- 🟢 **Enterprise Ready** (≥0.8): Suitable for production
- 🟡 **Needs Improvement** (0.6-0.8): Requires refinement
- 🔴 **Not Ready** (<0.6): Significant improvements needed

## Interpreting Results

### Overall Score
- Composite performance across all 6 scenarios
- Higher scores indicate better enterprise suitability

### Scenario Breakdown
- Identify specific strengths and weaknesses
- Guide targeted improvements

### DPO vs Base Comparison
- Measure improvement from fine-tuning
- Quantify personalization benefits

## Troubleshooting

### Model Loading Issues
```bash
# Check model paths exist
ls -la "/Users/ronel/Downloads/llm twin/dpo_llm_twin_improved_merged"
```

### Memory Issues
```bash
# Use CPU instead of GPU
python enterprise_scenarios_evaluator.py --device cpu --models ...
```

### Missing Dependencies
```bash
# Install requirements
pip install -r evaluation/requirements.txt
```

## Next Steps

1. **Run Evaluation**: Start with the comparison script
2. **Analyze Results**: Review scenario-specific performance
3. **Target Improvements**: Focus on weaker areas
4. **Deploy**: Use models scoring ≥0.8 for enterprise applications

## Integration with Existing Evaluations

The Enterprise Scenarios evaluation integrates with your existing evaluation framework:

- Combines with MMLU, HellaSwag, and other benchmarks
- Supports unified reporting
- Maintains consistent model loading and evaluation patterns
