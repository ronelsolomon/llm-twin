# LLM Twin Evaluation Framework

This directory contains comprehensive evaluation tools for your LLM twin using **MMLU-Pro** and **LMSYS Chatbot Arena** style evaluations.

## 🎯 Evaluation Methods

### 1. MMLU-Pro Evaluation
- **Dataset**: 12,000+ challenging questions across 14 subjects
- **Format**: 10-choice multiple choice (vs 4 in original MMLU)
- **Focus**: Expert-level knowledge and reasoning
- **Subjects**: Biology, Business, Chemistry, Computer Science, Economics, Engineering, Health, History, Law, Math, Philosophy, Physics, Psychology, Others

### 2. Chatbot Arena Evaluation
- **Method**: Pairwise model battles with Elo rating system
- **Comparison**: Side-by-side model responses
- **Rating**: Chess-style Elo ratings for relative performance
- **Metrics**: Win rates, battle statistics, model rankings

## 📁 File Structure

```
evaluation/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── mmlu_evaluator.py           # MMLU-Pro evaluation script
├── chatbot_arena_evaluator.py  # Chatbot Arena evaluation script
├── run_evaluation.py           # Main evaluation runner
├── examples/                   # Example usage scripts
└── results/                    # Evaluation results output
```

## 🚀 Quick Start

### Install Dependencies
```bash
pip install -r evaluation/requirements.txt
```

### Run MMLU-Pro Evaluation
```bash
python evaluation/run_evaluation.py \
    --eval-type mmlu \
    --mmlu-model /path/to/your/dpo_llm_twin \
    --output evaluation_results \
    --mmlu-samples 1000  # Optional: limit samples
```

### Run Chatbot Arena Evaluation
```bash
python evaluation/run_evaluation.py \
    --eval-type arena \
    --arena-models /path/to/model1 /path/to/model2 \
    --arena-names "Base Model" "DPO Model" \
    --arena-rounds 2 \
    --output evaluation_results
```

### Run Both Evaluations
```bash
python evaluation/run_evaluation.py \
    --eval-type both \
    --mmlu-model /path/to/your/dpo_llm_twin \
    --arena-models /path/to/base/model /path/to/dpo/model \
    --arena-names "Base" "DPO" \
    --output evaluation_results
```

## 📊 Understanding Results

### MMLU-Pro Results
- **Overall Accuracy**: Percentage of correct answers across all subjects
- **Subject Accuracies**: Performance breakdown by subject area
- **Detailed Results**: Individual question responses and analysis

### Chatbot Arena Results
- **Elo Ratings**: Relative skill scores (1400 = average)
- **Win Rates**: Percentage of battles won
- **Battle History**: Detailed pairwise comparison results

## 🎯 Benchmarking Your Model

### Expected Performance Ranges
- **Random Guessing**: ~10% (10 choices)
- **Good Performance**: 50-70% accuracy
- **Excellent Performance**: 70%+ accuracy

### Elo Rating Interpretation
- **< 1200**: Below average performance
- **1200-1500**: Average performance
- **1500-1700**: Good performance
- **> 1700**: Excellent performance

## 🔧 Advanced Usage

### Custom Test Prompts for Arena
Create a file with one prompt per line:
```bash
echo "Explain quantum computing simply" > custom_prompts.txt
echo "Write a Python function for sorting" >> custom_prompts.txt

python evaluation/run_evaluation.py \
    --eval-type arena \
    --arena-models model1 model2 \
    --prompts custom_prompts.txt
```

### Individual Script Usage

#### MMLU Evaluation Only
```bash
python evaluation/mmlu_evaluator.py \
    --model /path/to/model \
    --output mmlu_results \
    --samples 500
```

#### Arena Evaluation Only
```bash
python evaluation/chatbot_arena_evaluator.py \
    --models /path/to/model1 /path/to/model2 \
    --names "Model A" "Model B" \
    --rounds 3 \
    --output arena_results
```

## 📈 Evaluation Reports

The framework generates several output files:

### Combined Report (`combined_evaluation_report.json`)
- Complete evaluation results in JSON format
- MMLU accuracy scores and subject breakdowns
- Arena Elo ratings and battle statistics

### Markdown Report (`evaluation_report.md`)
- Human-readable summary
- Tables and charts for easy interpretation
- Performance insights and recommendations

### Detailed Results
- `mmlu_pro_results.json`: Raw MMLU evaluation data
- `arena_results.json`: Complete arena battle history
- `leaderboard.csv`: Model ranking summary

## 🔍 Analysis Features

### Subject-wise Analysis
```python
# Load MMLU results
with open('evaluation_results/mmlu/mmlu_pro_results.json', 'r') as f:
    results = json.load(f)

# Find strongest/weakest subjects
subjects = results['subject_accuracies']
strongest = max(subjects.items(), key=lambda x: x[1])
weakest = min(subjects.items(), key=lambda x: x[1])

print(f"Strongest: {strongest[0]} ({strongest[1]:.2%})")
print(f"Weakest: {weakest[0]} ({weakest[1]:.2%})")
```

### Battle Analysis
```python
# Load arena results
with open('evaluation_results/arena/arena_results.json', 'r') as f:
    arena = json.load(f)

# Analyze pairwise performance
for battle in arena['battle_results']:
    if battle['winner'] != 'tie':
        print(f"{battle['model_a']} vs {battle['model_b']}: {battle['winner']} wins")
```

## 🎯 Tips for Good Evaluation

### MMLU-Pro Best Practices
1. **Use Chain of Thought**: Models perform better with reasoning steps
2. **Test Multiple Samples**: More questions = more reliable scores
3. **Subject Balance**: Ensure coverage across different domains

### Chatbot Arena Best Practices
1. **Diverse Prompts**: Test various task types and difficulties
2. **Multiple Rounds**: More battles = more stable Elo ratings
3. **Fair Comparison**: Use consistent prompting across models

## 🐛 Troubleshooting

### Common Issues
- **CUDA Memory**: Reduce model size or use CPU evaluation
- **Slow Evaluation**: Limit samples or use fewer arena rounds
- **Loading Errors**: Check model paths and dependencies

### Performance Optimization
```bash
# For faster MMLU evaluation
python evaluation/mmlu_evaluator.py --model model_path --samples 100

# For quicker arena evaluation
python evaluation/chatbot_arena_evaluator.py --models model1 model2 --rounds 1
```

## 📚 References

- [MMLU-Pro Paper](https://arxiv.org/abs/2403.07690)
- [Chatbot Arena Paper](https://arxiv.org/abs/2403.04132)
- [Elo Rating System](https://en.wikipedia.org/wiki/Elo_rating_system)

## 🤝 Contributing

To add new evaluation metrics:
1. Create evaluator class following existing patterns
2. Add to `run_evaluation.py` main script
3. Update documentation

---

**Note**: These evaluations provide complementary insights:
- **MMLU-Pro**: Absolute knowledge and reasoning capability
- **Chatbot Arena**: Relative performance in practical tasks
