# Enhanced Prompt Retrieval for Text Classification

## Abstract
This dissertation introduces a novel pair-wise compatibility approach for enhancing prompt retrieval in text classification tasks, focusing on sentiment analysis. We develop a technique that evaluates the synergistic effects of example pairs, going beyond traditional semantic similarity measures. Our approach demonstrates significant improvements in classification accuracy, particularly in achieving high recall rates.

## Key Features
- **Pair-wise compatibility method for prompt retrieval**
- **Comparative analysis with zero-shot, localized k-shot, and Rubin's approach**
- **Enhanced performance in text classification tasks**
- **Insights into optimal context size for in-context learning**

## Dataset
- **Stanford Sentiment Treebank (SST-2)**

## Models Used
- **Sentence Transformer**: `'all-MiniLM-L6-v2'`
- **LLaMA-2**: 7B parameters

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Combination-of-demonstrations.git
   cd Combination-of-demonstrations

2. Install all the dependencies
   ```bash
   pip install -r requirements.txt

## Usage
To run all the experiment:
```bash
python run_all.py
```
This script will execute the baseline, Rubin's approach, and our approach sequentially.
