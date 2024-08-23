# Towards Intelligent Causal Graph Discovery Leveraging LLM and MCTS

We propose an intelligent causal graph discovery method that combines LLMs and Monte Carlo Tree Search (MCTS). Our approach consists of knowledge-guided search, data-proven pruning, and reward-based evaluation.

This repository contains the implementation and experimental code for our proposed method.

## Requirements
python==3.11  
causal-learn==0.1.3.7  
matplotlib==3.8.0  
networkx==3.1  
numpy==1.26.0  
requests==2.31.0  
openai==1.25.0  
pandas==2.1.1  
scipy==1.11.3  
dowhy==0.11  

## Running the Experiments
To run the experiments, execute the file `exp/run_exp.py`. The parameters for experiments on each dataset can be adjusted in the corresponding `exp/exp_*.py` files.

If you wish to run the LLM-based method (i.e., the `exp_bcgllm` function), you need to modify the LLM `api_key` and `base_url` in `bcgllm/exp.py` as follows:
```python
# bcgllm/exp.py
api_key = 'sk-******'
base_url = 'https://api.openai.com/v1'
client = GPTClient(api_key, base_url)
```

## Directory Structure
- `bcgllm`: Contains the implementation and experiment code for the LLM-based causal discovery method.
  - `ccg.py`: Constructs the causal graph using LLM.
  - `exp.py`: Contains experiment code.
  - `llm.py`: LLM client.
  - `*utils.py`: Utility functions.
- `data`: Contains the datasets used in the experiments and code for reading the data.
  - `dataset`: Preprocessed datasets.
  - `origin`: Original datasets.
  - `utils`: Code for preprocessing data.
  - `*bcgllm.json`: Causal priors extracted from LLM, i.e., $G_{LLM}$.
  - `data_*.py`: Code for reading datasets.
- `draw`：Code for generating the charts and graphs used in the paper.
- `exp`：Contains code for running the experiments.
  - `exp.py`：Main experiment code.
  - `exp_*.py`：Code for experiments on different datasets.
  - `run_exp.py`：Entry point for running experiments.
- `mcts`：Contains the implementation of our search method.
  - `calculate_utils.py`：Computational tasks during the search, including causal graph evaluation and conditional independence tests.
  - `graph_utils.py`： Operations related to graph structures, such as checking for cycles.
  - `mcts.py`： Our search method, based on the MCTS framework.
  - `my_utils.py`：Utility functions.
  - `state.py`：Defines the states and related operations within the search process.
- `result`：Stores the experimental results.