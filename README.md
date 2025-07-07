# LLaVA logit lens analysis

Application of logit lens method to multimodal models by analyzing how the language model component of LLaVA handles conflicting information.

## Report analysis
Task solution and full analysis are available [in this file](./report.pdf)

## Setup

Experiments were conducted on NVIDIA A100 (40GB).

```bash
conda create --name llava-lens python==3.10.8
conda activate llava-lens
pip install -r requirements.txt
```

Add jupyter kernel and select it when running jupyter notebooks:
```bash
python -m ipykernel install --user --name llava-lens --display-name "llava-lens"
```

## Project Structure
├── src/
│   ├── llava.py                        # Main script
│   ├── attention_generate_tokens.ipynb # Analyze attention patterns for "cat" and "dog" tokens
│   ├── generate_tokens.ipynb           # Predict tokens during generation
│   ├── probability.ipynb               # Compute probabilities of "cat" and "dog" tokens
│   ├── visualize_attention.ipynb       # Visualize attention patterns
│   └── analysis_results/               # Generated output files (after running scripts)
├── report.pdf                          # Full analysis report
├── requirements.txt                    # Python dependencies
└── README.md                           # This file

## Run scripts

[llava.py](./src/llava.py) - create llava with hooks and compute probabilities for "cat" and "dog" tokens.
```bash
CUDA_VISIBLE_DEVICES=$DEVICE python llava.py
```
$DEVICE - CUDA device number.

All output files will be created in ./src/analysis_results.