## Task C2: Extreme Summarisation (XSum) Dataset

**Source:** EdinburghNLP/xsum (HuggingFace Hub)  
**URL:** `https://huggingface.co/datasets/EdinburghNLP/xsum`  
**Paper:** S. Narayan, S. B. Cohen, and M. Lapata, "Don't Give Me the Details, Just the Summary! Topic-Conditioned Abstractive Summarization," EMNLP 2018.

### Dataset Overview
- **Total BBC Articles:** 226,711 (204k Train / 11.3k Validation / 11.3k Test)
- **Task:** Abstractive sequence-to-sequence text generation (full news document $\to$ single-sentence professional journalist summary).
- **Abstractiveness:** Highly abstractive; reference summaries rarely copy full sentences from the source document, featuring high rates of novel unigrams and bigrams.

### Accessing the Full Dataset
To download the full 226k dataset:
```python
from datasets import load_dataset
dataset = load_dataset("EdinburghNLP/xsum")
```

A local sample of 500 verified real articles is cached in `xsum_sample.json` for reproducible evaluation and headless testing. Per assignment requirements, large dataset caches are excluded from Git via `.gitignore`.

