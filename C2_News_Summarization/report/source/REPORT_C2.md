# Task C2: Extreme Abstractive News Summarisation Using Sequence-to-Sequence and Pretrained Transformers
**AI-101 Machine Learning Assignment — Pool C: Sequence-to-Sequence Modality**  
**The University of Lahore — Department of Intelligent Systems**  
**Student:** Iman Fatima | **Roll No.:** 70177713 | **Section:** Section C  
**Instructor:** Dr. Muhammad Awais, AI Architect  

---

## 1. Introduction
Information overload in digital journalism demands automated summarisation systems that distill lengthy articles into telegraphic overviews. Traditional extractive methods fail on modern news curation because professional editors write synthetic headlines that rephrase core propositions rather than copying opening sentences. In this investigation, we study sequence-to-sequence models on the Extreme Summarisation (XSum) benchmark. Our primary finding is that pretrained encoder-decoder transformers (T5-small) generate superior abstractive summaries (ROUGE-1 = 34.82, ROUGE-L = 28.15), outperforming the standard Lead-1 baseline (ROUGE-1 = 16.07) by over 18.7 ROUGE points and overcoming the severe repetition traps of recurrent attention models.

---

## 2. Problem Statement
We frame abstractive summarisation as a sequence-to-sequence conditional generation task. Given an article token sequence $\mathbf{x} = (x_1, \dots, x_M)$, the goal is to generate a target summary sequence $\mathbf{y} = (y_1, \dots, y_N)$ that maximizes conditional log-likelihood $\sum_{t=1}^N \log P(y_t \mid y_{<t}, \mathbf{x})$, where $N \ll M$. Success is measured via standard ROUGE F1 metrics (ROUGE-1, ROUGE-2, ROUGE-L). Constraints include extreme compression ($\approx$14:1 ratio), source sequence truncation, exposure bias during autoregressive decoding, and the hazard of generating unfaithful hallucinations.

---

## 3. Dataset Used
We utilize the BBC Extreme Summarisation (XSum) benchmark curated by Narayan et al. (2018). The full corpus comprises 226,711 British Broadcasting Corporation (BBC) news articles paired with professional one-sentence summaries written by original journalists (split into 204k train, 11.3k validation, and 11.3k test). In this study, we sample a representative 20,000 train, 1,000 validation, and 1,000 test subset with fixed seed 42. Summaries are strictly abstractive and telegraphic.

---

## 4. Dataset Statistics
Table 1 provides the descriptive length and compression statistics. Source articles average 367.4 words (median 299, 95th percentile 925.1 words), whereas target summaries average 21.4 words (median 21.0 words), reflecting a median compression ratio of 0.0725 (13.8$\times$ compression).

### Table 1: XSum Dataset Word Length Statistics
| Sequence | Mean Words | Median Words | Std Dev | 95th Percentile | Min Words | Max Words |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Source Article** ($\mathbf{x}$) | 367.40 | 299.00 | 280.36 | 925.10 | 11.00 | 2187.00 |
| **Target Summary** ($\mathbf{y}$) | 21.44 | 21.00 | 5.33 | 30.00 | 3.00 | 48.00 |
| **Lead-1 Sentence** | 21.82 | 20.00 | 8.92 | 39.00 | 4.00 | 62.00 |

Crucially, n-gram novelty analysis reveals that 38.09% of summary unigrams, 82.17% of bigrams, and 94.48% of trigrams do not occur in the source text. This extreme lexical divergence proves that XSum cannot be solved by extractive copying.

*Figures saved: `figures/fig1_lengths.png` and `figures/fig2_novel_ngrams.png`.*

---

## 5. Methodology

### 5.1 Learning Setting
We examine three distinct summarisation frameworks:
1. **Lead-1 Baseline:** An extractive heuristic taking the opening sentence of the article as the summary.
2. **Model 1 (GRU Seq2Seq with Attention):** A recurrent neural network trained from scratch. The encoder is a bidirectional GRU ($d_{\text{hidden}}=256, d_{\text{embed}}=128$); the decoder is an autoregressive GRU with Bahdanau additive attention, teacher forcing (ratio 0.5), Adam optimizer ($10^{-3}$), and early stopping on validation cross-entropy loss. Source articles are truncated to 400 tokens; summaries to 40 tokens.
3. **Model 2 (Pretrained Transformer):** T5-small (60M parameters), pre-trained on C4 and fine-tuned on XSum. Autoregressive inference uses beam search ($k=4$, length penalty 0.6, and `no_repeat_ngram_size=3`).

### 5.2 Parameter Settings
Table 2 details the architectural parameters across evaluated models.

### Table 2: Sequence-to-Sequence Model Parameters
| Configuration | Model 1 (GRU Seq2Seq) | Model 2 (T5-small) |
| :--- | :--- | :--- |
| **Architecture** | BiGRU + Bahdanau Attention | Encoder-Decoder Transformer |
| **Hidden / Embedding** | 256 / 128 | $d_{\text{model}}=512, d_{\text{ff}}=2048$ |
| **Max Source / Target** | 400 tokens / 40 tokens | 512 tokens / 64 tokens |
| **Optimization** | Adam ($10^{-3}$), early stop=7 | AdamW ($5\times 10^{-5}$), linear decay |
| **Decoding** | Greedy / Beam ($k=4$) | Beam Search ($k=4, \text{no\_repeat}=3$) |

---

## 6. Results and Analysis

### 6.1 Results
Table 3 summarizes the test-set performance across the three systems.

### Table 3: ROUGE F1 Scores on Held-Out XSum Test Data
| System | ROUGE-1 | ROUGE-2 | ROUGE-L | Avg. Words |
| :--- | :---: | :---: | :---: | :---: |
| **Lead-1 Baseline** | 16.07 | 1.57 | 12.10 | 21.82 |
| **Model 1: GRU Seq2Seq + Attention** | 15.30 | 1.30 | 11.79 | 18.45 |
| **Model 2: T5-small (Pretrained)** | **34.82** | **13.41** | **28.15** | 19.40 |

*Diagnostic figures saved: `figures/fig3_loss_curves.png` and `figures/fig4_rouge_comparison.png`.*

### 6.2 Analysis

#### 1. Why Lead-1 Under-performs on XSum
On extractive corpora such as CNN/DailyMail, journalists employ the inverted pyramid structure, causing Lead-3 to achieve high ROUGE-1 ($>40$). In contrast, Lead-1 on XSum achieves a dismal ROUGE-2 of only 1.57. Because 82.2% of summary bigrams are novel, verbatim sentence copying fails to capture the high-level abstractive synthesis formulated by human editors.

#### 2. Factual Hallucination in Model 2 Summaries
We conducted a manual factual audit across 20 generated T5-small summaries:
- 16 summaries (80%) were strictly faithful to the source text.
- 3 summaries (15%) exhibited intrinsic hallucinations (swapping entities or misattributing quotes between individuals mentioned in the article).
- 1 summary (5%) exhibited extrinsic hallucination (generating an ungrounded temporal qualifier due to language model priors).
Hallucinations arise because standard cross-entropy loss optimizes lexical token likelihood without explicit factual consistency constraints.

#### 3. Input Truncation and Attention Repetition
Truncating source articles to 400 tokens discards trailing text in 14% of articles. However, journalistic content is front-loaded; ROUGE degradation from 400-token truncation is under 0.8%, while reducing computational complexity by over 50%. In Model 1, unconstrained decoding produces degenerate repetitive loops due to exposure bias. Enforcing `no_repeat_ngram_size=3` in Model 2 completely suppresses repetitive sequences.

---

## 7. Conclusion
Pretrained encoder-decoder transformers (T5-small) vastly surpass extractive heuristics and recurrent models on extreme abstractive summarisation, achieving a ROUGE-1 of 34.82 and ROUGE-L of 28.15. The extreme abstractiveness of XSum renders extractive lead heuristics ineffective. Future work will explore Reinforcement Learning with Human Feedback (RLHF) and fact-checking reward models to eliminate factual hallucinations.

---

## AI Use Disclosure
I used Google DeepMind Antigravity AI assistant for drafting sequence-to-sequence pipeline scaffolding, implementing ROUGE evaluation metrics, and formatting the IEEE conference report. All experiments, metric calculations, qualitative audits, and written interpretations were produced, verified, and checked by me, and I take full responsibility for this work.
