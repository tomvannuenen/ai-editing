# AI Text Sanitization Study

Empirical study measuring how LLM "improvement" rewrites systematically alter the linguistic properties of creative writing. The core hypothesis is that when users ask an LLM to "make this better," the model produces text that is more uniform, more readable, and less linguistically diverse — a flattening effect we call **text sanitization**.

## Motivation

LLMs are increasingly used as writing assistants for creative work. But what exactly changes when a model rewrites a piece of fiction? This project quantifies those changes across eight linguistic dimensions, using paired statistical tests to identify which properties shift significantly and by how much.

## Study Design

1. **Corpus**: 300 creative writing samples from the [WritingPrompts](https://www.kaggle.com/datasets/ratthachat/writing-prompts) dataset (Reddit r/WritingPrompts), filtered to 150–2000 words.
2. **Rewriting**: Each story is sent to two LLMs with the prompt *"Please improve and rewrite the following creative writing piece."* Models:
   - **GPT-5.2 Instant** (`gpt-5.2-chat-latest`) — OpenAI's current flagship conversational model
   - **Claude Sonnet 4.6** (`claude-sonnet-4-6-20250220`) — Anthropic's balanced workhorse model
3. **Measurement**: ~35 linguistic markers computed across 8 dimensions for both originals and rewrites.
4. **Analysis**: Paired Wilcoxon signed-rank tests with Benjamini-Hochberg FDR correction, Cohen's d effect sizes, and cross-model comparison.

## Project Structure

```
run_pipeline.py              # CLI entry point — runs all 4 stages
requirements.txt             # Python dependencies
src/
  config.py                  # Paths, sampling parameters, model config
  data_loader.py             # Load WritingPrompts, filter by length, sample 300 stories
  rewriter.py                # LLM rewriting — batch API (50% off) and sync modes
  markers.py                 # Compute 8 dimensions of linguistic markers
  stats.py                   # Paired Wilcoxon tests, Cohen's d, FDR correction
  visualize.py               # Effect size plots, dimension summaries, KDE overlays
```

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Download the [WritingPrompts dataset](https://www.kaggle.com/datasets/ratthachat/writing-prompts) and unzip into `data/raw/writingPrompts/`.

Set API keys:
```bash
export OPENAI_API_KEY="..."
export ANTHROPIC_API_KEY="..."
```

## Running

```bash
# Full pipeline (batch mode, ~$5-6 total API cost)
python run_pipeline.py

# Individual stages
python run_pipeline.py --stage sample                   # 1. Sample 300 stories
python run_pipeline.py --stage rewrite                  # 2. Batch API rewrites (default, 50% off)
python run_pipeline.py --stage rewrite --mode sync      # 2. One-at-a-time (full price, for testing)
python run_pipeline.py --stage analyze                  # 3. Compute linguistic markers
python run_pipeline.py --stage compare                  # 4. Statistical tests & plots
```

Both the rewrite stage (batch metadata + checkpoint files) and the analyze stage (per-column marker files) support **resumption** — if interrupted, re-running the same command picks up where it left off.

## Linguistic Markers (8 Dimensions)

| # | Dimension | Markers | Source |
|---|-----------|---------|--------|
| 1 | **Lexical diversity** | TTR, root TTR, MTLD (bidirectional), hapax ratio | spaCy tokenization |
| 2 | **Syntactic complexity** | Sentence length (mean/std/min/max), subordination ratio, mean parse tree depth | spaCy dependency parser |
| 3 | **Semantic distance** | Consecutive sentence cosine distance (mean/std/max), pairwise distance, semantic spread from centroid | sentence-transformers (all-MiniLM-L6-v2) |
| 4 | **Textual entropy** | Shannon entropy at word, character, and bigram levels | Custom (scipy/numpy) |
| 5 | **Sentiment & affect** | VADER compound (mean/std/min/max/range), % positive sentences, % negative sentences | NLTK VADER |
| 6 | **Discourse cohesion** | Adjacent sentence similarity, global pairwise similarity, discourse connective density | sentence-transformers + keyword list |
| 7 | **Stylometric features** | POS tag ratios, function word ratio, punctuation ratio, mean word length, exclamation/question density | spaCy POS tagger |
| 8 | **Readability** | Flesch Reading Ease, Flesch-Kincaid Grade, Gunning Fog, Coleman-Liau, ARI, syllables per word | textstat |

## Statistical Approach

- **Paired design**: Each story serves as its own control (original vs. rewrite), eliminating between-text variance.
- **Wilcoxon signed-rank test**: Non-parametric paired test, no normality assumption required.
- **Benjamini-Hochberg FDR correction**: Controls false discovery rate across the ~35 simultaneous tests.
- **Cohen's d (paired)**: Effect size on the paired differences, to distinguish statistical significance from practical significance.
- **Cross-model comparison**: Effect sizes compared between GPT-5.2 and Claude Sonnet to identify model-specific vs. universal sanitization patterns.

## Output

```
results/
  comparison_gpt52_instant.csv    # Per-marker test results (p-values, effect sizes, direction)
  comparison_claude_sonnet.csv
  summary_gpt52_instant.csv       # Dimension-level aggregation
  summary_claude_sonnet.csv
  cross_model_effects.csv         # Side-by-side Cohen's d for both models
  effect_sizes_*.png              # Horizontal bar charts of Cohen's d per marker
  dimension_summary_*.png         # Significant markers and mean effect by dimension
  distributions_*.png             # KDE overlays for top-6 effect markers
```

## Current Status

- [x] Project scaffolding and config
- [x] Data loader with length filtering and sampling
- [x] LLM rewriter with Batch API support (OpenAI + Anthropic, 50% discount)
- [x] Synchronous rewrite fallback with checkpointing
- [x] 8-dimension linguistic marker computation
- [x] Paired statistical testing with FDR correction
- [x] Visualization (effect sizes, dimension summaries, distribution overlays)
- [x] Cross-model comparison
- [ ] Run the study (pending dataset download + API keys)
- [ ] Interpret results and write up findings

## Cost Estimate

Using the Batch API (50% off) with 300 samples:

| Model | Estimated cost |
|-------|---------------|
| GPT-5.2 Instant | ~$2.50–3.00 |
| Claude Sonnet 4.6 | ~$2.50–3.00 |
| **Total** | **~$5–6** |
