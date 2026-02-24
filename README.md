# AI Text Sanitization Study

Empirical study measuring how LLM rewriting changes the linguistic properties of creative writing across eight dimensions.

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
# Full pipeline
python run_pipeline.py

# Individual stages
python run_pipeline.py --stage sample    # 1. Sample stories
python run_pipeline.py --stage rewrite   # 2. LLM rewrites (needs API keys)
python run_pipeline.py --stage analyze   # 3. Compute linguistic markers
python run_pipeline.py --stage compare   # 4. Statistical tests & plots
```

## Linguistic Markers (8 Dimensions)

| Dimension | Markers |
|-----------|---------|
| Lexical diversity | TTR, root TTR, MTLD, hapax ratio |
| Syntactic complexity | Sentence length (mean/std/range), subordination ratio, parse depth |
| Semantic distance | Consecutive & pairwise sentence embedding distances, semantic spread |
| Textual entropy | Word, character, and bigram Shannon entropy |
| Sentiment & affect | VADER compound (mean/std/range), % positive/negative sentences |
| Discourse cohesion | Local & global sentence similarity, connective density |
| Stylometric features | POS ratios, function word ratio, punctuation, word length |
| Readability | Flesch ease, Flesch-Kincaid, Gunning Fog, Coleman-Liau, ARI |

## Output

- `results/comparison_<model>.csv` — per-marker paired Wilcoxon tests + Cohen's d
- `results/summary_<model>.csv` — dimension-level aggregation
- `results/*.png` — effect size plots, distribution comparisons
