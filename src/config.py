"""Configuration for the AI text sanitization study."""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

# ── Sampling ───────────────────────────────────────────────────────────
N_SAMPLES = 500               # number of stories to sample from the dataset
MIN_WORD_COUNT = 150          # minimum words to keep a story
MAX_WORD_COUNT = 2000         # maximum words (longer texts get truncated)
RANDOM_SEED = 42

# ── LLM rewriting ─────────────────────────────────────────────────────
# Set these via environment variables: OPENAI_API_KEY, ANTHROPIC_API_KEY
REWRITE_MODELS = [
    {
        "provider": "openai",
        "model": "gpt-5.2-chat-latest",
        "label": "gpt52_instant",
    },
    {
        "provider": "anthropic",
        "model": "claude-sonnet-4-6-20250220",
        "label": "claude_sonnet",
    },
]

REWRITE_PROMPT = (
    "Please improve and rewrite the following creative writing piece. "
    "Make it better-written, more polished, and more engaging, while "
    "preserving the original story, characters, and meaning. "
    "Return only the rewritten text, with no commentary or preamble.\n\n"
    "{text}"
)

REWRITE_MAX_TOKENS = 4096
REWRITE_TEMPERATURE = 0.7     # moderate temperature to reflect typical usage

# ── Linguistic analysis ────────────────────────────────────────────────
SPACY_MODEL = "en_core_web_sm"

# ── Statistical testing ────────────────────────────────────────────────
ALPHA = 0.05                  # significance level
