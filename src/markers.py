"""Compute linguistic markers of creativity across eight dimensions.

Each public function takes a text string and returns a flat dict of metrics.
``compute_all_markers`` runs everything and returns a single merged dict.

Dimensions
----------
1. Lexical diversity
2. Syntactic complexity
3. Figurative language / semantic distance
4. Textual entropy
5. Sentiment & affective range
6. Discourse cohesion
7. Stylometric / POS features
8. Readability
"""

from __future__ import annotations

import math
import re
from collections import Counter
from functools import lru_cache

import numpy as np
import spacy
import textstat

from src.config import SPACY_MODEL

# ── Lazy model loading ─────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _nlp():
    return spacy.load(SPACY_MODEL)


@lru_cache(maxsize=1)
def _sentence_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


@lru_cache(maxsize=1)
def _vader():
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    import nltk
    nltk.download("vader_lexicon", quiet=True)
    return SentimentIntensityAnalyzer()


# ═══════════════════════════════════════════════════════════════════════
# 1. LEXICAL DIVERSITY
# ═══════════════════════════════════════════════════════════════════════

def _mtld_forward(tokens: list[str], threshold: float = 0.72) -> float:
    """One-pass MTLD (left-to-right)."""
    factor_count = 0.0
    factor_start = 0
    for i in range(1, len(tokens) + 1):
        segment = tokens[factor_start:i]
        types = len(set(segment))
        ttr = types / len(segment)
        if ttr <= threshold:
            factor_count += 1
            factor_start = i
    # partial factor for the remaining tail
    if factor_start < len(tokens):
        remaining = tokens[factor_start:]
        types = len(set(remaining))
        ttr = types / len(remaining)
        if len(remaining) > 1 and ttr < 1.0:
            factor_count += (1.0 - ttr) / (1.0 - threshold)
    return len(tokens) / factor_count if factor_count > 0 else len(tokens)


def _mtld(tokens: list[str]) -> float:
    """Bidirectional MTLD."""
    forward = _mtld_forward(tokens)
    backward = _mtld_forward(tokens[::-1])
    return (forward + backward) / 2


def lexical_diversity(text: str) -> dict:
    doc = _nlp()(text)
    tokens = [t.text.lower() for t in doc if t.is_alpha]
    n_tokens = len(tokens)
    if n_tokens == 0:
        return {k: 0.0 for k in [
            "ld_ttr", "ld_root_ttr", "ld_mtld", "ld_hapax_ratio",
            "ld_unique_words", "ld_total_words",
        ]}

    types = set(tokens)
    n_types = len(types)
    freq = Counter(tokens)
    hapax = sum(1 for w, c in freq.items() if c == 1)

    return {
        "ld_ttr": n_types / n_tokens,
        "ld_root_ttr": n_types / math.sqrt(n_tokens),
        "ld_mtld": _mtld(tokens),
        "ld_hapax_ratio": hapax / n_types if n_types else 0.0,
        "ld_unique_words": n_types,
        "ld_total_words": n_tokens,
    }


# ═══════════════════════════════════════════════════════════════════════
# 2. SYNTACTIC COMPLEXITY
# ═══════════════════════════════════════════════════════════════════════

def syntactic_complexity(text: str) -> dict:
    doc = _nlp()(text)
    sents = list(doc.sents)
    if not sents:
        return {k: 0.0 for k in [
            "syn_mean_sent_len", "syn_std_sent_len", "syn_max_sent_len",
            "syn_min_sent_len", "syn_n_sents",
            "syn_subordination_ratio", "syn_mean_depth",
        ]}

    sent_lengths = [len([t for t in s if t.is_alpha]) for s in sents]
    sent_lengths = [l for l in sent_lengths if l > 0]

    # Subordination: count subordinating conjunctions and relative pronouns
    n_clauses = sum(
        1 for t in doc if t.dep_ in ("advcl", "relcl", "csubj", "ccomp", "acl")
    )
    n_main_clauses = len(sents)

    # Tree depth per sentence
    def _tree_depth(token):
        depth = 0
        while token.head != token:
            depth += 1
            token = token.head
        return depth

    depths = [max((_tree_depth(t) for t in s), default=0) for s in sents]

    arr = np.array(sent_lengths) if sent_lengths else np.array([0])
    return {
        "syn_mean_sent_len": float(arr.mean()),
        "syn_std_sent_len": float(arr.std()),
        "syn_max_sent_len": float(arr.max()),
        "syn_min_sent_len": float(arr.min()),
        "syn_n_sents": len(sents),
        "syn_subordination_ratio": n_clauses / n_main_clauses if n_main_clauses else 0.0,
        "syn_mean_depth": float(np.mean(depths)) if depths else 0.0,
    }


# ═══════════════════════════════════════════════════════════════════════
# 3. FIGURATIVE LANGUAGE / SEMANTIC DISTANCE
# ═══════════════════════════════════════════════════════════════════════

def semantic_distance(text: str) -> dict:
    """Measure semantic novelty via consecutive-sentence embedding distances
    and overall semantic spread of the text."""
    model = _sentence_model()
    doc = _nlp()(text)
    sents = [s.text.strip() for s in doc.sents if len(s.text.strip()) > 10]

    if len(sents) < 2:
        return {
            "sd_mean_consecutive": 0.0,
            "sd_std_consecutive": 0.0,
            "sd_max_consecutive": 0.0,
            "sd_mean_pairwise": 0.0,
            "sd_spread": 0.0,
        }

    embeddings = model.encode(sents, show_progress_bar=False)
    # Normalise
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeddings = embeddings / norms

    # Consecutive cosine distances (1 - similarity)
    consec = [
        1 - float(np.dot(embeddings[i], embeddings[i + 1]))
        for i in range(len(embeddings) - 1)
    ]

    # Pairwise distances (sample if too many sentences)
    if len(embeddings) <= 100:
        sim_matrix = embeddings @ embeddings.T
        # Upper triangle, excluding diagonal
        triu_idx = np.triu_indices(len(embeddings), k=1)
        pairwise = 1 - sim_matrix[triu_idx]
    else:
        # Random sample of 2000 pairs
        rng = np.random.default_rng(42)
        idx_a = rng.integers(0, len(embeddings), 2000)
        idx_b = rng.integers(0, len(embeddings), 2000)
        pairwise = [
            1 - float(np.dot(embeddings[a], embeddings[b]))
            for a, b in zip(idx_a, idx_b) if a != b
        ]

    # Spread: std of embeddings from centroid
    centroid = embeddings.mean(axis=0)
    dists_to_centroid = np.linalg.norm(embeddings - centroid, axis=1)

    ca = np.array(consec)
    pa = np.array(pairwise) if len(pairwise) else np.array([0.0])
    return {
        "sd_mean_consecutive": float(ca.mean()),
        "sd_std_consecutive": float(ca.std()),
        "sd_max_consecutive": float(ca.max()),
        "sd_mean_pairwise": float(pa.mean()),
        "sd_spread": float(dists_to_centroid.std()),
    }


# ═══════════════════════════════════════════════════════════════════════
# 4. TEXTUAL ENTROPY
# ═══════════════════════════════════════════════════════════════════════

def _shannon_entropy(sequence: list) -> float:
    """Shannon entropy in bits."""
    n = len(sequence)
    if n == 0:
        return 0.0
    freq = Counter(sequence)
    return -sum((c / n) * math.log2(c / n) for c in freq.values())


def textual_entropy(text: str) -> dict:
    doc = _nlp()(text)
    tokens = [t.text.lower() for t in doc if t.is_alpha]
    chars = [c.lower() for c in text if c.isalpha()]

    # Bigram entropy
    bigrams = [f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens) - 1)]

    return {
        "ent_word": _shannon_entropy(tokens),
        "ent_char": _shannon_entropy(chars),
        "ent_bigram": _shannon_entropy(bigrams),
    }


# ═══════════════════════════════════════════════════════════════════════
# 5. SENTIMENT & AFFECTIVE RANGE
# ═══════════════════════════════════════════════════════════════════════

def sentiment_affect(text: str) -> dict:
    """Per-sentence VADER sentiment → aggregate stats on the distribution."""
    sia = _vader()
    doc = _nlp()(text)
    sents = [s.text.strip() for s in doc.sents if len(s.text.split()) >= 3]

    if not sents:
        return {k: 0.0 for k in [
            "sent_mean_compound", "sent_std_compound",
            "sent_min_compound", "sent_max_compound",
            "sent_range_compound", "sent_pct_positive", "sent_pct_negative",
        ]}

    compounds = [sia.polarity_scores(s)["compound"] for s in sents]
    arr = np.array(compounds)

    return {
        "sent_mean_compound": float(arr.mean()),
        "sent_std_compound": float(arr.std()),
        "sent_min_compound": float(arr.min()),
        "sent_max_compound": float(arr.max()),
        "sent_range_compound": float(arr.max() - arr.min()),
        "sent_pct_positive": float((arr > 0.05).mean()),
        "sent_pct_negative": float((arr < -0.05).mean()),
    }


# ═══════════════════════════════════════════════════════════════════════
# 6. DISCOURSE COHESION
# ═══════════════════════════════════════════════════════════════════════

def discourse_cohesion(text: str) -> dict:
    """Measure local and global cohesion via sentence-embedding similarity
    and explicit connective usage."""
    model = _sentence_model()
    doc = _nlp()(text)
    sents = [s.text.strip() for s in doc.sents if len(s.text.strip()) > 10]

    # Count explicit discourse connectives
    connectives = {
        "however", "moreover", "furthermore", "therefore", "thus",
        "consequently", "nevertheless", "additionally", "meanwhile",
        "in addition", "as a result", "on the other hand", "in contrast",
        "for example", "in summary", "in conclusion", "specifically",
    }
    text_lower = text.lower()
    connective_count = sum(text_lower.count(c) for c in connectives)
    n_sents = len(sents)

    if len(sents) < 2:
        return {
            "coh_local_similarity": 0.0,
            "coh_global_similarity": 0.0,
            "coh_connective_density": 0.0,
        }

    embeddings = model.encode(sents, show_progress_bar=False)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeddings = embeddings / norms

    # Local: mean cosine similarity of adjacent sentences
    local_sims = [
        float(np.dot(embeddings[i], embeddings[i + 1]))
        for i in range(len(embeddings) - 1)
    ]

    # Global: mean pairwise similarity
    sim_matrix = embeddings @ embeddings.T
    triu_idx = np.triu_indices(len(embeddings), k=1)
    global_sims = sim_matrix[triu_idx]

    return {
        "coh_local_similarity": float(np.mean(local_sims)),
        "coh_global_similarity": float(np.mean(global_sims)),
        "coh_connective_density": connective_count / n_sents if n_sents else 0.0,
    }


# ═══════════════════════════════════════════════════════════════════════
# 7. STYLOMETRIC / POS FEATURES
# ═══════════════════════════════════════════════════════════════════════

_FUNCTION_WORDS = {
    "the", "a", "an", "and", "or", "but", "if", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "as", "is", "was", "are", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "shall", "should", "may", "might", "must", "can",
    "could", "that", "which", "who", "whom", "this", "these", "those",
    "it", "its", "he", "she", "they", "we", "you", "i", "me", "him",
    "her", "us", "them", "my", "your", "his", "our", "their", "not",
    "so", "than", "then", "when", "where", "how", "what", "all", "each",
    "every", "both", "few", "more", "most", "other", "some", "such",
    "no", "nor", "only", "own", "same", "very",
}


def stylometric_features(text: str) -> dict:
    doc = _nlp()(text)
    tokens = [t for t in doc if not t.is_space]
    n = len(tokens)
    if n == 0:
        return {}

    # POS tag distribution (coarse universal tags)
    pos_counts = Counter(t.pos_ for t in tokens)
    pos_total = sum(pos_counts.values())
    pos_ratios = {
        f"pos_{pos.lower()}_ratio": count / pos_total
        for pos, count in pos_counts.items()
    }

    # Function-word ratio
    fw = sum(1 for t in tokens if t.text.lower() in _FUNCTION_WORDS)

    # Punctuation ratio
    punct = sum(1 for t in tokens if t.is_punct)

    # Mean word length (alpha tokens only)
    alpha_lens = [len(t.text) for t in tokens if t.is_alpha]
    mean_wl = float(np.mean(alpha_lens)) if alpha_lens else 0.0

    # Exclamation and question mark counts (normalised per sentence)
    n_sents = max(len(list(doc.sents)), 1)
    excl = text.count("!")
    ques = text.count("?")

    result = {
        "sty_function_word_ratio": fw / n,
        "sty_punct_ratio": punct / n,
        "sty_mean_word_length": mean_wl,
        "sty_exclamation_per_sent": excl / n_sents,
        "sty_question_per_sent": ques / n_sents,
    }
    result.update(pos_ratios)
    return result


# ═══════════════════════════════════════════════════════════════════════
# 8. READABILITY
# ═══════════════════════════════════════════════════════════════════════

def readability(text: str) -> dict:
    return {
        "read_flesch_ease": textstat.flesch_reading_ease(text),
        "read_flesch_kincaid": textstat.flesch_kincaid_grade(text),
        "read_gunning_fog": textstat.gunning_fog(text),
        "read_coleman_liau": textstat.coleman_liau_index(text),
        "read_ari": textstat.automated_readability_index(text),
        "read_syllables_per_word": (
            textstat.syllable_count(text) / max(textstat.lexicon_count(text, True), 1)
        ),
    }


# ═══════════════════════════════════════════════════════════════════════
# AGGREGATE
# ═══════════════════════════════════════════════════════════════════════

ALL_DIMENSIONS = [
    ("lexical_diversity", lexical_diversity),
    ("syntactic_complexity", syntactic_complexity),
    ("semantic_distance", semantic_distance),
    ("textual_entropy", textual_entropy),
    ("sentiment_affect", sentiment_affect),
    ("discourse_cohesion", discourse_cohesion),
    ("stylometric", stylometric_features),
    ("readability", readability),
]


def compute_all_markers(text: str) -> dict:
    """Run all eight dimension functions and merge into one flat dict."""
    result = {}
    for name, fn in ALL_DIMENSIONS:
        try:
            result.update(fn(text))
        except Exception as e:
            print(f"  Warning: {name} failed: {e}")
    return result
