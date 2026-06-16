"""
Phase 2 — Semantic and Emotion-Aware Tag Representation
=========================================================

    The VAD dimensions map to:
      [0] Valence    — pleasantness (1=unhappy → 9=happy)
      [1] Arousal    — energy/activation (1=calm → 9=excited)
      [2] Dominance  — control/power (1=submissive → 9=dominant)


Inputs  (from phase1_output/)
------
  clean_events.csv              — final validated dataset from Phase 1

Outputs (written to phase2_output/)
-------
  semantic_embeddings.pkl       — {tag: np.ndarray shape (100,)}
  emotion_vectors.pkl           — {tag: np.ndarray shape (3,)}  [VAD]
  emotion_aware_embeddings.pkl  — {tag: np.ndarray shape (103,)}
  tag_coverage_report.txt       — coverage statistics
  phase2.log                    — full run log

Requirements
------------
  pip install gensim nltk numpy

Warriner et al. (2013) Dataset
-------------------------------
  File: BRM-emot-submit.csv
  Place it in the same folder as this script.
  Download from: https://link.springer.com/article/10.3758/s13428-012-0314-x
  (Supplementary materials — freely accessible)
"""

import os
import sys
import csv
import pickle
import logging
import numpy as np
from collections import defaultdict

# ── third-party ──────────────────────────────────────────────
try:
    from gensim.models import Word2Vec
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "gensim", "-q"], check=True)
    from gensim.models import Word2Vec

try:
    import nltk
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "nltk", "-q"], check=True)
    import nltk
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize

# Download required NLTK data quietly
for pkg in ("punkt", "wordnet", "omw-1.4", "punkt_tab"):
    try:
        nltk.data.find(f"tokenizers/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

# ────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────

# Paths
PHASE1_DIR   = os.getenv("PHASE1_OUTPUT_DIR",
                          os.path.join(os.path.dirname(__file__), "phase1_output"))
OUTPUT_DIR   = os.getenv("PHASE2_OUTPUT_DIR",
                          os.path.join(os.path.dirname(__file__), "phase2_output"))
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLEAN_CSV    = os.path.join(PHASE1_DIR,  "clean_events.csv")
LOG_FILE     = os.path.join(OUTPUT_DIR,  "phase2.log")

# Warriner et al. (2013) VAD lexicon path
WARRINER_PATH = os.getenv(
    "WARRINER_PATH",
    os.path.join(os.path.dirname(__file__), "BRM-emot-submit.csv")
)

# Output files
SEMANTIC_EMB_FILE       = os.path.join(OUTPUT_DIR, "semantic_embeddings.pkl")
EMOTION_VEC_FILE        = os.path.join(OUTPUT_DIR, "emotion_vectors.pkl")
EMOTION_AWARE_EMB_FILE  = os.path.join(OUTPUT_DIR, "emotion_aware_embeddings.pkl")
W2V_MODEL_FILE          = os.path.join(OUTPUT_DIR, "word2vec_model.bin")
COVERAGE_REPORT_FILE    = os.path.join(OUTPUT_DIR, "tag_coverage_report.txt")

# Word2Vec hyperparameters 
W2V_VECTOR_SIZE  = 100
W2V_WINDOW       = 5
W2V_MIN_COUNT    = 5
W2V_EPOCHS       = 10
W2V_SG           = 1
W2V_WORKERS      = 4

# VAD dimensions
VAD_DIMS = ["valence", "arousal", "dominance"]   # 3-dim continuous
EMO_DIM  = 3   
TOTAL_DIM = W2V_VECTOR_SIZE + EMO_DIM   # 103-dim 

# ── Logging ──────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

log.info("=" * 60)
log.info("Phase 2 — Semantic and Emotion-Aware Tag Representation")
log.info("  [UPDATED: Warriner et al. VAD lexicon, 3-dim continuous]")
log.info(f"  Phase 1 input  : {CLEAN_CSV}")
log.info(f"  Output dir     : {OUTPUT_DIR}")
log.info(f"  Warriner file  : {WARRINER_PATH}")
log.info(f"  Embedding dims : {W2V_VECTOR_SIZE} semantic + {EMO_DIM} VAD = {TOTAL_DIM} total")
log.info("=" * 60)


# ────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────

def load_clean_events(path: str) -> list[dict]:
    """Load clean_events.csv produced by Phase 1."""
    events = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            events.append({
                "user":      row["user"],
                "track":     row["track"],
                "artist":    row["artist"],
                "timestamp": int(row["timestamp"]),
                "tags":      [t for t in row["tags"].split("|") if t],
            })
    return events


def save_pickle(obj, path: str):
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    log.info(f"  Saved: {path}  ({os.path.getsize(path) / 1024:.1f} KB)")


# ────────────────────────────────────────────────
# Section 4.4.1 — Semantic Tag Embedding Learning
# ────────────────────────────────────────────────

def build_tag_sentences(events: list[dict]) -> list[list[str]]:
    """
    Each listening event becomes one 'sentence' of co-occurring tags.
    This is the contextual unit for Word2Vec skip-gram training.
    Only events with at least one tag are included.
    """
    sentences = [ev["tags"] for ev in events if ev["tags"]]
    log.info(f"  Tag sentences (events with tags): {len(sentences):,}")
    return sentences


def train_word2vec(sentences: list[list[str]]) -> tuple[Word2Vec, dict]:
    """
    Train Word2Vec skip-gram model on tag sequences.
    Returns the model and a {tag: vector} dict.

    Parameters match Section 4.4.1 exactly:
      vector_size=100, window=5, min_count=5, sg=1, epochs=10
    """
    if os.path.exists(W2V_MODEL_FILE):
        log.info(f"  Word2Vec model found — loading from {W2V_MODEL_FILE}")
        model = Word2Vec.load(W2V_MODEL_FILE)
    else:
        log.info("  Training Word2Vec skip-gram model…")
        log.info(f"    vector_size={W2V_VECTOR_SIZE}, window={W2V_WINDOW}, "
                 f"min_count={W2V_MIN_COUNT}, epochs={W2V_EPOCHS}, sg={W2V_SG}")
        model = Word2Vec(
            sentences=sentences,
            vector_size=W2V_VECTOR_SIZE,
            window=W2V_WINDOW,
            min_count=W2V_MIN_COUNT,
            sg=W2V_SG,
            epochs=W2V_EPOCHS,
            workers=W2V_WORKERS,
            seed=42,
        )
        model.save(W2V_MODEL_FILE)
        log.info(f"  Model saved: {W2V_MODEL_FILE}")

    vocab = model.wv.key_to_index
    log.info(f"  Vocabulary size: {len(vocab):,} tags")

    semantic_embeddings = {tag: model.wv[tag] for tag in vocab}
    log.info(f"  Semantic embeddings built: {len(semantic_embeddings):,} tags "
             f"× {W2V_VECTOR_SIZE} dims")
    return model, semantic_embeddings


# ────────────────────────────────────────────────
# Section 4.4.2 — Emotion Mapping (Warriner VAD)
# ────────────────────────────────────────────────

def load_warriner_lexicon(path: str) -> dict[str, np.ndarray]:
    """
    Load Warriner et al. (2013) affective norms from BRM-emot-submit.csv.

    Relevant columns:
      Word, V.Mean.Sum (valence), A.Mean.Sum (arousal), D.Mean.Sum (dominance)

    Scores are on a 1–9 scale. They are normalized to [0, 1] before
    storage so that VAD vectors are on a consistent scale for
    concatenation with Word2Vec embeddings.

    Returns {word: np.ndarray shape (3,)} — [valence, arousal, dominance]
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\nWarriner dataset not found at:\n  {path}\n\n"
            "Download BRM-emot-submit.csv from:\n"
            "  https://link.springer.com/article/10.3758/s13428-012-0314-x\n"
            "(Supplementary materials — freely accessible)\n"
            "and place it in the same folder as this script."
        )

    lexicon = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                word = row["Word"].lower().strip()
                v    = float(row["V.Mean.Sum"])
                a    = float(row["A.Mean.Sum"])
                d    = float(row["D.Mean.Sum"])
                # Normalize 1–9 scale → 0–1
                lexicon[word] = np.array(
                    [(v - 1) / 8, (a - 1) / 8, (d - 1) / 8],
                    dtype=np.float32
                )
            except (KeyError, ValueError):
                continue

    log.info(f"  Warriner lexicon loaded: {len(lexicon):,} words with VAD scores")
    return lexicon


def build_emotion_vectors(
    tags: list[str],
    warriner_lexicon: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict]:
    """
    Assign a 3-dimensional VAD vector [valence, arousal, dominance]
    to each tag using the Warriner et al. (2013) affective norms.

    Same matching strategy as original:
      - Tags are tokenized and lemmatized before lookup
      - Multi-word tags: average VAD vectors of component words
      - No lexicon match: zero vector

    Returns (emotion_vectors dict, coverage_stats dict).
    """
    lemmatizer = WordNetLemmatizer()

    def lookup_word(word: str) -> np.ndarray | None:
        w = word.lower().strip()
        if w in warriner_lexicon:
            return warriner_lexicon[w]
        lemma = lemmatizer.lemmatize(w)
        if lemma in warriner_lexicon:
            return warriner_lexicon[lemma]
        return None

    emotion_vectors = {}
    stats = {"full_match": 0, "partial_match": 0, "no_match": 0}

    for tag in tags:
        tokens    = word_tokenize(tag.lower())
        tokens    = [t for t in tokens if t.isalpha()]

        if not tokens:
            emotion_vectors[tag] = np.zeros(EMO_DIM, dtype=np.float32)
            stats["no_match"] += 1
            continue

        word_vecs = [v for t in tokens if (v := lookup_word(t)) is not None]

        if len(word_vecs) == len(tokens):
            stats["full_match"] += 1
        elif word_vecs:
            stats["partial_match"] += 1
        else:
            stats["no_match"] += 1

        if word_vecs:
            emotion_vectors[tag] = np.mean(word_vecs, axis=0).astype(np.float32)
        else:
            emotion_vectors[tag] = np.zeros(EMO_DIM, dtype=np.float32)

    total = len(tags)
    stats["total"]        = total
    stats["coverage_pct"] = round(
        100 * (stats["full_match"] + stats["partial_match"]) / max(total, 1), 2
    )

    log.info(f"  VAD vector mapping complete ({total:,} tags):")
    log.info(f"    Full match    : {stats['full_match']:,}")
    log.info(f"    Partial match : {stats['partial_match']:,}")
    log.info(f"    No match      : {stats['no_match']:,}")
    log.info(f"    Coverage      : {stats['coverage_pct']}%")

    return emotion_vectors, stats


def write_coverage_report(stats: dict, tags: list[str],
                           emotion_vectors: dict[str, np.ndarray]):
    """Write a human-readable coverage report for the thesis record."""
    non_zero = sum(1 for v in emotion_vectors.values() if v.sum() > 0)
    lines = [
        "Phase 2 — Warriner et al. VAD Coverage Report",
        "=" * 50,
        f"Lexicon                  : Warriner et al. (2013) BRM",
        f"Emotion dimensions       : Valence, Arousal, Dominance (3-dim continuous)",
        f"Score normalization      : 1–9 scale → [0, 1]",
        "",
        f"Total tags in vocabulary : {stats['total']:,}",
        f"Full match               : {stats['full_match']:,}",
        f"Partial match            : {stats['partial_match']:,}",
        f"No match (zero vector)   : {stats['no_match']:,}",
        f"Overall coverage         : {stats['coverage_pct']}%",
        f"Tags with non-zero VAD   : {non_zero:,}",
        "",
        "VAD dimension statistics (matched tags only):",
    ]
    matched_vecs = [v for v in emotion_vectors.values() if v.sum() > 0]
    if matched_vecs:
        arr = np.stack(matched_vecs)
        for i, dim in enumerate(VAD_DIMS):
            lines.append(
                f"  {dim:<12}: mean={arr[:,i].mean():.3f}  "
                f"std={arr[:,i].std():.3f}  "
                f"min={arr[:,i].min():.3f}  "
                f"max={arr[:,i].max():.3f}"
            )

    with open(COVERAGE_REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log.info(f"  Coverage report saved: {COVERAGE_REPORT_FILE}")


# ────────────────────────────────────────────────
# Section 4.4.3 — Emotion-Aware Tag Representation
# ────────────────────────────────────────────────

def build_emotion_aware_embeddings(
    semantic_embeddings: dict[str, np.ndarray],
    emotion_vectors: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """
    Concatenate semantic vector (100-dim) and VAD vector (3-dim)
    into a single emotion-aware embedding (103-dim) per tag.

    v_t^(e) = [v_t ; e_t]   (Section 4.4.3)

    CHANGED: emotion vector is now 3-dim continuous VAD
    instead of 8-dim binary NRC, giving 103-dim total.
    """
    emotion_aware = {}
    missing_emotion = 0

    for tag, sem_vec in semantic_embeddings.items():
        emo_vec = emotion_vectors.get(tag, np.zeros(EMO_DIM, dtype=np.float32))
        if tag not in emotion_vectors:
            missing_emotion += 1
        emotion_aware[tag] = np.concatenate([sem_vec, emo_vec]).astype(np.float32)

    log.info(f"  Emotion-aware embeddings built: {len(emotion_aware):,} tags × "
             f"{TOTAL_DIM} dims ({W2V_VECTOR_SIZE} semantic + {EMO_DIM} VAD)")
    log.info(f"  Tags using zero VAD vector: {missing_emotion:,}")

    sample_tag = next(iter(emotion_aware))
    sample_vec = emotion_aware[sample_tag]
    assert sample_vec.shape == (TOTAL_DIM,), \
        f"Unexpected embedding shape: {sample_vec.shape}"
    log.info(f"  Shape check passed: {sample_vec.shape} ✓")

    return emotion_aware


# ────────────────────────────────────────────────
# Summary
# ────────────────────────────────────────────────

def print_summary(semantic_embeddings, emotion_vectors, emotion_aware_embeddings):
    log.info("")
    log.info("=" * 60)
    log.info("PHASE 2 SUMMARY")
    log.info("=" * 60)
    log.info(f"  Semantic embeddings      : {len(semantic_embeddings):,} tags × {W2V_VECTOR_SIZE} dims")
    log.info(f"  VAD emotion vectors      : {len(emotion_vectors):,} tags × {EMO_DIM} dims")
    log.info(f"  Emotion-aware embeddings : {len(emotion_aware_embeddings):,} tags × {TOTAL_DIM} dims")
    log.info("")
    log.info("  VAD dimensions (Warriner et al. 2013):")
    for i, dim in enumerate(VAD_DIMS):
        log.info(f"    [{i}] {dim}")
    log.info("")

    # Show a sample tag with its VAD breakdown
    sample_tag = next(iter(emotion_aware_embeddings))
    emo_vec    = emotion_vectors.get(sample_tag, np.zeros(EMO_DIM))
    log.info(f"  Sample tag: '{sample_tag}'")
    log.info(f"    VAD vector     : {np.round(emo_vec, 3).tolist()}  "
             f"(valence={emo_vec[0]:.3f}, arousal={emo_vec[1]:.3f}, dominance={emo_vec[2]:.3f})")
    log.info(f"    Full vector dim: {emotion_aware_embeddings[sample_tag].shape[0]}")
    log.info("")
    log.info("Phase 2 complete. Ready for Phase 3.")


# ────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────

if __name__ == "__main__":

    # ── Guard: check Warriner file exists ────────────────────────
    if not os.path.exists(WARRINER_PATH):
        log.error(
            f"\nWarriner dataset not found at: {WARRINER_PATH}\n"
            "Download BRM-emot-submit.csv from:\n"
            "  https://link.springer.com/article/10.3758/s13428-012-0314-x\n"
            "(Supplementary materials — freely accessible)\n"
            "Then place it in the same folder as this script and re-run."
        )
        sys.exit(1)

    # ── Step 1: Load Phase 1 output ──────────────────────────────
    log.info("Loading clean events from Phase 1…")
    events = load_clean_events(CLEAN_CSV)
    log.info(f"  Loaded {len(events):,} events")

    # ── Step 2: Build tag sentences for Word2Vec ─────────────────
    log.info("\n[4.4.1] Building tag sentences…")
    sentences = build_tag_sentences(events)

    # ── Step 3: Train Word2Vec skip-gram (or load existing) ──────
    log.info("\n[4.4.1] Training Word2Vec skip-gram…")
    if os.path.exists(SEMANTIC_EMB_FILE):
        log.info(f"  Semantic embeddings already exist — loading from {SEMANTIC_EMB_FILE}")
        with open(SEMANTIC_EMB_FILE, "rb") as f:
            semantic_embeddings = pickle.load(f)
    else:
        _, semantic_embeddings = train_word2vec(sentences)
        save_pickle(semantic_embeddings, SEMANTIC_EMB_FILE)

    all_tags = list(semantic_embeddings.keys())
    log.info(f"  Tags in semantic vocabulary: {len(all_tags):,}")

    # ── Step 4: Load Warriner lexicon and build VAD vectors ──────
    log.info("\n[4.4.2] Loading Warriner et al. (2013) VAD lexicon…")
    warriner_lexicon = load_warriner_lexicon(WARRINER_PATH)

    if os.path.exists(EMOTION_VEC_FILE):
        log.info(f"  VAD vectors already exist — loading from {EMOTION_VEC_FILE}")
        with open(EMOTION_VEC_FILE, "rb") as f:
            emotion_vectors = pickle.load(f)
        coverage_stats = {"total": len(all_tags)}
    else:
        log.info("[4.4.2] Building VAD vectors for all tags…")
        emotion_vectors, coverage_stats = build_emotion_vectors(all_tags, warriner_lexicon)
        save_pickle(emotion_vectors, EMOTION_VEC_FILE)
        write_coverage_report(coverage_stats, all_tags, emotion_vectors)

    # ── Step 5: Concatenate → emotion-aware embeddings ───────────
    log.info("\n[4.4.3] Building emotion-aware embeddings (concatenation)…")
    if os.path.exists(EMOTION_AWARE_EMB_FILE):
        log.info(f"  Already exists — loading from {EMOTION_AWARE_EMB_FILE}")
        with open(EMOTION_AWARE_EMB_FILE, "rb") as f:
            emotion_aware_embeddings = pickle.load(f)
    else:
        emotion_aware_embeddings = build_emotion_aware_embeddings(
            semantic_embeddings, emotion_vectors
        )
        save_pickle(emotion_aware_embeddings, EMOTION_AWARE_EMB_FILE)

    # ── Step 6: Summary ───────────────────────────────────────────
    print_summary(semantic_embeddings, emotion_vectors, emotion_aware_embeddings)