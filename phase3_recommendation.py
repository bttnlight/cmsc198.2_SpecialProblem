"""
Phase 3 — Context Inference and Recommendation Generation
==========================================================
UPDATED VERSION — Changes from original:

  CHANGE 1 (Fix #4): Context window increased from 20 → 50 events.
    Rationale: With avg 377 events per user, a 50-event window produces
    more stable context vectors with richer emotional tag coverage while
    remaining within short-term session behavior. The original 20-event
    window was insufficient to accumulate meaningful emotional signal
    given that only 17.16% of tags carry NRC emotion labels.

  CHANGE 2 (Fix #3): Recency-weighted aggregation replaces equal weighting.
    Rationale: More recent events are stronger signals of current emotional
    state. An exponential decay weight is applied per event, with the most
    recent event receiving the highest weight. This replaces the equal-
    weighting assumption of the original implementation while remaining
    consistent with the theoretical trade-off discussed in Section 2.5.
    Equal weighting is still used for track representations (unchanged)
    since tracks have no temporal ordering.

Re-run instructions
-------------------
  Delete these files from phase3_output/ before running:
    user_context_semantic.pkl
    user_context_emotion.pkl
    recommendations_semantic.pkl
    recommendations_emotion.pkl
    recommendations_popularity.pkl
  Train/test split and track representations are unchanged — keep those.
  Then run: python phase3_recommendation.py

Original methodology sections implemented
------------------------------------------
Implements sections 4.5.1, 4.5.2, 4.5.3, and the train-test split
from section 4.6.2 of the methodology.

Steps
-----
  4.6.2  Time-aware 80/20 train-test split per user
  4.5.1  Build user emotional-context vectors (mean of 20 most recent
         training events' emotion-aware tag embeddings)
  4.5.2  Build track representations (mean of track's tag embeddings)
  4.5.3  Generate top-K recommendations via cosine similarity,
         excluding tracks in the user's training history

Three models are implemented (Section 4.6.1):
  1. Popularity-based baseline   — ranks by global training frequency
  2. Semantic tag-based model    — uses semantic embeddings only (100-dim)
  3. Emotion-aware tag-based model — uses full 108-dim embeddings

Inputs  (from phase1_output/ and phase2_output/)
------
  clean_events.csv
  semantic_embeddings.pkl
  emotion_aware_embeddings.pkl

Outputs (written to phase3_output/)
-------
  train_events.pkl / test_events.pkl     — split datasets
  track_representations_semantic.pkl    — {track_key: array(100,)}
  track_representations_emotion.pkl     — {track_key: array(108,)}
  user_context_semantic.pkl             — {user: array(100,)}
  user_context_emotion.pkl              — {user: array(108,)}
  recommendations_popularity.pkl        — {user: [(track_key, score), ...]}
  recommendations_semantic.pkl
  recommendations_emotion.pkl
  phase3.log

Requirements
------------
  pip install numpy scikit-learn tqdm
"""

import os
import sys
import csv
import pickle
import logging
import numpy as np
from collections import defaultdict, Counter

try:
    from tqdm import tqdm
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "tqdm", "-q"], check=True)
    from tqdm import tqdm

try:
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "scikit-learn", "-q"], check=True)
    from sklearn.metrics.pairwise import cosine_similarity

# ────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────

PHASE1_DIR  = os.getenv("PHASE1_OUTPUT_DIR",
                         os.path.join(os.path.dirname(__file__), "phase1_output"))
PHASE2_DIR  = os.getenv("PHASE2_OUTPUT_DIR",
                         os.path.join(os.path.dirname(__file__), "phase2_output"))
OUTPUT_DIR  = os.getenv("PHASE3_OUTPUT_DIR",
                         os.path.join(os.path.dirname(__file__), "phase3_output"))
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLEAN_CSV              = os.path.join(PHASE1_DIR,  "clean_events.csv")
SEMANTIC_EMB_FILE      = os.path.join(PHASE2_DIR,  "semantic_embeddings.pkl")
EMOTION_AWARE_EMB_FILE = os.path.join(PHASE2_DIR,  "emotion_aware_embeddings.pkl")
LOG_FILE               = os.path.join(OUTPUT_DIR,  "phase3.log")

# Output files
TRAIN_FILE              = os.path.join(OUTPUT_DIR, "train_events.pkl")
TEST_FILE               = os.path.join(OUTPUT_DIR, "test_events.pkl")
TRACK_REP_SEM_FILE      = os.path.join(OUTPUT_DIR, "track_representations_semantic.pkl")
TRACK_REP_EMO_FILE      = os.path.join(OUTPUT_DIR, "track_representations_emotion.pkl")
USER_CTX_SEM_FILE       = os.path.join(OUTPUT_DIR, "user_context_semantic.pkl")
USER_CTX_EMO_FILE       = os.path.join(OUTPUT_DIR, "user_context_emotion.pkl")
RECS_POPULARITY_FILE    = os.path.join(OUTPUT_DIR, "recommendations_popularity.pkl")
RECS_SEMANTIC_FILE      = os.path.join(OUTPUT_DIR, "recommendations_semantic.pkl")
RECS_EMOTION_FILE       = os.path.join(OUTPUT_DIR, "recommendations_emotion.pkl")

# ── Methodology parameters ────────────────────────
# CHANGE 1: window size increased from 20 → 50
CONTEXT_WINDOW    = 50      # Section 4.5.1 (updated from 20)

# CHANGE 2: recency decay for context aggregation
# decay=1.0 means oldest event in window has weight e^-1 ≈ 0.37
# relative to most recent event (weight = 1.0)
RECENCY_DECAY     = 1.0

TRAIN_RATIO       = 0.80    # Section 4.6.2: unchanged
TOP_K_VALUES      = [5, 10, 20]   # Section 4.6.3
TOP_K_GENERATE    = 20      # generate top-20 so all K values are covered

# Embedding dimensions
SEM_DIM = 100
EMO_DIM = 108

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
log.info("Phase 3 — Context Inference and Recommendation Generation")
log.info("  [UPDATED: window=50, recency-weighted aggregation]")
log.info(f"  Output dir : {OUTPUT_DIR}")
log.info("=" * 60)


# ────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────

def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(obj, path: str):
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    log.info(f"  Saved: {path}  ({os.path.getsize(path) / 1024:.1f} KB)")


def load_clean_events(path: str) -> list[dict]:
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


def track_key(event: dict) -> str:
    """Consistent lowercase key for a track: 'artist||track'"""
    return f"{event['artist'].lower()}||{event['track'].lower()}"


# ────────────────────────────────────────────────
# Section 4.6.2 — Time-Aware Train-Test Split
# ────────────────────────────────────────────────

def train_test_split(events: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Per-user chronological 80/20 split.
    Each user's events are sorted by timestamp, earliest 80% → train,
    remaining 20% → test. Users with fewer than 5 events in test are
    excluded from evaluation (but their training data is retained).
    """
    if os.path.exists(TRAIN_FILE) and os.path.exists(TEST_FILE):
        log.info("  Train/test split already exists — loading from disk.")
        return load_pickle(TRAIN_FILE), load_pickle(TEST_FILE)

    # Group by user and sort chronologically
    by_user = defaultdict(list)
    for ev in events:
        by_user[ev["user"]].append(ev)
    for user in by_user:
        by_user[user].sort(key=lambda e: e["timestamp"])

    train_events, test_events = [], []
    excluded = 0

    for user, evs in by_user.items():
        split_idx = max(1, int(len(evs) * TRAIN_RATIO))
        train     = evs[:split_idx]
        test      = evs[split_idx:]

        train_events.extend(train)
        if len(test) >= 5:
            test_events.extend(test)
        else:
            excluded += 1

    log.info(f"  Train events : {len(train_events):,}")
    log.info(f"  Test events  : {len(test_events):,}")
    log.info(f"  Users with <5 test events (excluded from eval): {excluded}")

    save_pickle(train_events, TRAIN_FILE)
    save_pickle(test_events,  TEST_FILE)
    return train_events, test_events


# ────────────────────────────────────────────────
# Section 4.5.2 — Track Representations
# ────────────────────────────────────────────────

def build_track_representations(
    train_events: list[dict],
    embeddings: dict,
    dim: int,
    label: str,
) -> dict[str, np.ndarray]:
    """
    For each unique track in the training set, compute its representation
    as the mean of its associated tags' embeddings (Section 4.5.2).

    x_i = (1/|T_i|) * sum_{t in T_i} v_t^(e)

    Tracks with no tags in the embedding vocabulary are excluded.
    """
    # Collect all tags per track across all training events
    track_tags = defaultdict(set)
    for ev in train_events:
        key = track_key(ev)
        for tag in ev["tags"]:
            if tag in embeddings:
                track_tags[key].add(tag)

    representations = {}
    no_tags = 0

    for key, tags in track_tags.items():
        if not tags:
            no_tags += 1
            continue
        vecs = [embeddings[t] for t in tags]
        representations[key] = np.mean(vecs, axis=0).astype(np.float32)

    log.info(f"  [{label}] Track representations: {len(representations):,} tracks × {dim} dims")
    log.info(f"  [{label}] Tracks with no embeddings (excluded): {no_tags}")
    return representations


# ────────────────────────────────────────────────
# Section 4.5.1 — User Emotional-Context Vectors
# ────────────────────────────────────────────────

def build_user_context_vectors(
    train_events: list[dict],
    embeddings: dict,
    dim: int,
    label: str,
) -> dict[str, np.ndarray]:
    """
    CHANGE 1: Window size increased from 20 → 50 most recent events.
    CHANGE 2: Recency-weighted aggregation replaces equal weighting.

    For each event in the window, all its tags share the same event-level
    weight. Events are weighted by exponential decay from oldest to newest:
      weight(i) = exp(decay * i/N)  where i=0 is oldest, i=N-1 is newest
    This ensures more recent events contribute proportionally more to the
    user's inferred emotional context.

    c_u = sum_{e in window} w_e * mean_{t in e}(v_t^(e))
          ─────────────────────────────────────────────────
                        sum_{e in window} w_e
    """
    by_user = defaultdict(list)
    for ev in train_events:
        by_user[ev["user"]].append(ev)

    context_vectors = {}
    no_context = 0

    for user, evs in by_user.items():
        evs_sorted = sorted(evs, key=lambda e: e["timestamp"])
        recent     = evs_sorted[-CONTEXT_WINDOW:]   # CHANGE 1: up to 50 events

        # CHANGE 2: exponential decay weights — oldest → lowest, newest → highest
        n_events  = len(recent)
        positions = np.linspace(-RECENCY_DECAY, 0, n_events)
        weights   = np.exp(positions)   # shape: (n_events,)

        weighted_vecs = []
        event_weights = []

        for i, ev in enumerate(recent):
            tag_vecs = [embeddings[t] for t in ev["tags"] if t in embeddings]
            if not tag_vecs:
                continue
            event_mean = np.mean(tag_vecs, axis=0)
            weighted_vecs.append(event_mean)
            event_weights.append(weights[i])

        if not weighted_vecs:
            no_context += 1
            continue

        event_weights  = np.array(event_weights)
        weighted_stack = np.stack(weighted_vecs)
        context_vectors[user] = (
            np.sum(weighted_stack * event_weights[:, np.newaxis], axis=0)
            / event_weights.sum()
        ).astype(np.float32)

    log.info(f"  [{label}] User context vectors: {len(context_vectors):,} users × {dim} dims")
    log.info(f"  [{label}] Users with no context (excluded): {no_context}")
    return context_vectors


# ────────────────────────────────────────────────
# Section 4.5.3 — Recommendation Generation
# ────────────────────────────────────────────────

def get_user_train_tracks(train_events: list[dict]) -> dict[str, set]:
    """Return a set of track keys seen by each user during training."""
    seen = defaultdict(set)
    for ev in train_events:
        seen[ev["user"]].add(track_key(ev))
    return seen


def generate_recommendations_cosine(
    user_context: dict[str, np.ndarray],
    track_representations: dict[str, np.ndarray],
    user_train_tracks: dict[str, set],
    k: int,
    label: str,
) -> dict[str, list[tuple[str, float]]]:
    """
    For each user, rank all candidate tracks by cosine similarity with
    the user's context vector. Exclude tracks already in training history.

    score(u, i) = cos(c_u, x_i)   (Section 4.5.3)

    Returns {user: [(track_key, score), ...]} sorted descending, top-k.
    """
    # Build candidate matrix once for efficiency
    candidate_keys = list(track_representations.keys())
    candidate_matrix = np.array(
        [track_representations[k] for k in candidate_keys],
        dtype=np.float32
    )   # shape: (n_tracks, dim)

    recommendations = {}

    for user, ctx_vec in tqdm(user_context.items(),
                               desc=f"Generating [{label}] recs", unit="user"):
        seen = user_train_tracks.get(user, set())

        # Cosine similarity: (1, dim) vs (n_tracks, dim) → (1, n_tracks)
        ctx_matrix = ctx_vec.reshape(1, -1)
        sims       = cosine_similarity(ctx_matrix, candidate_matrix)[0]

        # Rank and exclude training tracks
        ranked_indices = np.argsort(sims)[::-1]
        top_k = []
        for idx in ranked_indices:
            ckey = candidate_keys[idx]
            if ckey not in seen:
                top_k.append((ckey, float(sims[idx])))
            if len(top_k) >= k:
                break

        recommendations[user] = top_k

    log.info(f"  [{label}] Recommendations generated for {len(recommendations):,} users")
    return recommendations


def generate_recommendations_popularity(
    train_events: list[dict],
    user_train_tracks: dict[str, set],
    all_users: list[str],
    k: int,
) -> dict[str, list[tuple[str, float]]]:
    """
    Popularity baseline: rank tracks by global listening frequency
    in training data, excluding each user's already-heard tracks.
    Score is normalized listen count (Section 4.6.1).
    """
    # Count global track frequency in training set
    global_counts = Counter(track_key(ev) for ev in train_events)
    max_count     = max(global_counts.values()) if global_counts else 1
    ranked_global = sorted(global_counts.items(), key=lambda x: x[1], reverse=True)

    recommendations = {}
    for user in all_users:
        seen  = user_train_tracks.get(user, set())
        top_k = []
        for ckey, count in ranked_global:
            if ckey not in seen:
                top_k.append((ckey, count / max_count))
            if len(top_k) >= k:
                break
        recommendations[user] = top_k

    log.info(f"  [popularity] Recommendations generated for {len(recommendations):,} users")
    return recommendations


# ────────────────────────────────────────────────
# Summary
# ────────────────────────────────────────────────

def print_summary(train_events, test_events,
                  track_rep_sem, track_rep_emo,
                  user_ctx_sem, user_ctx_emo,
                  recs_pop, recs_sem, recs_emo):

    test_users = {ev["user"] for ev in test_events}

    log.info("")
    log.info("=" * 60)
    log.info("PHASE 3 SUMMARY")
    log.info("=" * 60)
    log.info(f"  Train events              : {len(train_events):,}")
    log.info(f"  Test events               : {len(test_events):,}")
    log.info(f"  Users with test data      : {len(test_users):,}")
    log.info(f"  Context window size       : {CONTEXT_WINDOW} events (updated from 20)")
    log.info(f"  Aggregation               : recency-weighted (decay={RECENCY_DECAY})")
    log.info("")
    log.info(f"  Track representations")
    log.info(f"    Semantic  : {len(track_rep_sem):,} tracks × {SEM_DIM} dims")
    log.info(f"    Emotion   : {len(track_rep_emo):,} tracks × {EMO_DIM} dims")
    log.info("")
    log.info(f"  User context vectors")
    log.info(f"    Semantic  : {len(user_ctx_sem):,} users × {SEM_DIM} dims")
    log.info(f"    Emotion   : {len(user_ctx_emo):,} users × {EMO_DIM} dims")
    log.info("")
    log.info(f"  Recommendations (top-{TOP_K_GENERATE})")
    log.info(f"    Popularity : {len(recs_pop):,} users")
    log.info(f"    Semantic   : {len(recs_sem):,} users")
    log.info(f"    Emotion    : {len(recs_emo):,} users")
    log.info("")

    # Show a sample recommendation for one user
    sample_user = next(iter(recs_emo))
    log.info(f"  Sample recommendations for user '{sample_user}':")
    log.info(f"    Emotion-aware top-5:")
    for ckey, score in recs_emo[sample_user][:5]:
        artist, track = ckey.split("||")
        log.info(f"      {artist} — {track}  (score: {score:.4f})")
    log.info("")
    log.info("Phase 3 complete. Ready for Phase 4 evaluation.")


# ────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────

if __name__ == "__main__":

    # ── Step 1: Load inputs ───────────────────────
    log.info("Loading Phase 1 and Phase 2 outputs…")
    events             = load_clean_events(CLEAN_CSV)
    semantic_emb       = load_pickle(SEMANTIC_EMB_FILE)
    emotion_aware_emb  = load_pickle(EMOTION_AWARE_EMB_FILE)
    log.info(f"  Events loaded            : {len(events):,}")
    log.info(f"  Semantic embeddings      : {len(semantic_emb):,} tags")
    log.info(f"  Emotion-aware embeddings : {len(emotion_aware_emb):,} tags")

    # ── Step 2: Train-test split (Section 4.6.2) ──
    log.info("\n[4.6.2] Time-aware train-test split (80/20 per user)…")
    train_events, test_events = train_test_split(events)

    user_train_tracks = get_user_train_tracks(train_events)
    all_users         = list(user_train_tracks.keys())

    # ── Step 3: Track representations (Section 4.5.2) ──
    log.info("\n[4.5.2] Building track representations…")

    if os.path.exists(TRACK_REP_SEM_FILE):
        log.info("  Semantic track reps already exist — loading.")
        track_rep_sem = load_pickle(TRACK_REP_SEM_FILE)
    else:
        track_rep_sem = build_track_representations(
            train_events, semantic_emb, SEM_DIM, "semantic")
        save_pickle(track_rep_sem, TRACK_REP_SEM_FILE)

    if os.path.exists(TRACK_REP_EMO_FILE):
        log.info("  Emotion track reps already exist — loading.")
        track_rep_emo = load_pickle(TRACK_REP_EMO_FILE)
    else:
        track_rep_emo = build_track_representations(
            train_events, emotion_aware_emb, EMO_DIM, "emotion-aware")
        save_pickle(track_rep_emo, TRACK_REP_EMO_FILE)

    # ── Step 4: User context vectors (Section 4.5.1) ──
    log.info("\n[4.5.1] Building user context vectors (window=20)…")

    if os.path.exists(USER_CTX_SEM_FILE):
        log.info("  Semantic context vectors already exist — loading.")
        user_ctx_sem = load_pickle(USER_CTX_SEM_FILE)
    else:
        user_ctx_sem = build_user_context_vectors(
            train_events, semantic_emb, SEM_DIM, "semantic")
        save_pickle(user_ctx_sem, USER_CTX_SEM_FILE)

    if os.path.exists(USER_CTX_EMO_FILE):
        log.info("  Emotion context vectors already exist — loading.")
        user_ctx_emo = load_pickle(USER_CTX_EMO_FILE)
    else:
        user_ctx_emo = build_user_context_vectors(
            train_events, emotion_aware_emb, EMO_DIM, "emotion-aware")
        save_pickle(user_ctx_emo, USER_CTX_EMO_FILE)

    # ── Step 5: Generate recommendations (Section 4.5.3) ──
    log.info("\n[4.5.3] Generating recommendations…")

    if os.path.exists(RECS_POPULARITY_FILE):
        log.info("  Popularity recs already exist — loading.")
        recs_pop = load_pickle(RECS_POPULARITY_FILE)
    else:
        recs_pop = generate_recommendations_popularity(
            train_events, user_train_tracks, all_users, TOP_K_GENERATE)
        save_pickle(recs_pop, RECS_POPULARITY_FILE)

    if os.path.exists(RECS_SEMANTIC_FILE):
        log.info("  Semantic recs already exist — loading.")
        recs_sem = load_pickle(RECS_SEMANTIC_FILE)
    else:
        recs_sem = generate_recommendations_cosine(
            user_ctx_sem, track_rep_sem, user_train_tracks,
            TOP_K_GENERATE, "semantic")
        save_pickle(recs_sem, RECS_SEMANTIC_FILE)

    if os.path.exists(RECS_EMOTION_FILE):
        log.info("  Emotion recs already exist — loading.")
        recs_emo = load_pickle(RECS_EMOTION_FILE)
    else:
        recs_emo = generate_recommendations_cosine(
            user_ctx_emo, track_rep_emo, user_train_tracks,
            TOP_K_GENERATE, "emotion-aware")
        save_pickle(recs_emo, RECS_EMOTION_FILE)

    # ── Step 6: Summary ───────────────────────────
    print_summary(
        train_events, test_events,
        track_rep_sem, track_rep_emo,
        user_ctx_sem, user_ctx_emo,
        recs_pop, recs_sem, recs_emo
    )