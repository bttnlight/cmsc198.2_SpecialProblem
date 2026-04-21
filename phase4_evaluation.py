"""
Phase 4 — Experimental Evaluation (Offline)
============================================
Implements sections 4.6.1 through 4.6.4 of the methodology.

Steps
-----
  4.6.1  Comparative experimental design (3 models)
  4.6.2  Train-test split already done in Phase 3
  4.6.3  Precision@K and Recall@K at K = 5, 10, 20
  4.6.4  Emotion alignment evaluation (cosine similarity
         between emotional components of user context
         and recommended track vectors)

Models evaluated
----------------
  1. Popularity-based baseline
  2. Semantic tag-based model
  3. Emotion-aware tag-based model

Inputs  (from phase3_output/ and phase2_output/)
------
  test_events.pkl
  recommendations_popularity/semantic/emotion.pkl
  user_context_semantic/emotion.pkl
  track_representations_semantic/emotion.pkl
  emotion_vectors.pkl  (from Phase 2, for alignment evaluation)

Outputs (written to phase4_output/)
-------
  evaluation_results.csv     — Precision@K and Recall@K per model per K
  emotion_alignment.csv      — emotion alignment scores per model
  full_results_report.txt    — human-readable summary for thesis write-up
  per_user_metrics.pkl       — per-user scores for statistical testing
  phase4.log

Requirements
------------
  pip install numpy scipy
"""

import os
import sys
import csv
import pickle
import logging
import numpy as np
from collections import defaultdict

try:
    from scipy import stats
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "scipy", "-q"], check=True)
    from scipy import stats

# ────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────

PHASE2_DIR  = os.getenv("PHASE2_OUTPUT_DIR",
                         os.path.join(os.path.dirname(__file__), "phase2_output"))
PHASE3_DIR  = os.getenv("PHASE3_OUTPUT_DIR",
                         os.path.join(os.path.dirname(__file__), "phase3_output"))
OUTPUT_DIR  = os.getenv("PHASE4_OUTPUT_DIR",
                         os.path.join(os.path.dirname(__file__), "phase4_output"))
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Input files
TEST_FILE            = os.path.join(PHASE3_DIR, "test_events.pkl")
TRAIN_FILE           = os.path.join(PHASE3_DIR, "train_events.pkl")
RECS_POPULARITY_FILE = os.path.join(PHASE3_DIR, "recommendations_popularity.pkl")
RECS_SEMANTIC_FILE   = os.path.join(PHASE3_DIR, "recommendations_semantic.pkl")
RECS_EMOTION_FILE    = os.path.join(PHASE3_DIR, "recommendations_emotion.pkl")
TRACK_REP_SEM_FILE   = os.path.join(PHASE3_DIR, "track_representations_semantic.pkl")
TRACK_REP_EMO_FILE   = os.path.join(PHASE3_DIR, "track_representations_emotion.pkl")
EMOTION_VEC_FILE     = os.path.join(PHASE2_DIR, "emotion_vectors.pkl")

# Output files
EVAL_CSV_FILE        = os.path.join(OUTPUT_DIR, "evaluation_results.csv")
ALIGNMENT_CSV_FILE   = os.path.join(OUTPUT_DIR, "emotion_alignment.csv")
REPORT_FILE          = os.path.join(OUTPUT_DIR, "full_results_report.txt")
PER_USER_FILE        = os.path.join(OUTPUT_DIR, "per_user_metrics.pkl")
LOG_FILE             = os.path.join(OUTPUT_DIR, "phase4.log")

# Evaluation parameters — Section 4.6.3
K_VALUES   = [5, 10, 20]
# VAD dimensions from Warriner — 3-dim continuous (updated from NRC 8-dim binary)
VAD_DIM    = 3
VAD_DIMS   = ["valence", "arousal", "dominance"]

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
log.info("Phase 4 — Experimental Evaluation (Offline)")
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


def track_key(artist: str, track: str) -> str:
    return f"{artist.lower()}||{track.lower()}"


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ────────────────────────────────────────────────
# Build ground truth from test set
# ────────────────────────────────────────────────

def build_ground_truth(test_events: list[dict]) -> dict[str, set]:
    """
    For each user, collect the set of track keys they actually
    listened to in the test period. These are the relevant items
    for Precision@K and Recall@K.
    """
    ground_truth = defaultdict(set)
    for ev in test_events:
        key = track_key(ev["artist"], ev["track"])
        ground_truth[ev["user"]].add(key)
    return dict(ground_truth)


# ────────────────────────────────────────────────
# Section 4.6.3 — Precision@K and Recall@K
# ────────────────────────────────────────────────

def precision_at_k(recommended: list[str], relevant: set, k: int) -> float:
    """Proportion of top-k recommendations that are relevant."""
    top_k = recommended[:k]
    hits  = sum(1 for r in top_k if r in relevant)
    return hits / k


def recall_at_k(recommended: list[str], relevant: set, k: int) -> float:
    """Proportion of relevant items successfully retrieved in top-k."""
    top_k = recommended[:k]
    hits  = sum(1 for r in top_k if r in relevant)
    return hits / len(relevant) if relevant else 0.0


def evaluate_model(
    recommendations: dict[str, list[tuple[str, float]]],
    ground_truth: dict[str, set],
    k_values: list[int],
    model_name: str,
) -> dict:
    """
    Compute Precision@K and Recall@K for a model across all K values.
    Scores are computed per user then macro-averaged (Section 4.6.3).
    Only users present in both recommendations and ground truth are evaluated.
    """
    eval_users = [u for u in recommendations if u in ground_truth]
    log.info(f"  [{model_name}] Evaluating {len(eval_users):,} users…")

    results     = {}
    per_user_scores = {k: {"precision": [], "recall": []} for k in k_values}

    for user in eval_users:
        recs    = [r[0] for r in recommendations[user]]   # track keys only
        relevant = ground_truth[user]

        for k in k_values:
            p = precision_at_k(recs, relevant, k)
            r = recall_at_k(recs, relevant, k)
            per_user_scores[k]["precision"].append(p)
            per_user_scores[k]["recall"].append(r)

    for k in k_values:
        p_scores = per_user_scores[k]["precision"]
        r_scores = per_user_scores[k]["recall"]
        results[k] = {
            "precision_mean": float(np.mean(p_scores)),
            "precision_std":  float(np.std(p_scores)),
            "recall_mean":    float(np.mean(r_scores)),
            "recall_std":     float(np.std(r_scores)),
            "n_users":        len(eval_users),
            "per_user_precision": p_scores,
            "per_user_recall":    r_scores,
        }
        log.info(f"    K={k:>2}  Precision={results[k]['precision_mean']:.4f} "
                 f"(±{results[k]['precision_std']:.4f})  "
                 f"Recall={results[k]['recall_mean']:.4f} "
                 f"(±{results[k]['recall_std']:.4f})")

    return results


# ────────────────────────────────────────────────
# Section 4.6.4 — Emotion Alignment Evaluation
# ────────────────────────────────────────────────

def build_user_emotion_profile(
    train_events: list[dict],
    emotion_vectors: dict[str, np.ndarray],
    context_window: int = 50,
) -> dict[str, np.ndarray]:
    """
    Build each user's emotion profile directly from Warriner VAD vectors,
    independent of any embedding representation.

    Per Section 4.6.4: emotion alignment must be measured in a shared
    affective space independent of embedding representations.
    Uses Warriner 3-dim continuous VAD vectors [valence, arousal, dominance].
    Only tags with non-zero VAD vectors contribute to the profile.
    """
    by_user = defaultdict(list)
    for ev in train_events:
        by_user[ev["user"]].append(ev)

    user_emotion_profiles = {}

    for user, evs in by_user.items():
        evs_sorted = sorted(evs, key=lambda e: e["timestamp"])
        recent     = evs_sorted[-context_window:]

        emo_vecs = []
        for ev in recent:
            for tag in ev["tags"]:
                if tag in emotion_vectors:
                    vec = emotion_vectors[tag]
                    if vec.sum() > 0:
                        emo_vecs.append(vec)

        if emo_vecs:
            user_emotion_profiles[user] = np.mean(emo_vecs, axis=0).astype(np.float32)
        else:
            user_emotion_profiles[user] = np.zeros(VAD_DIM, dtype=np.float32)

    return user_emotion_profiles


def build_track_emotion_profile(
    track_representations_keys: list[str],
    train_events: list[dict],
    emotion_vectors: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """
    Build each track's emotion profile from Warriner VAD vectors.
    Same shared affective space as build_user_emotion_profile.
    """
    track_tags = defaultdict(set)
    for ev in train_events:
        key = f"{ev['artist'].lower()}||{ev['track'].lower()}"
        for tag in ev["tags"]:
            if tag in emotion_vectors:
                track_tags[key].add(tag)

    track_emotion_profiles = {}
    for key in track_representations_keys:
        tags     = track_tags.get(key, set())
        emo_vecs = [emotion_vectors[t] for t in tags
                    if t in emotion_vectors and emotion_vectors[t].sum() > 0]
        if emo_vecs:
            track_emotion_profiles[key] = np.mean(emo_vecs, axis=0).astype(np.float32)
        else:
            track_emotion_profiles[key] = np.zeros(VAD_DIM, dtype=np.float32)

    return track_emotion_profiles


def compute_emotion_alignment(
    recommendations: dict[str, list[tuple[str, float]]],
    user_emotion_profiles: dict[str, np.ndarray],
    track_emotion_profiles: dict[str, np.ndarray],
    model_name: str,
    k: int = 10,
) -> dict:
    """
    Emotion alignment computed in a shared Warriner VAD space for BOTH models.

    Alignment = cosine similarity between user's VAD emotion profile
    and each recommended track's VAD emotion profile, averaged over
    top-k recommendations then averaged over all users.

    Per Section 4.6.4: measured in a shared affective space independent
    of embedding representations. Uses Warriner 3-dim continuous VAD
    vectors [valence, arousal, dominance].
    """
    eval_users = [u for u in recommendations if u in user_emotion_profiles]
    per_user_alignment = []
    zero_vector_users  = 0

    for user in eval_users:
        ctx_emo = user_emotion_profiles[user]

        if ctx_emo.sum() == 0:
            zero_vector_users += 1
            continue

        recs = recommendations[user][:k]
        if not recs:
            continue

        track_alignments = []
        for track_key_str, _ in recs:
            track_emo = track_emotion_profiles.get(
                track_key_str, np.zeros(VAD_DIM, dtype=np.float32))
            sim = cosine_sim(ctx_emo, track_emo)
            track_alignments.append(sim)

        if track_alignments:
            per_user_alignment.append(float(np.mean(track_alignments)))

    mean_alignment = float(np.mean(per_user_alignment)) if per_user_alignment else 0.0
    std_alignment  = float(np.std(per_user_alignment))  if per_user_alignment else 0.0

    log.info(f"  [{model_name}] Emotion alignment @{k}: "
             f"{mean_alignment:.4f} (±{std_alignment:.4f})  "
             f"n={len(per_user_alignment)}  "
             f"(zero-profile users skipped: {zero_vector_users})")

    return {
        "mean":               mean_alignment,
        "std":                std_alignment,
        "n_users":            len(per_user_alignment),
        "zero_profile_users": zero_vector_users,
        "per_user":           per_user_alignment,
    }


# ────────────────────────────────────────────────
# Statistical comparison (Section 4.6.6)
# ────────────────────────────────────────────────

def compare_models(scores_a: list[float], scores_b: list[float],
                   label_a: str, label_b: str, metric: str):
    """
    Paired-sample t-test between two models' per-user scores.
    Falls back to Wilcoxon signed-rank test if normality fails.
    Reports effect size (Cohen's d).
    Per Section 4.6.5.4.
    """
    # Align user counts
    n  = min(len(scores_a), len(scores_b))
    a  = np.array(scores_a[:n])
    b  = np.array(scores_b[:n])
    d  = a - b

    # Shapiro-Wilk normality test on differences (sample up to 5000)
    sample = d[:5000] if len(d) > 5000 else d
    _, p_normality = stats.shapiro(sample)
    normal = p_normality > 0.05

    if normal:
        t_stat, p_val = stats.ttest_rel(a, b)
        test_name = "paired t-test"
    else:
        t_stat, p_val = stats.wilcoxon(a, b, zero_method="wilcox")
        test_name = "Wilcoxon signed-rank"

    # Cohen's d effect size
    pooled_std = np.std(d) if np.std(d) > 0 else 1e-9
    cohens_d   = float(np.mean(d) / pooled_std)

    # Effect size interpretation
    if abs(cohens_d) < 0.2:
        magnitude = "negligible"
    elif abs(cohens_d) < 0.5:
        magnitude = "small"
    elif abs(cohens_d) < 0.8:
        magnitude = "medium"
    else:
        magnitude = "large"

    result = {
        "metric":     metric,
        "label_a":    label_a,
        "label_b":    label_b,
        "test":       test_name,
        "statistic":  float(t_stat),
        "p_value":    float(p_val),
        "cohens_d":   cohens_d,
        "magnitude":  magnitude,
        "significant": p_val < 0.05,
        "n":          n,
    }

    sig = "✓ significant" if p_val < 0.05 else "✗ not significant"
    log.info(f"    {label_a} vs {label_b} [{metric}]: "
             f"{test_name}, p={p_val:.4f} ({sig}), "
             f"d={cohens_d:.3f} ({magnitude})")
    return result


# ────────────────────────────────────────────────
# Fix #5 — Stratified analysis by emotional tag density
# ────────────────────────────────────────────────

def compute_emotional_density(
    train_events: list[dict],
    emotion_vectors: dict[str, np.ndarray],
) -> dict[str, float]:
    """
    For each user, compute the proportion of their training tags
    that carry non-zero NRC emotion signal.

    Users with higher emotional density have more emotionally-tagged
    listening history and are expected to benefit more from emotion-aware
    recommendations.
    """
    by_user = defaultdict(list)
    for ev in train_events:
        by_user[ev["user"]].append(ev)

    densities = {}
    for user, evs in by_user.items():
        all_tags      = [t for ev in evs for t in ev["tags"]]
        emotional     = sum(1 for t in all_tags
                            if t in emotion_vectors
                            and emotion_vectors[t].sum() > 0)
        densities[user] = emotional / len(all_tags) if all_tags else 0.0

    return densities


def stratified_evaluation(
    recommendations_sem: dict,
    recommendations_emo: dict,
    ground_truth: dict,
    emotional_densities: dict,
    k_values: list[int],
) -> dict:
    """
    Split users at the median emotional density into HIGH and LOW groups.
    Evaluate each model separately per group.

    Hypothesis: emotion-aware model outperforms semantic model specifically
    for HIGH-density users whose context vectors carry meaningful emotion signal.
    """
    users_with_data = [u for u in recommendations_sem if u in ground_truth]
    densities       = [emotional_densities.get(u, 0.0) for u in users_with_data]
    median_density  = float(np.median(densities))

    high_users = [u for u, d in zip(users_with_data, densities)
                  if d >= median_density]
    low_users  = [u for u, d in zip(users_with_data, densities)
                  if d < median_density]

    log.info(f"  Stratification median density : {median_density:.4f}")
    log.info(f"  High-density users            : {len(high_users):,}")
    log.info(f"  Low-density users             : {len(low_users):,}")

    results = {}
    for group_name, group_users in [("high_density", high_users),
                                     ("low_density",  low_users)]:
        results[group_name] = {}
        for model_name, recs in [("semantic",     recommendations_sem),
                                   ("emotion-aware", recommendations_emo)]:
            group_results = {}
            for k in k_values:
                p_scores, r_scores = [], []
                for user in group_users:
                    if user not in recs:
                        continue
                    rec_list = [r[0] for r in recs[user]]
                    relevant = ground_truth[user]
                    p_scores.append(precision_at_k(rec_list, relevant, k))
                    r_scores.append(recall_at_k(rec_list, relevant, k))
                group_results[k] = {
                    "precision_mean": float(np.mean(p_scores)) if p_scores else 0.0,
                    "recall_mean":    float(np.mean(r_scores)) if r_scores else 0.0,
                    "n_users":        len(p_scores),
                }
                log.info(
                    f"  [{group_name}][{model_name}] K={k}: "
                    f"P={group_results[k]['precision_mean']:.4f}  "
                    f"R={group_results[k]['recall_mean']:.4f}  "
                    f"n={group_results[k]['n_users']}"
                )
            results[group_name][model_name] = group_results

    results["median_density"] = median_density
    return results


def save_stratified_csv(stratified_results: dict, output_dir: str):
    path = os.path.join(output_dir, "stratified_results.csv")
    rows = []
    for group in ["high_density", "low_density"]:
        if group not in stratified_results:
            continue
        for model in ["semantic", "emotion-aware"]:
            if model not in stratified_results[group]:
                continue
            for k, metrics in stratified_results[group][model].items():
                rows.append({
                    "group":           group,
                    "model":           model,
                    "k":               k,
                    "precision_mean":  round(metrics["precision_mean"], 6),
                    "recall_mean":     round(metrics["recall_mean"],    6),
                    "n_users":         metrics["n_users"],
                })
    if rows:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        log.info(f"  Stratified results saved: {path}")


# ────────────────────────────────────────────────
# Save results
# ────────────────────────────────────────────────

def save_evaluation_csv(all_results: dict):
    """Save Precision@K and Recall@K to CSV."""
    rows = []
    for model_name, k_results in all_results.items():
        for k, metrics in k_results.items():
            rows.append({
                "model":           model_name,
                "k":               k,
                "precision_mean":  round(metrics["precision_mean"], 6),
                "precision_std":   round(metrics["precision_std"],  6),
                "recall_mean":     round(metrics["recall_mean"],    6),
                "recall_std":      round(metrics["recall_std"],     6),
                "n_users":         metrics["n_users"],
            })
    with open(EVAL_CSV_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    log.info(f"  Evaluation results saved: {EVAL_CSV_FILE}")


def save_alignment_csv(alignment_results: dict):
    """Save emotion alignment scores to CSV."""
    rows = []
    for model_name, k_results in alignment_results.items():
        for k, metrics in k_results.items():
            rows.append({
                "model":  model_name,
                "k":      k,
                "alignment_mean": round(metrics["mean"], 6),
                "alignment_std":  round(metrics["std"],  6),
                "n_users":        metrics["n_users"],
            })
    with open(ALIGNMENT_CSV_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    log.info(f"  Emotion alignment results saved: {ALIGNMENT_CSV_FILE}")


def write_report(all_results: dict, alignment_results: dict,
                 stat_comparisons: list[dict]):
    """Write a full human-readable report for the thesis."""
    lines = []
    lines.append("=" * 65)
    lines.append("PHASE 4 — OFFLINE EVALUATION REPORT")
    lines.append("=" * 65)
    lines.append("")

    # ── Precision@K and Recall@K ─────────────────
    lines.append("4.6.3  RECOMMENDATION ACCURACY (Precision@K / Recall@K)")
    lines.append("-" * 65)
    header = f"{'Model':<25} {'K':>3}  {'Prec':>8}  {'±':>8}  {'Rec':>8}  {'±':>8}"
    lines.append(header)
    lines.append("-" * 65)

    model_order = ["popularity", "semantic", "emotion-aware"]
    for model in model_order:
        if model not in all_results:
            continue
        for k in K_VALUES:
            m = all_results[model][k]
            lines.append(
                f"{model:<25} {k:>3}  "
                f"{m['precision_mean']:>8.4f}  {m['precision_std']:>8.4f}  "
                f"{m['recall_mean']:>8.4f}  {m['recall_std']:>8.4f}"
            )
        lines.append("")

    # ── Emotion Alignment ────────────────────────
    lines.append("4.6.4  EMOTION ALIGNMENT")
    lines.append("-" * 65)
    header2 = f"{'Model':<25} {'K':>3}  {'Alignment':>10}  {'±':>8}  {'N':>6}"
    lines.append(header2)
    lines.append("-" * 65)
    for model in ["semantic", "emotion-aware"]:
        if model not in alignment_results:
            continue
        for k in K_VALUES:
            m = alignment_results[model][k]
            lines.append(
                f"{model:<25} {k:>3}  "
                f"{m['mean']:>10.4f}  {m['std']:>8.4f}  {m['n_users']:>6}"
            )
        lines.append("")

    # ── Statistical Tests ────────────────────────
    lines.append("4.6.6  COMPARATIVE ANALYSIS (Statistical Tests)")
    lines.append("-" * 65)
    for comp in stat_comparisons:
        sig = "SIGNIFICANT" if comp["significant"] else "not significant"
        lines.append(
            f"  {comp['label_a']} vs {comp['label_b']} [{comp['metric']}]"
        )
        lines.append(
            f"    Test     : {comp['test']}"
        )
        lines.append(
            f"    Statistic: {comp['statistic']:.4f}   p-value: {comp['p_value']:.4f}  ({sig})"
        )
        lines.append(
            f"    Cohen's d: {comp['cohens_d']:.3f} ({comp['magnitude']} effect)"
        )
        lines.append(f"    N users  : {comp['n']}")
        lines.append("")

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log.info(f"  Full report saved: {REPORT_FILE}")


# ────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────

if __name__ == "__main__":

    # ── Step 1: Load all Phase 3 outputs ─────────
    log.info("Loading Phase 3 outputs…")
    test_events   = load_pickle(TEST_FILE)
    recs_pop      = load_pickle(RECS_POPULARITY_FILE)
    recs_sem      = load_pickle(RECS_SEMANTIC_FILE)
    recs_emo      = load_pickle(RECS_EMOTION_FILE)
    train_events  = load_pickle(os.path.join(PHASE3_DIR, "train_events.pkl"))
    track_rep_sem = load_pickle(TRACK_REP_SEM_FILE)
    track_rep_emo = load_pickle(TRACK_REP_EMO_FILE)
    emotion_vecs  = load_pickle(EMOTION_VEC_FILE)
    log.info(f"  Test events  : {len(test_events):,}")
    log.info(f"  Test users   : {len({e['user'] for e in test_events}):,}")

    # ── Step 2: Build ground truth ────────────────
    log.info("\nBuilding ground truth from test events…")
    ground_truth = build_ground_truth(test_events)
    log.info(f"  Users with ground truth: {len(ground_truth):,}")
    avg_relevant = np.mean([len(v) for v in ground_truth.values()])
    log.info(f"  Avg relevant tracks per user: {avg_relevant:.1f}")

    # ── Step 3: Precision@K and Recall@K ─────────
    log.info("\n[4.6.3] Computing Precision@K and Recall@K…")
    log.info("  Popularity baseline:")
    results_pop = evaluate_model(recs_pop, ground_truth, K_VALUES, "popularity")
    log.info("  Semantic model:")
    results_sem = evaluate_model(recs_sem, ground_truth, K_VALUES, "semantic")
    log.info("  Emotion-aware model:")
    results_emo = evaluate_model(recs_emo, ground_truth, K_VALUES, "emotion-aware")

    all_results = {
        "popularity":    results_pop,
        "semantic":      results_sem,
        "emotion-aware": results_emo,
    }

    # ── Step 4: Build shared NRC emotion space ────
    log.info("\n[4.6.4] Building shared Warriner VAD emotion profiles…")
    log.info("  (both models evaluated in shared 3-dim VAD space)")
    user_emo_profiles   = build_user_emotion_profile(
        train_events, emotion_vecs, context_window=50)
    track_emo_profiles  = build_track_emotion_profile(
        list(track_rep_sem.keys()), train_events, emotion_vecs)

    users_with_signal = sum(1 for v in user_emo_profiles.values() if v.sum() > 0)
    log.info(f"  Users with non-zero emotion profile: {users_with_signal:,} / {len(user_emo_profiles):,}")

    # ── Step 5: Emotion Alignment (FIXED) ────────
    log.info("\n[4.6.4] Computing emotion alignment (shared lexicon space)…")
    alignment_results = {"semantic": {}, "emotion-aware": {}}

    for k in K_VALUES:
        alignment_results["semantic"][k] = compute_emotion_alignment(
            recs_sem, user_emo_profiles, track_emo_profiles,
            model_name="semantic", k=k)

        alignment_results["emotion-aware"][k] = compute_emotion_alignment(
            recs_emo, user_emo_profiles, track_emo_profiles,
            model_name="emotion-aware", k=k)

    # ── Step 6: Statistical comparisons ──────────
    log.info("\n[4.6.6] Statistical comparisons (semantic vs emotion-aware)…")
    stat_comparisons = []

    for k in K_VALUES:
        stat_comparisons.append(compare_models(
            results_emo[k]["per_user_precision"],
            results_sem[k]["per_user_precision"],
            "emotion-aware", "semantic", f"Precision@{k}"))
        stat_comparisons.append(compare_models(
            results_emo[k]["per_user_recall"],
            results_sem[k]["per_user_recall"],
            "emotion-aware", "semantic", f"Recall@{k}"))

    stat_comparisons.append(compare_models(
        alignment_results["emotion-aware"][10]["per_user"],
        alignment_results["semantic"][10]["per_user"],
        "emotion-aware", "semantic", "EmotionAlignment@10"))

    # ── Step 7: Stratified analysis (Fix #5) ─────
    log.info("\n[Stratified] Evaluating by emotional tag density…")
    emotional_densities = compute_emotional_density(train_events, emotion_vecs)
    stratified_results  = stratified_evaluation(
        recs_sem, recs_emo, ground_truth,
        emotional_densities, K_VALUES)

    # ── Step 8: Save all outputs ──────────────────
    log.info("\nSaving results…")
    save_evaluation_csv(all_results)
    save_alignment_csv(alignment_results)
    save_stratified_csv(stratified_results, OUTPUT_DIR)

    per_user_data = {
        "popularity":       results_pop,
        "semantic":         results_sem,
        "emotion-aware":    results_emo,
        "alignment":        alignment_results,
        "stat_comparisons": stat_comparisons,
        "stratified":       stratified_results,
        "emotional_densities": emotional_densities,
    }
    save_pickle(per_user_data, PER_USER_FILE)
    write_report(all_results, alignment_results, stat_comparisons)

    # ── Step 9: Console summary ───────────────────
    log.info("")
    log.info("=" * 60)
    log.info("PHASE 4 SUMMARY")
    log.info("=" * 60)
    log.info(f"  {'Model':<20} {'P@5':>7} {'P@10':>7} {'P@20':>7} "
             f"{'R@5':>7} {'R@10':>7} {'R@20':>7}")
    log.info(f"  {'-'*62}")
    for model in ["popularity", "semantic", "emotion-aware"]:
        r = all_results[model]
        log.info(
            f"  {model:<20} "
            f"{r[5]['precision_mean']:>7.4f} "
            f"{r[10]['precision_mean']:>7.4f} "
            f"{r[20]['precision_mean']:>7.4f} "
            f"{r[5]['recall_mean']:>7.4f} "
            f"{r[10]['recall_mean']:>7.4f} "
            f"{r[20]['recall_mean']:>7.4f}"
        )
    log.info("")
    log.info(f"  {'Model':<20} {'Align@5':>9} {'Align@10':>9} {'Align@20':>9}")
    log.info(f"  {'-'*50}")
    for model in ["semantic", "emotion-aware"]:
        a = alignment_results[model]
        log.info(
            f"  {model:<20} "
            f"{a[5]['mean']:>9.4f} "
            f"{a[10]['mean']:>9.4f} "
            f"{a[20]['mean']:>9.4f}"
        )
    log.info("")
    log.info("  STRATIFIED RESULTS (emotion-aware vs semantic, P@10):")
    log.info(f"  {'Group':<20} {'Semantic':>10} {'Emo-Aware':>10} {'Difference':>12}")
    log.info(f"  {'-'*55}")
    for group in ["high_density", "low_density"]:
        if group not in stratified_results:
            continue
        sem_p = stratified_results[group]["semantic"][10]["precision_mean"]
        emo_p = stratified_results[group]["emotion-aware"][10]["precision_mean"]
        diff  = emo_p - sem_p
        log.info(f"  {group:<20} {sem_p:>10.4f} {emo_p:>10.4f} {diff:>+12.4f}")
    log.info("")
    log.info("Phase 4 offline evaluation complete.")
    log.info(f"Full report: {REPORT_FILE}")