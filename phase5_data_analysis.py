"""
Section 4.7.4 Data Analysis

Implements:
  - Descriptive statistics (mean, SD) per dimension per model
  - Shapiro-Wilk normality test on difference scores
  - Wilcoxon signed-rank test (primary, non-parametric)
  - Paired-sample t-test (supplementary)
  - Cohen's d effect sizes
  - Comparative/categorical question frequencies

Output:
  - Prints results to terminal
  - Saves identical output to  data_analysis_results.txt
"""

import pandas as pd
import numpy as np
from scipy import stats
import sys
from datetime import datetime

# ── CONFIG ────────────────────────────────────────────────────────────────────

FILE_PATH = "Stage_2__Playlist_Evaluation_Survey__Responses_.xlsx"
LOG_PATH  = "phase5_data_analysis_results.txt"
ALPHA     = 0.05

# Column mappings: dimension → (Playlist A column, Playlist B column)
DIMENSIONS = {
    "Perceived Relevance": (
        "The songs in Playlist A match my musical taste.",
        "The songs in Playlist B match my musical taste.",
    ),
    "Emotional Alignment": (
        "Playlist A reflects my current emotional state or mood.",
        "Playlist B reflects my current emotional state or mood.",
    ),
    "User Satisfaction": (
        "I am satisfied with the overall quality of Playlist A.",
        "I am satisfied with the overall quality of Playlist B.",
    ),
    "Everyday Listening": (
        " I would listen to these songs in my everyday life.",   # note leading space in raw data
        "I would listen to these songs in my everyday life.",
    ),
    "Emotional Coherence": (
        "The songs in Playlist A feel emotionally coherent — they belong together.",
        "The songs in Playlist B feel emotionally coherent — they belong together.",
    ),
}

COMPARATIVE_QUESTIONS = [
    "Overall, which playlist did you prefer?",
    "Which playlist felt more emotionally appropriate for your current mood?",
    "Which playlist felt more varied and interesting?",
    "Did the two playlists feel noticeably different from each other?",
    "Which playlist do you feel better understood your taste in music — beyond just mood?",
]

LIKELIHOOD_COL = (
    "If a music app generated these playlists automatically, "
    "how likely would you be to use such a feature?"
)

# ── LOGGING SETUP ─────────────────────────────────────────────────────────────
# Tee: mirrors every print() to both the terminal and the log file

class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()

log_file   = open(LOG_PATH, "w", encoding="utf-8")
sys.stdout = Tee(sys.__stdout__, log_file)

# ── HEADER ────────────────────────────────────────────────────────────────────

print("=" * 70)
print("  SECTION 4.7.4 — DATA ANALYSIS RESULTS")
print(f"  Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  Input     : {FILE_PATH}")
print(f"  Log saved : {LOG_PATH}")
print("=" * 70)
print()

# ── LOAD DATA ─────────────────────────────────────────────────────────────────

df = pd.read_excel(FILE_PATH)
print(f"Loaded {len(df)} responses.\n")

# ── HELPER ────────────────────────────────────────────────────────────────────

def effect_label(d: float) -> str:
    d = abs(d)
    if d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    return "large"

# ── 1. DESCRIPTIVE STATISTICS ────────────────────────────────────────────────

print("=" * 70)
print("1. DESCRIPTIVE STATISTICS")
print("=" * 70)
print(f"{'Dimension':<22} {'Mean_A':>7} {'SD_A':>6} {'Mean_B':>7} {'SD_B':>6} {'Diff(B-A)':>10}")
print("-" * 70)

results = []

for dim, (col_a, col_b) in DIMENSIONS.items():
    a = df[col_a].dropna()
    b = df[col_b].dropna()
    idx = a.index.intersection(b.index)
    a, b = a.loc[idx], b.loc[idx]
    diff = b - a
    n = len(a)

    mean_a, sd_a = a.mean(), a.std(ddof=1)
    mean_b, sd_b = b.mean(), b.std(ddof=1)

    # Normality (Shapiro-Wilk on difference scores)
    sw_stat, sw_p = stats.shapiro(diff)
    normal = sw_p > ALPHA

    # Paired t-test
    t_stat, p_t = stats.ttest_rel(a, b)

    # Wilcoxon signed-rank
    w_stat, p_w = stats.wilcoxon(a, b, alternative="two-sided")

    # Cohen's d (from difference scores)
    cohens_d = diff.mean() / diff.std(ddof=1)

    results.append({
        "Dimension":   dim,
        "N":           n,
        "Mean_A":      round(mean_a, 3),
        "SD_A":        round(sd_a, 3),
        "Mean_B":      round(mean_b, 3),
        "SD_B":        round(sd_b, 3),
        "Mean_diff":   round(diff.mean(), 3),
        "SW_stat":     round(sw_stat, 4),
        "SW_p":        round(sw_p, 4),
        "Normal":      normal,
        "t_stat":      round(t_stat, 4),
        "p_t":         round(p_t, 4),
        "Sig_t":       p_t < ALPHA,
        "W_stat":      round(w_stat, 4),
        "p_w":         round(p_w, 4),
        "Sig_w":       p_w < ALPHA,
        "Cohens_d":    round(cohens_d, 4),
        "Effect_size": effect_label(cohens_d),
    })

    print(f"{dim:<22} {mean_a:>7.3f} {sd_a:>6.3f} {mean_b:>7.3f} {sd_b:>6.3f} {diff.mean():>+10.3f}")

# ── 2. NORMALITY TEST ─────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("2. NORMALITY TEST (Shapiro-Wilk on difference scores)")
print("=" * 70)
print(f"{'Dimension':<22} {'W':>8} {'p':>8}   Normal?")
print("-" * 70)
for r in results:
    flag = "Yes" if r["Normal"] else "No  (-> use Wilcoxon)"
    print(f"{r['Dimension']:<22} {r['SW_stat']:>8.4f} {r['SW_p']:>8.4f}   {flag}")

# ── 3. WILCOXON SIGNED-RANK (PRIMARY) ────────────────────────────────────────

print("\n" + "=" * 70)
print("3. WILCOXON SIGNED-RANK TEST  (Primary — Non-Parametric)")
print("=" * 70)
print(f"{'Dimension':<22} {'W':>8} {'p':>8} {'Sig?':>8}")
print("-" * 70)
for r in results:
    sig = "YES *" if r["Sig_w"] else "No"
    print(f"{r['Dimension']:<22} {r['W_stat']:>8.1f} {r['p_w']:>8.4f} {sig:>8}")

# ── 4. PAIRED T-TEST (SUPPLEMENTARY) ─────────────────────────────────────────

print("\n" + "=" * 70)
print("4. PAIRED-SAMPLE T-TEST  (Supplementary)")
print("=" * 70)
print(f"{'Dimension':<22} {'t':>8} {'p':>8} {'Sig?':>8}")
print("-" * 70)
for r in results:
    sig = "YES *" if r["Sig_t"] else "No"
    print(f"{r['Dimension']:<22} {r['t_stat']:>8.4f} {r['p_t']:>8.4f} {sig:>8}")

# ── 5. EFFECT SIZES ───────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("5. EFFECT SIZES  (Cohen's d from paired difference scores)")
print("=" * 70)
print(f"{'Dimension':<22} {'d':>8} {'Interpretation':>16}")
print("-" * 70)
for r in results:
    print(f"{r['Dimension']:<22} {r['Cohens_d']:>8.4f} {r['Effect_size']:>16}")

# ── 6. COMPARATIVE QUESTIONS ──────────────────────────────────────────────────

print("\n" + "=" * 70)
print("6. COMPARATIVE / CATEGORICAL QUESTION FREQUENCIES")
print("=" * 70)
for q in COMPARATIVE_QUESTIONS:
    counts = df[q].value_counts()
    pcts   = (counts / counts.sum() * 100).round(1)
    print(f"\n{q}")
    print("-" * 50)
    for option in counts.index:
        print(f"  {option:<40} {counts[option]:>3}  ({pcts[option]:.1f}%)")

# ── 7. LIKELIHOOD TO USE FEATURE ─────────────────────────────────────────────

print("\n" + "=" * 70)
print("7. LIKELIHOOD TO USE AUTOMATIC PLAYLIST FEATURE")
print("=" * 70)
lu = df[LIKELIHOOD_COL].dropna()
print(f"  N      : {len(lu)}")
print(f"  Mean   : {lu.mean():.3f}")
print(f"  SD     : {lu.std(ddof=1):.3f}")
print(f"  Min    : {int(lu.min())}")
print(f"  Max    : {int(lu.max())}")
print(f"  Median : {lu.median():.1f}")

# ── 8. FULL SUMMARY TABLE ─────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("8. FULL SUMMARY TABLE")
print("=" * 70)
summary = pd.DataFrame(results)[[
    "Dimension", "N", "Mean_A", "SD_A", "Mean_B", "SD_B",
    "Mean_diff", "SW_p", "Normal", "W_stat", "p_w", "Sig_w",
    "t_stat", "p_t", "Sig_t", "Cohens_d", "Effect_size"
]]
print(summary.to_string(index=False))

# ── CLOSE LOG ─────────────────────────────────────────────────────────────────

sys.stdout = sys.__stdout__          # restore normal stdout
log_file.close()
print(f"\nResults saved to: {LOG_PATH}")