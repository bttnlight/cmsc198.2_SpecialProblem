"""
Section 4.7.4 Data Analysis

Implements:
  - Descriptive statistics (median, mode, range) per dimension per model
  - Comparative/categorical question frequencies
  - Likelihood to use feature (ordinal descriptives)

Output:
  - Prints results to terminal
  - Saves identical output to phase5_data_analysis_results.txt
"""

import pandas as pd
import numpy as np
import sys
from datetime import datetime

# ── CONFIG ────────────────────────────────────────────────────────────────────

FILE_PATH = "Stage_2__Playlist_Evaluation_Survey__Responses_.xlsx"
LOG_PATH  = "phase5_data_analysis_results.txt"

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
        " I would listen to these songs in my everyday life.",
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

def get_mode(series: pd.Series) -> list:
    """Return mode(s) as a sorted list."""
    return sorted(series.mode().tolist())


def mode_str(modes: list) -> str:
    """Format mode list as a slash-separated string."""
    return "/".join(str(int(m)) for m in modes)


# ── 1. DESCRIPTIVE STATISTICS (ORDINAL) ──────────────────────────────────────

print("=" * 70)
print("1. DESCRIPTIVE STATISTICS  (Median / Mode / Range)")
print("   Ordinal Likert data — mean and SD not reported")
print("=" * 70)
print(f"{'Dimension':<22}  {'Med_A':>5}  {'Mode_A':>6}  {'Rng_A':>5}    "
      f"{'Med_B':>5}  {'Mode_B':>6}  {'Rng_B':>5}")
print("-" * 70)

results = []

for dim, (col_a, col_b) in DIMENSIONS.items():
    a = df[col_a].dropna()
    b = df[col_b].dropna()
    idx = a.index.intersection(b.index)
    a, b = a.loc[idx], b.loc[idx]
    n = len(a)

    med_a   = a.median()
    mode_a  = get_mode(a)
    range_a = int(a.max()) - int(a.min())

    med_b   = b.median()
    mode_b  = get_mode(b)
    range_b = int(b.max()) - int(b.min())

    results.append({
        "Dimension": dim,
        "N":         n,
        "Median_A":  med_a,
        "Mode_A":    mode_a,
        "Range_A":   range_a,
        "Median_B":  med_b,
        "Mode_B":    mode_b,
        "Range_B":   range_b,
    })

    print(f"{dim:<22}  {med_a:>5.1f}  {mode_str(mode_a):>6}  {range_a:>5}    "
          f"{med_b:>5.1f}  {mode_str(mode_b):>6}  {range_b:>5}")

# ── 2. COMPARATIVE QUESTIONS ──────────────────────────────────────────────────

print("\n" + "=" * 70)
print("2. COMPARATIVE / CATEGORICAL QUESTION FREQUENCIES")
print("=" * 70)

for q in COMPARATIVE_QUESTIONS:
    counts = df[q].value_counts()
    pcts   = (counts / counts.sum() * 100).round(1)
    print(f"\n{q}")
    print("-" * 50)
    for option in counts.index:
        print(f"  {option:<40} {counts[option]:>3}  ({pcts[option]:.1f}%)")

# ── 3. LIKELIHOOD TO USE FEATURE ─────────────────────────────────────────────

print("\n" + "=" * 70)
print("3. LIKELIHOOD TO USE AUTOMATIC PLAYLIST FEATURE")
print("   Ordinal — Median, Mode, and Range reported")
print("=" * 70)
lu = df[LIKELIHOOD_COL].dropna()
print(f"  N      : {len(lu)}")
print(f"  Median : {lu.median():.1f}")
print(f"  Mode   : {mode_str(sorted(lu.mode().tolist()))}")
print(f"  Min    : {int(lu.min())}")
print(f"  Max    : {int(lu.max())}")
print(f"  Range  : {int(lu.max()) - int(lu.min())}")

# ── 4. FULL SUMMARY TABLE ─────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("4. FULL SUMMARY TABLE")
print("=" * 70)
summary_rows = []
for r in results:
    summary_rows.append({
        "Dimension": r["Dimension"],
        "N":         r["N"],
        "Median_A":  r["Median_A"],
        "Mode_A":    mode_str(r["Mode_A"]),
        "Range_A":   r["Range_A"],
        "Median_B":  r["Median_B"],
        "Mode_B":    mode_str(r["Mode_B"]),
        "Range_B":   r["Range_B"],
    })
summary = pd.DataFrame(summary_rows)
print(summary.to_string(index=False))

# ── CLOSE LOG ─────────────────────────────────────────────────────────────────

sys.stdout = sys.__stdout__
log_file.close()
print(f"\nResults saved to: {LOG_PATH}")