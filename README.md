## Overview

This repository contains the implementation and evaluation of an emotion-aware music recommendation system developed as part of an undergraduate special problem.

The study investigates whether incorporating emotional information into music tag embeddings improves recommendation quality compared to a semantic-only recommendation approach.

Three recommendation approaches are evaluated:

1. Popularity-based baseline
2. Semantic tag-based recommender
3. Emotion-aware tag-based recommender


### Phase 1 — Dataset Preparation

Processes the raw Last.fm dataset and prepares listening events and tag information.

**Outputs**

* Cleaned listening history
* Processed tag metadata

### Phase 2 — Tag Representation Learning

Generates:

* Semantic tag embeddings
* Emotion-aware tag embeddings
* Emotion vector resources

**Outputs**

* semantic_embeddings.pkl
* emotion_aware_embeddings.pkl
* emotion_vectors.pkl

### Phase 3 — Recommendation Generation

Builds track representations and generates recommendation candidates for:

* Popularity model
* Semantic model
* Emotion-aware model

**Outputs**

* recommendations_popularity.pkl
* recommendations_semantic.pkl
* recommendations_emotion.pkl

### Phase 4 — Offline Evaluation

Evaluates recommendation performance using:

* Precision@K
* Recall@K
* Emotion Alignment
* Stratified user analysis

**Outputs**

* evaluation_results.csv
* emotion_alignment.csv
* full_results_report.txt

### Phase 5 — User Study

Generates recommendation playlists for study participants and analyzes survey responses.

Components:

1. Playlist generation
2. Survey analysis

## Installation

### Requirements

Python 3.10+ recommended.

Install dependencies:

```bash
pip install -r requirements.txt
```

## Last.fm API Key

The user study recommendation generator requires a Last.fm API key.

Set the environment variable:

### Windows

```cmd
set LASTFM_API_KEY=YOUR_API_KEY
```

### Linux/macOS

```bash
export LASTFM_API_KEY=YOUR_API_KEY
```

## Running the Pipeline

Run phases sequentially:

```bash
python phase1.py
python phase2.py
python phase3.py
python phase4.py
```

For user study playlist generation:

```bash
python phase5_userstudy.py
```

For a specific participant:

```bash
python phase5_userstudy.py --participant PART-001
```

## Evaluation Metrics

### Recommendation Accuracy

* Precision@5
* Precision@10
* Precision@20
* Recall@5
* Recall@10
* Recall@20

### Emotion Alignment

Emotion alignment is measured using cosine similarity between:

* User emotional profile
* Recommended track emotional profile

Both are represented in a shared Warriner Valence–Arousal–Dominance (VAD) space.

## Reproducibility

To reproduce the reported results:

1. Run Phases 1–4 sequentially.
2. Ensure all intermediate output files are available.
3. Use the same dataset version and configuration parameters described in the thesis.

## Author

Bettina C. Ligero

Bachelor of Science in Computer Science
University of The Philippines Cebu

License

This repository is provided for academic and research purposes.

This repository is provided for academic and research purposes.
