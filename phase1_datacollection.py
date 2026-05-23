"""
Phase 1 — Data Collection and Preprocessing
============================================

Usage
-----
  1. pip install requests tqdm
  2. Set your API_KEY and OUTPUT_DIR below (or via environment variables).
  3. python phase1_datacollection.py
  4. If interrupted, just run again — it will skip already-collected users automatically.
"""

# ────────────────────────────────────────────────
# 0.  Imports and configuration
# ────────────────────────────────────────────────
import requests
import time
import json
import csv
import os
import sys
import logging
import signal
from collections import Counter
from datetime import datetime

try:
    from tqdm import tqdm
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "tqdm", "-q"], check=True)
    from tqdm import tqdm

# ── API / sampling settings ──────────────────────
API_KEY                  = os.getenv("LASTFM_API_KEY", "f4f4b05dcb5e7186c9762534a06a4bda")
BASE_URL                 = "http://ws.audioscrobbler.com/2.0/"
DELAY                    = 0.25          # seconds between API calls
TARGET_USERS             = 1000
MIN_LISTENING_EVENTS     = 100
SEED_MIN_EVENTS          = 1000
SEED_MIN_FRIENDS         = 10
MIN_TAG_FREQUENCY        = 5
DUPLICATE_WINDOW_SECONDS = 300           # 5-minute dedup window
MAX_PAGES_PER_USER       = 2             # ~200 tracks per user
CACHE_FLUSH_INTERVAL     = 25            # flush tag cache every N users

# ── Paths ────────────────────────────────────────
OUTPUT_DIR       = os.getenv("PHASE1_OUTPUT_DIR", os.path.join(os.path.dirname(__file__), "phase1_output"))
os.makedirs(OUTPUT_DIR, exist_ok=True)

CACHE_FILE        = os.path.join(OUTPUT_DIR, "tag_cache.json")
USERS_FILE        = os.path.join(OUTPUT_DIR, "eligible_users.txt")
DONE_FILE         = os.path.join(OUTPUT_DIR, "done_users.txt")        
EVENTS_CSV        = os.path.join(OUTPUT_DIR, "raw_events.csv")
PREPROCESSED_CSV  = os.path.join(OUTPUT_DIR, "preprocessed_events.csv")
VALIDATED_CSV     = os.path.join(OUTPUT_DIR, "validated_events.csv")
CLEAN_CSV         = os.path.join(OUTPUT_DIR, "clean_events.csv")
LOG_FILE          = os.path.join(OUTPUT_DIR, "progress.log")

# ── Logging (file + console) ─────────────────────
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
log.info("Phase 1 — Data Collection and Preprocessing")
log.info(f"  Output dir       : {OUTPUT_DIR}")
log.info(f"  Target users     : {TARGET_USERS}")
log.info(f"  Min events/user  : {MIN_LISTENING_EVENTS}")
log.info(f"  Min tag freq     : {MIN_TAG_FREQUENCY}")
log.info(f"  Max pages/user   : {MAX_PAGES_PER_USER} (~{MAX_PAGES_PER_USER * 200} tracks)")
log.info("=" * 60)


# ────────────────────────────────────────────────
# 1.  CSV helpers
# ────────────────────────────────────────────────
CSV_FIELDS = ["user", "track", "artist", "timestamp", "tags", "tag_source"]


def _ensure_csv_header(path: str):
    """Write CSV header if the file does not yet exist."""
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()


def append_to_csv(path: str, events: list[dict]):
    """Append events to a CSV, writing the header first if needed."""
    _ensure_csv_header(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        for ev in events:
            row = dict(ev)
            row["tags"] = "|".join(ev["tags"])
            writer.writerow(row)


def load_csv_events(path: str) -> list[dict]:
    """Load events from a CSV produced by this script."""
    events = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            events.append({
                "user":       row["user"],
                "track":      row["track"],
                "artist":     row["artist"],
                "timestamp":  int(row["timestamp"]),
                "tags":       [t for t in row["tags"].split("|") if t],
                "tag_source": row["tag_source"],
            })
    return events


def save_csv_events(path: str, events: list[dict]):
    """Write a full list of events to a CSV (overwrites)."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for ev in events:
            row = dict(ev)
            row["tags"] = "|".join(ev["tags"])
            writer.writerow(row)


# ────────────────────────────────────────────────
# 2.  Done-user registry  (core resume mechanism)
# ────────────────────────────────────────────────

def load_done_users() -> set[str]:
    """Return the set of users whose events have already been saved."""
    if not os.path.exists(DONE_FILE):
        return set()
    with open(DONE_FILE, encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def mark_user_done(username: str):
    """Append a username to the done registry (one line per user)."""
    with open(DONE_FILE, "a", encoding="utf-8") as f:
        f.write(username + "\n")


# ────────────────────────────────────────────────
# 3.  Tag cache
# ────────────────────────────────────────────────

def load_tag_cache() -> dict:
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_tag_cache(cache: dict):
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f)


tag_cache = load_tag_cache()
log.info(f"Tag cache loaded: {len(tag_cache):,} entries")


# ────────────────────────────────────────────────
# 4.  Last.fm API helpers  (Section 4.3.1)
# ────────────────────────────────────────────────

def call_api(params: dict, retries: int = 3) -> dict | None:
    params = dict(params)
    params.update({"api_key": API_KEY, "format": "json"})
    for attempt in range(retries):
        try:
            r = requests.get(BASE_URL, params=params, timeout=10)
            r.raise_for_status()
            data = r.json()
            if "error" in data:
                return None
            return data
        except requests.RequestException:
            wait = 2 ** attempt
            log.debug(f"API error (attempt {attempt + 1}/{retries}), retrying in {wait}s…")
            time.sleep(wait)
    return None


def get_user_info(username: str) -> dict | None:
    return call_api({"method": "user.getInfo", "user": username})


def get_user_friends(username: str, limit: int = 50) -> list[str]:
    data = call_api({"method": "user.getFriends", "user": username, "limit": limit})
    if not data:
        return []
    friends = data.get("friends", {}).get("user", [])
    if isinstance(friends, dict):          # single-friend edge case
        friends = [friends]
    return [f["name"] for f in friends if isinstance(f, dict) and "name" in f]


def get_user_recent_tracks(username: str, limit: int = 200, page: int = 1) -> list[dict]:
    data = call_api({
        "method":   "user.getRecentTracks",
        "user":     username,
        "limit":    limit,
        "page":     page,
        "extended": 0,
    })
    if not data:
        return []
    tracks = data.get("recenttracks", {}).get("track", [])
    return [t for t in tracks if isinstance(t, dict) and "date" in t]


def get_track_top_tags(artist: str, track: str) -> list[str]:
    time.sleep(DELAY)
    data = call_api({"method": "track.getTopTags", "artist": artist, "track": track})
    if not data:
        return []
    tags = data.get("toptags", {}).get("tag", [])
    return [t["name"] for t in tags if isinstance(t, dict) and "name" in t]


def get_artist_top_tags(artist: str) -> list[str]:
    time.sleep(DELAY)
    data = call_api({"method": "artist.getTopTags", "artist": artist})
    if not data:
        return []
    tags = data.get("toptags", {}).get("tag", [])
    return [t["name"] for t in tags if isinstance(t, dict) and "name" in t]


def get_tags_for_track(artist: str, track: str) -> tuple[list[str], str]:
    """Fetch tags with caching + artist fallback. Returns (tags, source)."""
    track_key  = f"track:{artist.lower()}||{track.lower()}"
    artist_key = f"artist:{artist.lower()}"

    if track_key in tag_cache:
        return tag_cache[track_key], "track"
    if artist_key in tag_cache:
        return tag_cache[artist_key], "artist"

    tags = get_track_top_tags(artist, track)
    if tags:
        tag_cache[track_key] = tags
        return tags, "track"

    tags = get_artist_top_tags(artist)
    tag_cache[artist_key] = tags
    return tags, "artist"


# ────────────────────────────────────────────────
# 5.  User sampling — snowball  (Section 4.3.2)
# ────────────────────────────────────────────────

def count_user_events(username: str) -> int:
    data = call_api({"method": "user.getRecentTracks", "user": username, "limit": 1})
    if not data:
        return 0
    attr = data.get("recenttracks", {}).get("@attr", {})
    return int(attr.get("total", 0))


def load_eligible_users() -> list[str]:
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, encoding="utf-8") as f:
            users = [line.strip() for line in f if line.strip()]
        log.info(f"Eligible-user list loaded from disk: {len(users):,} users")
        return users
    return []


def save_eligible_users(users: list[str]):
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(users) + "\n")


def run_snowball_sampling(seed_user: str) -> list[str]:
    """
    BFS snowball from seed_user via user.getFriends until TARGET_USERS
    eligible users are collected. Eligibility = >= MIN_LISTENING_EVENTS.
    Saves incremental progress so a restart only re-checks unchecked users.
    """
    existing = load_eligible_users()
    if len(existing) >= TARGET_USERS:
        log.info(f"Already have {len(existing):,} eligible users — skipping sampling.")
        return existing

    eligible   = list(existing)
    eligible_s = set(eligible)
    queue      = [seed_user]
    visited    = set(existing) | {seed_user}

    log.info(f"Starting snowball from seed '{seed_user}' "
             f"(need {TARGET_USERS - len(eligible):,} more users)…")

    with tqdm(total=TARGET_USERS, initial=len(eligible), desc="Sampling users") as pbar:
        while queue and len(eligible) < TARGET_USERS:
            user = queue.pop(0)
            n    = count_user_events(user)
            time.sleep(DELAY)
            if n >= MIN_LISTENING_EVENTS:
                if user not in eligible_s:
                    eligible.append(user)
                    eligible_s.add(user)
                    pbar.update(1)
                    if len(eligible) % 50 == 0:
                        save_eligible_users(eligible)
                        log.info(f"  Checkpoint: {len(eligible)} eligible users saved.")

            friends = get_user_friends(user)
            time.sleep(DELAY)
            for f in friends:
                if f not in visited:
                    visited.add(f)
                    queue.append(f)

    save_eligible_users(eligible)
    log.info(f"Snowball complete. {len(eligible):,} eligible users saved to {USERS_FILE}")
    return eligible


# ────────────────────────────────────────────────
# 6.  Listening event construction  (Section 4.3.3)
# ────────────────────────────────────────────────

def build_listening_events(username: str) -> list[dict]:
    """
    Fetch up to MAX_PAGES_PER_USER pages of recent tracks for a user
    and return a list of event dicts matching the tuple (u, t, a, τ, T).
    """
    events = []
    for page in range(1, MAX_PAGES_PER_USER + 1):
        tracks = get_user_recent_tracks(username, limit=200, page=page)
        time.sleep(DELAY)
        if not tracks:
            break
        for t in tracks:
            artist = (t.get("artist") or {}).get("#text", "").strip()
            track  = (t.get("name") or "").strip()
            ts     = int(t.get("date", {}).get("uts", 0))
            if not artist or not track or not ts:
                continue
            tags, source = get_tags_for_track(artist, track)
            events.append({
                "user":       username,
                "track":      track,
                "artist":     artist,
                "timestamp":  ts,
                "tags":       tags,
                "tag_source": source,
            })
    return events


# ────────────────────────────────────────────────
# 7.  Main collection loop  (Section 4.3)
# ────────────────────────────────────────────────

def run_collection(eligible_users: list[str]):
    """
    Collect listening events for every eligible user.
    Each user is saved immediately after collection and registered in
    done_users.txt, so the loop can be safely interrupted and resumed.
    """
    done_users = load_done_users()
    remaining  = [u for u in eligible_users if u not in done_users]

    log.info(f"Collection status:")
    log.info(f"  Total eligible : {len(eligible_users):,}")
    log.info(f"  Already done   : {len(done_users):,}")
    log.info(f"  Remaining      : {len(remaining):,}")

    if not remaining:
        log.info("All users already collected. Skipping to preprocessing.")
        return

    # Ensure CSV header exists before appending
    _ensure_csv_header(EVENTS_CSV)

    def _flush_cache_and_exit(sig, frame):
        log.info("\nInterrupt received — flushing tag cache…")
        save_tag_cache(tag_cache)
        log.info(f"Tag cache saved ({len(tag_cache):,} entries). "
                 f"Re-run the script to continue from where you left off.")
        sys.exit(0)

    signal.signal(signal.SIGINT, _flush_cache_and_exit)
    signal.signal(signal.SIGTERM, _flush_cache_and_exit)

    for i, user in enumerate(tqdm(remaining, desc="Collecting events", unit="user")):
        try:
            events = build_listening_events(user)

            # ── Save immediately after each user ──────────────────────
            append_to_csv(EVENTS_CSV, events)          # append to CSV
            mark_user_done(user)                       # register as done

            log.info(f"  [{len(done_users) + i + 1}/{len(eligible_users)}] "
                     f"{user}: {len(events):,} events saved")

            # Flush tag cache periodically so crashes don't lose API calls
            if (i + 1) % CACHE_FLUSH_INTERVAL == 0:
                save_tag_cache(tag_cache)
                log.info(f"  Tag cache flushed ({len(tag_cache):,} entries).")

        except Exception as exc:
            log.warning(f"  Failed to collect {user}: {exc} — skipping.")

    save_tag_cache(tag_cache)
    log.info(f"Collection complete. Tag cache saved ({len(tag_cache):,} entries).")


# ────────────────────────────────────────────────
# 8.  Tag preprocessing  (Section 4.3.4)
# ────────────────────────────────────────────────

def run_preprocessing():
    if os.path.exists(PREPROCESSED_CSV):
        log.info(f"Preprocessed CSV already exists — loading from {PREPROCESSED_CSV}")
        return load_csv_events(PREPROCESSED_CSV)

    log.info("Loading raw events…")
    events = load_csv_events(EVENTS_CSV)
    log.info(f"  Loaded {len(events):,} raw events.")

    # Lowercase + dedup per event
    log.info("Normalizing tags (lowercase + dedup per track)…")
    for ev in events:
        seen, normalized = set(), []
        for tag in ev["tags"]:
            t = tag.lower().strip()
            if t and t not in seen:
                seen.add(t)
                normalized.append(t)
        ev["tags"] = normalized

    # Global frequency filter
    freq = Counter(tag for ev in events for tag in ev["tags"])
    log.info(f"  Unique tags before filter : {len(freq):,}")
    log.info(f"  Tags with freq >= {MIN_TAG_FREQUENCY}       : "
             f"{sum(1 for v in freq.values() if v >= MIN_TAG_FREQUENCY):,}")

    log.info(f"Filtering tags with freq < {MIN_TAG_FREQUENCY}…")
    for ev in events:
        ev["tags"] = [t for t in ev["tags"] if freq.get(t, 0) >= MIN_TAG_FREQUENCY]

    save_csv_events(PREPROCESSED_CSV, events)
    log.info(f"Preprocessed events saved: {PREPROCESSED_CSV}")
    log.info(f"  With tags    : {sum(1 for e in events if e['tags']):,}")
    log.info(f"  Without tags : {sum(1 for e in events if not e['tags']):,}")
    return events


# ────────────────────────────────────────────────
# 9.  Data quality validation  (Section 4.3.5)
# ────────────────────────────────────────────────

def run_validation(events: list[dict]) -> list[dict]:
    if os.path.exists(VALIDATED_CSV):
        log.info(f"Validated CSV already exists — loading from {VALIDATED_CSV}")
        return load_csv_events(VALIDATED_CSV)

    # Remove invalid timestamps
    before  = len(events)
    events  = [e for e in events if e.get("timestamp") and int(e["timestamp"]) > 0]
    removed = before - len(events)
    log.info(f"Timestamp validation: removed {removed:,} | retained {len(events):,}")

    # Dedup within 5-minute window per user
    events_sorted = sorted(events, key=lambda e: (e["user"], e["timestamp"]))
    dedup, skip_count, last = [], 0, {}
    for ev in events_sorted:
        key = (ev["user"], ev["track"].lower(), ev["artist"].lower())
        ts  = ev["timestamp"]
        if key in last and (ts - last[key]) <= DUPLICATE_WINDOW_SECONDS:
            skip_count += 1
        else:
            last[key] = ts
            dedup.append(ev)

    log.info(f"Dedup ({DUPLICATE_WINDOW_SECONDS}s window): removed {skip_count:,} | "
             f"retained {len(dedup):,}")

    save_csv_events(VALIDATED_CSV, dedup)
    log.info(f"Validated events saved: {VALIDATED_CSV}")
    return dedup


# ────────────────────────────────────────────────
# 10.  Summary and final export
# ────────────────────────────────────────────────

def print_summary(events: list[dict]):
    user_counts   = Counter(e["user"] for e in events)
    all_tags_flat = [t for e in events for t in e["tags"]]
    unique_tracks = {(e["track"].lower(), e["artist"].lower()) for e in events}
    counts        = sorted(user_counts.values())
    mid           = counts[len(counts) // 2] if counts else 0

    log.info("=" * 50)
    log.info("PHASE 1 DATASET SUMMARY")
    log.info("=" * 50)
    log.info(f"  Total users              : {len(user_counts):,}")
    log.info(f"  Total listening events   : {len(events):,}")
    log.info(f"  Unique tracks            : {len(unique_tracks):,}")
    log.info(f"  Unique tags (post-filter): {len(set(all_tags_flat)):,}")
    log.info(f"  Avg events per user      : {len(events) / max(len(user_counts), 1):.1f}")
    log.info(f"  Min / Median / Max events: "
             f"{counts[0] if counts else 0} / {mid} / {counts[-1] if counts else 0}")
    log.info("")
    log.info("  Top 10 tags:")
    for tag, cnt in Counter(all_tags_flat).most_common(10):
        log.info(f"    {tag:<30} {cnt:,}")

    save_csv_events(CLEAN_CSV, events)
    log.info(f"\nFinal clean dataset saved: {CLEAN_CSV}")
    log.info("Phase 1 complete. Ready for Phase 2.")


# ────────────────────────────────────────────────
# 11.  Entry point
# ────────────────────────────────────────────────

if __name__ == "__main__":
    # ── Step 1: Snowball sampling ─────────────────
    # Change SEED_USER to any publicly active Last.fm account you want to start from.
    SEED_USER    = "RJ"           # Last.fm founder — large, public friend network
    eligible     = load_eligible_users()
    if len(eligible) < TARGET_USERS:
        eligible = run_snowball_sampling(SEED_USER)

    # ── Step 2: Collect listening events ─────────
    run_collection(eligible)

    # ── Step 3: Tag preprocessing ─────────────────
    events = run_preprocessing()

    # ── Step 4: Data quality validation ──────────
    events = run_validation(events)

    # ── Step 5: Summary ───────────────────────────
    print_summary(events)
