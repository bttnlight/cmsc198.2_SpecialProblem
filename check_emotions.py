import pickle

with open("phase2_output/emotion_vectors.pkl", "rb") as f:
    emotion_vectors = pickle.load(f)

emotions = ["anger","fear","anticipation","trust","surprise","sadness","joy","disgust"]

# Show tags with non-zero emotion vectors
emotional_tags = {tag: vec for tag, vec in emotion_vectors.items() 
                  if vec.sum() > 0}

print(f"Tags with emotional signal: {len(emotional_tags)}")
print("\nSample emotional tags:")
for tag, vec in list(emotional_tags.items())[:20]:
    active = [emotions[i] for i, v in enumerate(vec) if v > 0]
    print(f"  {tag:<25} {active}")