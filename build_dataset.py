"""
Dataset Builder

Combines real_reviews.txt and fake_reviews.txt into a labeled CSV
ready for fine-tuning.

"""

import csv
import random

REAL_FILE  = "real_reviews.txt"
FAKE_FILE  = "fake_reviews.txt"
OUTPUT_CSV = "reviews.csv"
SEED       = 42

random.seed(SEED)

def load_reviews(filepath: str) -> list[str]:
    with open(filepath, "r", encoding="utf-8") as f:
        reviews = [line.strip() for line in f if line.strip()]
    return reviews

real_reviews = load_reviews(REAL_FILE)
fake_reviews = load_reviews(FAKE_FILE)

print(f"📄 Loaded {len(real_reviews)} real reviews")
print(f"🤖 Loaded {len(fake_reviews)} fake reviews")

rows = [(text, 0) for text in real_reviews] + \
       [(text, 1) for text in fake_reviews]

random.shuffle(rows)

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["text", "label"])
    writer.writerows(rows)

print(f"\n✅ Saved {len(rows)} labeled reviews to '{OUTPUT_CSV}'")
print(f"   Real: {len(real_reviews)}  |  Fake: {len(fake_reviews)}")
print(f"\n── Preview (first 5 rows) ───────────────────")
for text, label in rows[:5]:
    preview = text[:70] + "..." if len(text) > 70 else text
    print(f"  [{'real' if label==0 else 'fake'}]  {preview}")
