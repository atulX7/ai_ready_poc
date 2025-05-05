import re, pathlib, json
from collections import defaultdict
import random
import textstat
import tiktoken
from spellchecker import SpellChecker

spell = SpellChecker()
TOK = tiktoken.get_encoding("cl100k_base")
PROCESSED = pathlib.Path("data/processed")

# Intermediate storage for grouping by original file
doc_scores = defaultdict(list)

def detect_pii(text):
    pii_patterns = [
        r"\b\d{3}[-.\s]??\d{2}[-.\s]??\d{4}\b",
        r"\b(?:\+?\d{1,3})?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    ]
    return any(re.search(p, text) for p in pii_patterns)

# Process all chunks
for file in PROCESSED.glob("*.txt"):
    content = file.read_text()
    tokens = TOK.encode(content)
    word_list = content.split()
    misspelled = spell.unknown(word_list[:500])

    completeness = 1.0 if len(content.strip()) > 100 else 0.0
    accuracy = 1.0 - (len(misspelled) / max(len(word_list), 1))
    secure = 0.0 if detect_pii(content) else 1.0
    quality = min(1.0, max(0.0, textstat.flesch_reading_ease(content) / 100.0))
    timeliness = 1.0 if "2023" in file.name or random.random() > 0.5 else 0.5

    doc_id = file.stem.split("_")[0] + ".pdf"  # collapse chunks back to original PDF

    doc_scores[doc_id].append({
        "completeness": completeness,
        "accuracy": accuracy,
        "secure": secure,
        "quality": quality,
        "timeliness": timeliness,
        "token_count": len(tokens)
    })

# Aggregate per document
final_scores = []
for doc, chunks in doc_scores.items():
    n = len(chunks)
    agg = {
        "file": doc,
        "completeness": round(sum(c["completeness"] for c in chunks) / n, 2),
        "accuracy": round(sum(c["accuracy"] for c in chunks) / n, 2),
        "secure": round(sum(c["secure"] for c in chunks) / n, 2),
        "quality": round(sum(c["quality"] for c in chunks) / n, 2),
        "timeliness": round(sum(c["timeliness"] for c in chunks) / n, 2),
        "token_count": sum(c["token_count"] for c in chunks)
    }
    agg["ai_trust_score"] = round(
        (agg["completeness"] + agg["accuracy"] + agg["secure"] + agg["quality"] + agg["timeliness"]) / 5.0, 2
    )
    final_scores.append(agg)

# Save and print
metrics_file = pathlib.Path("data/processed/metrics.json")
metrics_file.write_text(json.dumps(final_scores, indent=2))

print(f"\n✅ Processed {len(final_scores)} PDF files. AI Trust Scores:\n")
for score in final_scores:
    print(json.dumps(score, indent=2))
