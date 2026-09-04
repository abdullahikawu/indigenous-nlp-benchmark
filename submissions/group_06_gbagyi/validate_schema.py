# JSONL Schema and Sentence Coverage Check for Group 06
import json
import os
import re

DATA_PATH = "data/gbagyi/raw/raw_data_group_06.jsonl"
REQUIRED_KEYS = {"id", "url", "date_retrieved", "raw_text"}

def test_coverage():
    if not os.path.exists(DATA_PATH):
        print(f"[!] Path {DATA_PATH} not found yet. Ready for data injection.")
        return
    count = 0
    splitter = re.compile(r'(?<=[.!?])\s+')
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            assert REQUIRED_KEYS.issubset(record.keys())
            count += len([s for s in splitter.split(record["raw_text"]) if len(s.strip()) > 3])
    print(f"Total Sentences: {count} | Validated: {count >= 2500}")

if __name__ == "__main__":
    test_coverage()
