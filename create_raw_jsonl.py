import json
from datetime import date

INPUT_FILE = "data/gbagyi/processed/Gbagyi_New_Testament_Bible_Order.txt"
OUTPUT_FILE = "data/gbagyi/raw/raw_data_group_08.jsonl"

# Replace this with the actual URL where the Gbagyi Bible was obtained.
SOURCE_URL = "https://preview.open.bible/bibles/68de939641a2a80e0f049a7d?utm_source=chatgpt.com"

retrieved = str(date.today())

with open(INPUT_FILE, "r", encoding="utf-8") as infile, \
     open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:

    for i, line in enumerate(infile, start=1):
        text = line.strip()

        if not text:
            continue

        record = {
            "id": f"gbagyi_nt_{i:06d}",
            "url": SOURCE_URL,
            "date_retrieved": retrieved,
            "raw_text": text
        }

        outfile.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"Created {OUTPUT_FILE}")

