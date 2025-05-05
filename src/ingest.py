import requests
import pathlib

# Resolve the raw directory relative to the script location
RAW = pathlib.Path(__file__).resolve().parent.parent / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

# Verified working setIds from DailyMed
setids = [
    "cbbd3dc0-6a39-48f7-91b9-a95c6c12ee0c",  # ABILIFY
    "2f7d4d67-10c1-4bf4-a7f2-c185fbad64ba",  # CYMBALTA
    "c88f33ed-6dfb-4c5e-bc01-d8e36dd97299"   # PROZAC
]

for sid in setids:
    pdf_url = f"https://dailymed.nlm.nih.gov/dailymed/downloadpdffile.cfm?setId={sid}"
    fpath = RAW / f"{sid}.pdf"

    try:
        print(f"Downloading {fpath.name} ...")
        r = requests.get(pdf_url, timeout=30)
        r.raise_for_status()
        fpath.write_bytes(r.content)
        print(f"✅ Saved to {fpath}")
    except requests.RequestException as e:
        print(f"❌ Failed to download {pdf_url}: {e}")
