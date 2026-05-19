"""
Download the real CTA ridership + station datasets from Chicago's Data Portal.

USAGE:
    python src/download_data.py

Run this ONCE locally. The datasets are public (no API key) and around
~40 MB total. After this finishes, the rest of the pipeline reads from
the local CSV files in `data/`.

Verified sources (Chicago Data Portal, May 2026):
    - Ridership : https://data.cityofchicago.org/Transportation/CTA-Ridership-L-Station-Entries-Daily-Totals/5neh-572f
    - Stations  : https://data.cityofchicago.org/Transportation/CTA-System-Information-List-of-L-Stops/8pix-ypme
"""
import sys
import urllib.request
from pathlib import Path

# Allow running this file directly: add src/ to path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_DIR, RAW_RIDERSHIP_CSV, RAW_STATIONS_CSV, RIDERSHIP_URL, STATIONS_URL


def download(url: str, dest: Path) -> None:
    """Stream a URL to disk with a basic progress indicator."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  -> {url}")
    print(f"     into {dest}")
    with urllib.request.urlopen(url) as response, open(dest, "wb") as out:
        # Stream in chunks so we don't hold the whole 40 MB in memory
        total = 0
        chunk = response.read(1024 * 1024)
        while chunk:
            out.write(chunk)
            total += len(chunk)
            print(f"     ... {total / 1_000_000:.1f} MB", end="\r")
            chunk = response.read(1024 * 1024)
    print(f"     ... {total / 1_000_000:.1f} MB  [done]")


def main() -> None:
    print("Downloading CTA datasets from Chicago Data Portal:")
    print()
    print("1/2  Daily ridership totals (~40 MB)")
    download(RIDERSHIP_URL, RAW_RIDERSHIP_CSV)
    print()
    print("2/2  Station info (~50 KB)")
    download(STATIONS_URL, RAW_STATIONS_CSV)
    print()
    print(f"All data saved to {DATA_DIR.resolve()}")


if __name__ == "__main__":
    main()
