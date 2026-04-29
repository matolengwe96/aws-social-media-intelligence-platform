import json
import re
from pathlib import Path

import pandas as pd


BRONZE_DIR = Path("data/bronze/youtube")
SILVER_DIR = Path("data/silver/youtube")


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_bronze_comments() -> pd.DataFrame:
    records = []

    for file_path in BRONZE_DIR.glob("*.json"):
        video_id = file_path.stem

        with open(file_path, "r", encoding="utf-8") as file:
            comments = json.load(file)

        for comment in comments:
            records.append(
                {
                    "video_id": video_id,
                    "author": comment.get("author"),
                    "raw_text": comment.get("text"),
                    "clean_text": clean_text(comment.get("text")),
                    "likes": comment.get("likes", 0),
                    "published_at": comment.get("published_at"),
                    "source_file": file_path.name,
                }
            )

    return pd.DataFrame(records)


def save_silver(df: pd.DataFrame) -> None:
    SILVER_DIR.mkdir(parents=True, exist_ok=True)

    output_csv = SILVER_DIR / "youtube_comments_cleaned.csv"
    output_parquet = SILVER_DIR / "youtube_comments_cleaned.parquet"

    df.to_csv(output_csv, index=False)
    df.to_parquet(output_parquet, index=False)

    print(f"Saved CSV: {output_csv}")
    print(f"Saved Parquet: {output_parquet}")


if __name__ == "__main__":
    print("Loading bronze YouTube comments...")
    df = load_bronze_comments()

    if df.empty:
        raise ValueError("No bronze YouTube comment data found.")

    print(f"Loaded {len(df)} comments")
    print("Cleaning and saving silver layer...")

    save_silver(df)

    print("Silver transformation completed")