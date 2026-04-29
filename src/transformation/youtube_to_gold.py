import pandas as pd
from pathlib import Path
from textblob import TextBlob

SILVER_PATH = Path("data/silver/youtube/youtube_comments_cleaned.parquet")
GOLD_DIR = Path("data/gold/youtube")


def get_sentiment(text):
    try:
        return TextBlob(text).sentiment.polarity
    except:
        return 0


def categorize_sentiment(score):
    if score > 0:
        return "positive"
    elif score < 0:
        return "negative"
    return "neutral"


def transform_to_gold():
    df = pd.read_parquet(SILVER_PATH)

    print(f"Loaded {len(df)} records from silver")

    # sentiment scoring
    df["sentiment_score"] = df["clean_text"].apply(get_sentiment)
    df["sentiment_label"] = df["sentiment_score"].apply(categorize_sentiment)

    # aggregate metrics
    summary = df.groupby("sentiment_label").agg(
        total_comments=("clean_text", "count"),
        avg_likes=("likes", "mean")
    ).reset_index()

    # save outputs
    GOLD_DIR.mkdir(parents=True, exist_ok=True)

    df.to_parquet(GOLD_DIR / "youtube_comments_enriched.parquet", index=False)
    summary.to_csv(GOLD_DIR / "youtube_sentiment_summary.csv", index=False)

    print("Gold layer created")


if __name__ == "__main__":
    transform_to_gold()