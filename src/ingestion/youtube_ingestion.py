import json
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


load_dotenv()

API_KEY = os.getenv("YOUTUBE_API_KEY")

BRONZE_DIR = Path("data/bronze/youtube")

VIDEO_IDS = [
    "dQw4w9WgXcQ",
    # Add more YouTube video IDs here
    # Example:
    # "VIDEO_ID_2",
    # "VIDEO_ID_3",
]


def get_youtube_client():
    if not API_KEY:
        raise ValueError("YOUTUBE_API_KEY not found. Add it to your .env file.")

    return build("youtube", "v3", developerKey=API_KEY)


def fetch_video_comments(youtube, video_id: str, max_pages: int = 3) -> list[dict]:
    comments = []
    next_page_token = None
    page_count = 0

    while page_count < max_pages:
        try:
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=100,
                pageToken=next_page_token,
                textFormat="plainText",
                order="relevance",
            )

            response = request.execute()

        except HttpError as error:
            print(f"Failed to fetch comments for video {video_id}: {error}")
            break

        for item in response.get("items", []):
            snippet = item["snippet"]["topLevelComment"]["snippet"]

            comments.append(
                {
                    "video_id": video_id,
                    "comment_id": item["snippet"]["topLevelComment"]["id"],
                    "author": snippet.get("authorDisplayName"),
                    "text": snippet.get("textDisplay"),
                    "likes": snippet.get("likeCount", 0),
                    "published_at": snippet.get("publishedAt"),
                    "updated_at": snippet.get("updatedAt"),
                    "source": "youtube",
                    "ingested_at": datetime.now(timezone.utc).isoformat(),
                }
            )

        next_page_token = response.get("nextPageToken")
        page_count += 1

        if not next_page_token:
            break

    return comments


def save_comments_to_bronze(video_id: str, comments: list[dict]) -> Path:
    BRONZE_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = BRONZE_DIR / f"{video_id}_{timestamp}.json"

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(comments, file, indent=4, ensure_ascii=False)

    return output_path


def main() -> None:
    youtube = get_youtube_client()

    total_comments = 0

    for video_id in VIDEO_IDS:
        print(f"Fetching comments for video: {video_id}")

        comments = fetch_video_comments(
            youtube=youtube,
            video_id=video_id,
            max_pages=3,
        )

        output_path = save_comments_to_bronze(video_id, comments)

        print(f"Saved {len(comments)} comments to {output_path}")

        total_comments += len(comments)

    print(f"Finished ingestion. Total comments fetched: {total_comments}")


if __name__ == "__main__":
    main()