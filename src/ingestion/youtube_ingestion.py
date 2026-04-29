import os
import json
from googleapiclient.discovery import build
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("YOUTUBE_API_KEY")

youtube = build('youtube', 'v3', developerKey=API_KEY)


def get_video_comments(video_id, max_results=100):
    comments = []

    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=max_results,
        textFormat="plainText"
    )

    response = request.execute()

    for item in response['items']:
        comment = item['snippet']['topLevelComment']['snippet']
        comments.append({
            "author": comment['authorDisplayName'],
            "text": comment['textDisplay'],
            "likes": comment['likeCount'],
            "published_at": comment['publishedAt']
        })

    return comments


def save_to_bronze(data, video_id):
    output_path = f"data/bronze/youtube/{video_id}.json"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    video_id = "dQw4w9WgXcQ"

    print("Fetching comments...")
    comments = get_video_comments(video_id)

    print(f"Fetched {len(comments)} comments")

    save_to_bronze(comments, video_id)

    print("Saved to bronze layer")