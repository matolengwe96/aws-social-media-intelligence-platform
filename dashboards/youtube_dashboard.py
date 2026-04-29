from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import os
import re
import time

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv


load_dotenv()

GOLD_COMMENTS_PATH = Path("data/gold/youtube/youtube_comments_enriched.parquet")
BRONZE_DIR = Path("data/bronze/youtube")
SAVED_QUERIES_PATH = Path("data/gold/youtube/saved_queries.csv")


st.set_page_config(
    page_title="SocialPulse AI",
    page_icon="📊",
    layout="wide",
)


st.markdown(
    """
    <style>
        .stMetric {
            background-color: #f8f9fa;
            padding: 14px;
            border-radius: 14px;
            border: 1px solid #e9ecef;
        }
        .block-container {
            padding-top: 2rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


USERS = {
    "admin": hash_password("admin123"),
    "yamkela": hash_password("socialpulse123"),
}


def login_screen() -> None:
    st.title("SocialPulse AI Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in USERS and USERS[username] == hash_password(password):
            st.session_state["logged_in"] = True
            st.session_state["username"] = username
            st.success("Login successful")
            st.rerun()
        else:
            st.error("Invalid username or password")

    st.info("Demo users: admin / admin123 OR yamkela / socialpulse123")


def require_login() -> None:
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False

    if not st.session_state["logged_in"]:
        login_screen()
        st.stop()


def logout_button() -> None:
    with st.sidebar:
        st.markdown(f"Logged in as **{st.session_state.get('username', 'user')}**")
        if st.button("Logout"):
            st.session_state["logged_in"] = False
            st.rerun()


def adjusted_average(series: pd.Series) -> float:
    if series.empty:
        return 0.0

    upper_limit = series.quantile(0.95)
    return round(series.clip(upper=upper_limit).mean(), 2)


def standardize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = {
        "video_id": "unknown",
        "author": "Unknown",
        "clean_text": "",
        "text": "",
        "likes": 0,
        "sentiment_label": "neutral",
        "sentiment_score": 0,
        "published_at": None,
    }

    for column, default_value in required_columns.items():
        if column not in df.columns:
            df[column] = default_value

    if "clean_text" not in df.columns or df["clean_text"].eq("").all():
        df["clean_text"] = df["text"].fillna("")

    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
    df["month"] = df["published_at"].dt.to_period("M").dt.to_timestamp()

    df["video_id"] = df["video_id"].fillna("unknown").astype(str)
    df["author"] = df["author"].fillna("Unknown").astype(str)
    df["clean_text"] = df["clean_text"].fillna("").astype(str)
    df["sentiment_label"] = df["sentiment_label"].fillna("neutral").astype(str)

    df["likes"] = pd.to_numeric(df["likes"], errors="coerce").fillna(0)
    df["sentiment_score"] = pd.to_numeric(df["sentiment_score"], errors="coerce").fillna(0)

    df["engagement_bucket"] = pd.cut(
        df["likes"],
        bins=[-1, 10, 100, 1000, 10000, float("inf")],
        labels=["Very Low", "Low", "Medium", "High", "Viral"],
    )

    return df


@st.cache_data(ttl=300)
def load_gold_data() -> pd.DataFrame:
    if not GOLD_COMMENTS_PATH.exists():
        return pd.DataFrame()

    df = pd.read_parquet(GOLD_COMMENTS_PATH)
    return standardize_dataframe(df)


def load_uploaded_csv(uploaded_file) -> pd.DataFrame:
    df = pd.read_csv(uploaded_file)
    return standardize_dataframe(df)


def simple_sentiment(text: str) -> tuple[str, float]:
    positive_words = {
        "good", "great", "amazing", "love", "excellent", "perfect", "wonderful",
        "best", "awesome", "nice", "happy", "cool", "beautiful", "strong"
    }

    negative_words = {
        "bad", "hate", "worst", "terrible", "awful", "angry", "annoying",
        "problem", "issue", "complain", "boring", "sad", "death", "risk"
    }

    words = set(re.findall(r"\b[a-zA-Z]{3,}\b", str(text).lower()))

    positive_count = len(words.intersection(positive_words))
    negative_count = len(words.intersection(negative_words))

    score = positive_count - negative_count

    if score > 0:
        return "positive", min(score / 3, 1)
    if score < 0:
        return "negative", max(score / 3, -1)

    return "neutral", 0


def fetch_live_youtube_comments(video_id: str, max_pages: int = 2) -> pd.DataFrame:
    api_key = os.getenv("YOUTUBE_API_KEY")

    if not api_key:
        st.error("YOUTUBE_API_KEY not found in your .env file.")
        return pd.DataFrame()

    try:
        from googleapiclient.discovery import build
        from googleapiclient.errors import HttpError
    except ImportError:
        st.error("Missing package. Run: pip install google-api-python-client")
        return pd.DataFrame()

    youtube = build("youtube", "v3", developerKey=api_key)

    comments = []
    next_page_token = None

    for _ in range(max_pages):
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
            st.error(f"YouTube API error: {error}")
            break

        for item in response.get("items", []):
            snippet = item["snippet"]["topLevelComment"]["snippet"]
            text = snippet.get("textDisplay", "")
            sentiment_label, sentiment_score = simple_sentiment(text)

            comments.append(
                {
                    "video_id": video_id,
                    "comment_id": item["snippet"]["topLevelComment"]["id"],
                    "author": snippet.get("authorDisplayName", "Unknown"),
                    "text": text,
                    "clean_text": text,
                    "likes": snippet.get("likeCount", 0),
                    "published_at": snippet.get("publishedAt"),
                    "updated_at": snippet.get("updatedAt"),
                    "source": "youtube_live_api",
                    "sentiment_label": sentiment_label,
                    "sentiment_score": sentiment_score,
                    "ingested_at": datetime.now(timezone.utc).isoformat(),
                }
            )

        next_page_token = response.get("nextPageToken")

        if not next_page_token:
            break

    if not comments:
        return pd.DataFrame()

    df = pd.DataFrame(comments)
    return standardize_dataframe(df)


def save_live_comments_to_bronze(df: pd.DataFrame, video_id: str) -> Path:
    BRONZE_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = BRONZE_DIR / f"{video_id}_live_{timestamp}.json"

    df.to_json(output_path, orient="records", indent=4, force_ascii=False)

    return output_path


def filter_data(
    df: pd.DataFrame,
    video_id: str,
    sentiment: str,
    min_likes: int,
    search_text: str,
) -> pd.DataFrame:
    filtered = df.copy()

    if video_id != "all":
        filtered = filtered[filtered["video_id"] == video_id]

    if sentiment != "all":
        filtered = filtered[filtered["sentiment_label"] == sentiment]

    filtered = filtered[filtered["likes"] >= min_likes]

    if search_text:
        filtered = filtered[
            filtered["clean_text"].str.contains(search_text, case=False, na=False)
        ]

    return filtered


def build_video_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    summary = (
        df.groupby("video_id")
        .agg(
            total_comments=("clean_text", "count"),
            total_likes=("likes", "sum"),
            avg_likes_including_viral=("likes", "mean"),
            avg_likes_typical=("likes", adjusted_average),
            avg_sentiment=("sentiment_score", "mean"),
            positive_comments=("sentiment_label", lambda x: (x == "positive").sum()),
            neutral_comments=("sentiment_label", lambda x: (x == "neutral").sum()),
            negative_comments=("sentiment_label", lambda x: (x == "negative").sum()),
        )
        .reset_index()
    )

    summary["positive_rate"] = round(
        (summary["positive_comments"] / summary["total_comments"]) * 100, 2
    )

    summary["negative_rate"] = round(
        (summary["negative_comments"] / summary["total_comments"]) * 100, 2
    )

    summary["avg_likes_including_viral"] = summary["avg_likes_including_viral"].round(2)
    summary["avg_sentiment"] = summary["avg_sentiment"].round(4)

    return summary.sort_values("total_likes", ascending=False)


def extract_keywords(texts: pd.Series, top_n: int = 15) -> pd.DataFrame:
    stopwords = {
        "this", "that", "with", "have", "from", "your", "just", "like",
        "they", "them", "what", "when", "where", "will", "would", "could",
        "there", "their", "about", "because", "video", "rick", "astley",
        "song", "comment", "comments", "youtube", "very", "really",
    }

    words = []

    for text in texts.dropna():
        extracted = re.findall(r"\b[a-zA-Z]{4,}\b", str(text).lower())
        words.extend([word for word in extracted if word not in stopwords])

    keyword_counts = Counter(words).most_common(top_n)

    return pd.DataFrame(keyword_counts, columns=["keyword", "count"])


def generate_recommendation(df: pd.DataFrame) -> str:
    if df.empty:
        return "No recommendation available because no data matches the selected filters."

    negative_rate = (df["sentiment_label"] == "negative").mean()
    positive_rate = (df["sentiment_label"] == "positive").mean()
    raw_avg_likes = df["likes"].mean()
    typical_likes = adjusted_average(df["likes"])

    if negative_rate >= 0.2:
        return "⚠️ High negative sentiment detected. Review negative comments and identify recurring issues."

    if raw_avg_likes > typical_likes * 5 and raw_avg_likes > 100:
        return "🚀 Viral engagement detected. Study the most liked comments and replicate the content style."

    if positive_rate > negative_rate:
        return "✅ Audience response is healthy. Continue producing similar content while monitoring negative feedback."

    return "📊 Engagement is stable. Focus on increasing audience interaction and comment quality."


def generate_insights(df: pd.DataFrame) -> list[str]:
    total = len(df)

    if total == 0:
        return ["No comments match the selected filters."]

    sentiment_counts = df["sentiment_label"].value_counts(normalize=True) * 100

    positive_pct = round(sentiment_counts.get("positive", 0), 2)
    neutral_pct = round(sentiment_counts.get("neutral", 0), 2)
    negative_pct = round(sentiment_counts.get("negative", 0), 2)

    raw_avg = round(df["likes"].mean(), 2)
    typical_avg = adjusted_average(df["likes"])

    most_liked = df.sort_values("likes", ascending=False).iloc[0]
    max_likes = int(most_liked["likes"])

    video_summary = build_video_summary(df)
    top_video = video_summary.iloc[0]

    insights = [
        (
            f"The current selection contains {total} analyzed comments across "
            f"{df['video_id'].nunique()} video(s). Neutral sentiment leads at "
            f"{neutral_pct}%, followed by positive sentiment at {positive_pct}% "
            f"and negative sentiment at {negative_pct}%."
        ),
        (
            f"Average likes including viral outliers is {raw_avg}, while typical "
            f"average likes is {typical_avg}. This shows whether engagement is being "
            f"skewed by viral comments."
        ),
        f"The most liked comment is from {most_liked['author']} with {max_likes:,} likes.",
        (
            f"The strongest video by total comment likes is `{top_video['video_id']}` "
            f"with {int(top_video['total_likes']):,} total likes."
        ),
    ]

    if max_likes > df["likes"].mean() * 10 and max_likes > 100:
        insights.append(
            "A viral engagement outlier was detected. Raw engagement metrics should "
            "be interpreted together with typical engagement metrics."
        )

    if negative_pct >= 20:
        insights.append(
            "Negative sentiment is high enough to require investigation. This may indicate "
            "audience dissatisfaction, controversy, or reputational risk."
        )
    elif negative_pct > 0:
        insights.append(
            "Negative sentiment exists but is currently low. These comments should still "
            "be monitored as possible early warning signals."
        )
    else:
        insights.append("No negative sentiment was detected in the current selection.")

    return insights


def save_query(question: str, answer: str) -> None:
    SAVED_QUERIES_PATH.parent.mkdir(parents=True, exist_ok=True)

    row = pd.DataFrame(
        [
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "username": st.session_state.get("username", "unknown"),
                "question": question,
                "answer": answer,
            }
        ]
    )

    if SAVED_QUERIES_PATH.exists():
        existing = pd.read_csv(SAVED_QUERIES_PATH)
        output = pd.concat([existing, row], ignore_index=True)
    else:
        output = row

    output.to_csv(SAVED_QUERIES_PATH, index=False)


def load_saved_queries() -> pd.DataFrame:
    if not SAVED_QUERIES_PATH.exists():
        return pd.DataFrame(columns=["timestamp", "username", "question", "answer"])

    return pd.read_csv(SAVED_QUERIES_PATH)


def rule_based_answer(df: pd.DataFrame, question: str) -> str:
    question_lower = question.lower()

    if df.empty:
        return "No data is available for the selected filters."

    if "negative" in question_lower or "complain" in question_lower or "risk" in question_lower:
        negative_comments = (
            df[df["sentiment_label"] == "negative"]
            .sort_values("sentiment_score")
            .head(5)
        )

        if negative_comments.empty:
            return "No negative comments were found in the current filtered dataset."

        examples = "\n".join(
            f"- {row['clean_text']}" for _, row in negative_comments.iterrows()
        )

        return f"Strongest negative comments:\n\n{examples}"

    if "positive" in question_lower or "love" in question_lower or "good" in question_lower:
        positive_comments = (
            df[df["sentiment_label"] == "positive"]
            .sort_values("sentiment_score", ascending=False)
            .head(5)
        )

        if positive_comments.empty:
            return "No positive comments were found in the current filtered dataset."

        examples = "\n".join(
            f"- {row['clean_text']}" for _, row in positive_comments.iterrows()
        )

        return f"Strongest positive comments:\n\n{examples}"

    if "video" in question_lower or "best" in question_lower or "perform" in question_lower:
        video_summary = build_video_summary(df).head(5)

        rows = "\n".join(
            f"- `{row['video_id']}`: {row['total_comments']} comments, "
            f"{int(row['total_likes']):,} likes, "
            f"{row['positive_rate']}% positive, {row['negative_rate']}% negative"
            for _, row in video_summary.iterrows()
        )

        return f"Top performing videos:\n\n{rows}"

    if "keyword" in question_lower or "theme" in question_lower or "topic" in question_lower:
        keywords = extract_keywords(df["clean_text"], top_n=10)

        if keywords.empty:
            return "No keywords could be extracted from the current filtered dataset."

        rows = "\n".join(
            f"- {row['keyword']}: {row['count']}"
            for _, row in keywords.iterrows()
        )

        return f"Top themes and keywords:\n\n{rows}"

    if "summary" in question_lower or "overall" in question_lower:
        return "\n\n".join(generate_insights(df))

    return (
        "I can answer questions about negative comments, positive comments, top videos, "
        "video performance, keywords, themes, risks, and overall summaries."
    )


def answer_with_optional_ai(df: pd.DataFrame, question: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        return rule_based_answer(df, question)

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)

        sample = df[
            [
                "video_id",
                "author",
                "clean_text",
                "likes",
                "sentiment_label",
                "sentiment_score",
                "published_at",
            ]
        ].head(150).to_dict(orient="records")

        prompt = f"""
You are a senior data analyst reviewing YouTube comment intelligence data.

Dataset sample:
{sample}

User question:
{question}

Answer clearly and practically. Mention patterns, risks, and opportunities where relevant.
Do not invent facts outside the data.
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )

        return response.choices[0].message.content

    except Exception as error:
        return (
            "AI answer failed, so I used the local rule-based analyst instead.\n\n"
            f"{rule_based_answer(df, question)}\n\n"
            f"AI error: {error}"
        )


def render_kpis(df: pd.DataFrame) -> None:
    total = len(df)
    raw_avg = round(df["likes"].mean(), 2) if total else 0.0
    typical_avg = adjusted_average(df["likes"])

    positive = (df["sentiment_label"] == "positive").sum()
    neutral = (df["sentiment_label"] == "neutral").sum()
    negative = (df["sentiment_label"] == "negative").sum()

    c1, c2, c3, c4, c5, c6 = st.columns(6)

    c1.metric("Total Comments", total)
    c2.metric("Avg Likes Including Viral", raw_avg)
    c3.metric("Avg Likes Typical", typical_avg)
    c4.metric("Positive", positive, delta="Good")
    c5.metric("Neutral", neutral)
    c6.metric("Negative", negative, delta="Risk")

    st.caption("Typical average likes reduces the effect of extreme viral outliers.")


def render_live_ingestion() -> pd.DataFrame:
    st.markdown("## 🔴 Live YouTube API Ingestion")

    video_id = st.text_input("Enter YouTube video ID", placeholder="Example: dQw4w9WgXcQ")
    max_pages = st.slider("Pages to fetch", min_value=1, max_value=5, value=2)

    if st.button("Fetch Live Comments"):
        if not video_id:
            st.error("Please enter a YouTube video ID.")
            return pd.DataFrame()

        with st.spinner("Fetching live comments from YouTube..."):
            live_df = fetch_live_youtube_comments(video_id=video_id, max_pages=max_pages)

        if live_df.empty:
            st.warning("No live comments were fetched.")
            return pd.DataFrame()

        output_path = save_live_comments_to_bronze(live_df, video_id)

        st.success(f"Fetched {len(live_df)} comments and saved to {output_path}")
        st.dataframe(live_df.head(20), use_container_width=True)

        return live_df

    return pd.DataFrame()


def render_upload_csv() -> pd.DataFrame:
    st.markdown("## 📤 Upload CSV")

    uploaded_file = st.file_uploader(
        "Upload a YouTube comments CSV",
        type=["csv"],
    )

    if uploaded_file is None:
        return pd.DataFrame()

    uploaded_df = load_uploaded_csv(uploaded_file)

    st.success(f"Uploaded {len(uploaded_df)} rows.")
    st.dataframe(uploaded_df.head(20), use_container_width=True)

    return uploaded_df


def render_ask_data(df: pd.DataFrame) -> None:
    st.markdown("## 💬 Ask Your Data")

    saved_queries = load_saved_queries()

    if not saved_queries.empty:
        previous_questions = [""] + saved_queries["question"].dropna().unique().tolist()
        selected_previous = st.selectbox("Load a saved question", previous_questions)
    else:
        selected_previous = ""

    question = st.text_input(
        "Ask a question",
        value=selected_previous,
        placeholder="Example: What are people complaining about?",
    )

    if question:
        answer = answer_with_optional_ai(df, question)
        st.markdown(answer)

        if st.button("Save this query"):
            save_query(question, answer)
            st.success("Query saved.")

    with st.expander("View saved queries"):
        st.dataframe(load_saved_queries(), use_container_width=True)


def render_video_analytics(df: pd.DataFrame) -> None:
    st.markdown("## 📹 Video-Level Analytics")

    video_summary = build_video_summary(df)

    if video_summary.empty:
        st.info("No video summary available.")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Top Performing Videos by Likes")
        fig = px.bar(
            video_summary.head(10),
            x="video_id",
            y="total_likes",
            title="Total Comment Likes by Video",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Sentiment Quality by Video")
        fig = px.bar(
            video_summary.head(10),
            x="video_id",
            y="positive_rate",
            title="Positive Comment Rate by Video",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Video Performance Table")
    st.dataframe(video_summary, use_container_width=True)

    st.markdown("### ⚠️ Risky Videos")
    risky_videos = video_summary.sort_values("negative_rate", ascending=False).head(5)

    st.dataframe(
        risky_videos[
            [
                "video_id",
                "total_comments",
                "total_likes",
                "negative_comments",
                "negative_rate",
                "positive_rate",
            ]
        ],
        use_container_width=True,
    )


def render_charts(df: pd.DataFrame) -> None:
    sentiment_counts = df["sentiment_label"].value_counts().reset_index()
    sentiment_counts.columns = ["sentiment_label", "comments"]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("## 📊 Sentiment Distribution")
        fig = px.pie(
            sentiment_counts,
            names="sentiment_label",
            values="comments",
            hole=0.45,
            title="Comment Sentiment Breakdown",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("## 🎯 Typical Likes by Sentiment")

        likes_by_sentiment = (
            df.groupby("sentiment_label")["likes"]
            .apply(adjusted_average)
            .reset_index(name="typical_likes")
        )

        fig = px.bar(
            likes_by_sentiment,
            x="sentiment_label",
            y="typical_likes",
            text_auto=True,
            title="Typical Engagement by Sentiment",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.markdown("## 📈 Sentiment Over Time")

    time_df = df.dropna(subset=["month"])

    if time_df.empty:
        st.info("No valid date values available for time trend analysis.")
    else:
        trend = (
            time_df.groupby(["month", "sentiment_label"])
            .size()
            .reset_index(name="comments")
        )

        fig = px.line(
            trend,
            x="month",
            y="comments",
            color="sentiment_label",
            markers=True,
            title="Monthly Sentiment Trend",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("## 👥 Top Commenters")

        top_authors = (
            df.groupby("author")
            .size()
            .reset_index(name="comments")
            .sort_values("comments", ascending=False)
            .head(10)
        )

        fig = px.bar(
            top_authors,
            x="author",
            y="comments",
            title="Most Active Commenters",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.markdown("## 🔎 Engagement vs Sentiment")

        scatter_df = df.copy()

        if len(scatter_df) > 1:
            upper_limit = scatter_df["likes"].quantile(0.95)
            scatter_df = scatter_df[scatter_df["likes"] <= upper_limit]

        fig = px.scatter(
            scatter_df,
            x="sentiment_score",
            y="likes",
            color="sentiment_label",
            hover_data=["author", "clean_text", "video_id"],
            title="Normal Engagement Pattern",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.caption("Extreme engagement outliers are excluded from this chart for readability.")


def render_engagement_segments(df: pd.DataFrame) -> None:
    st.markdown("## 🧩 Engagement Segmentation")

    bucket_counts = df["engagement_bucket"].value_counts().reset_index()
    bucket_counts.columns = ["engagement_bucket", "comments"]

    fig = px.bar(
        bucket_counts,
        x="engagement_bucket",
        y="comments",
        title="Comment Engagement Segments",
    )

    st.plotly_chart(fig, use_container_width=True)


def render_keywords(df: pd.DataFrame) -> None:
    st.markdown("## 🔤 Top Keywords / Themes")

    keywords = extract_keywords(df["clean_text"], top_n=15)

    if keywords.empty:
        st.info("No keywords available for the current filtered dataset.")
        return

    fig = px.bar(
        keywords,
        x="keyword",
        y="count",
        title="Most Common Keywords",
    )

    st.plotly_chart(fig, use_container_width=True)


def render_tables(df: pd.DataFrame) -> None:
    st.markdown("## 💬 Most Liked Comments")

    top_comments = df.sort_values("likes", ascending=False).head(10)

    st.dataframe(
        top_comments[
            [
                "video_id",
                "author",
                "clean_text",
                "likes",
                "sentiment_label",
                "sentiment_score",
            ]
        ],
        use_container_width=True,
    )

    st.divider()

    st.markdown("## 🔍 Explore Comments")

    st.dataframe(
        df[
            [
                "video_id",
                "author",
                "clean_text",
                "likes",
                "sentiment_label",
                "sentiment_score",
                "published_at",
            ]
        ],
        use_container_width=True,
    )


def render_export(df: pd.DataFrame) -> None:
    st.divider()
    st.markdown("## 📥 Export Filtered Data")

    csv_data = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download filtered comments as CSV",
        data=csv_data,
        file_name="socialpulse_filtered_comments.csv",
        mime="text/csv",
    )


def build_combined_data() -> pd.DataFrame:
    base_df = load_gold_data()

    if "live_df" in st.session_state and not st.session_state["live_df"].empty:
        base_df = pd.concat([base_df, st.session_state["live_df"]], ignore_index=True)

    if "uploaded_df" in st.session_state and not st.session_state["uploaded_df"].empty:
        base_df = pd.concat([base_df, st.session_state["uploaded_df"]], ignore_index=True)

    return standardize_dataframe(base_df)


def sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.title("Dashboard Filters")

    auto_refresh = st.sidebar.checkbox("Auto refresh dashboard", value=False)

    if auto_refresh:
        refresh_seconds = st.sidebar.slider(
            "Refresh every seconds",
            min_value=30,
            max_value=300,
            value=60,
        )

        time.sleep(refresh_seconds)
        st.rerun()

    video_options = ["all"] + sorted(df["video_id"].dropna().unique().tolist())
    sentiment_options = ["all"] + sorted(df["sentiment_label"].dropna().unique().tolist())

    selected_video = st.sidebar.selectbox("Select video", video_options)
    selected_sentiment = st.sidebar.selectbox("Select sentiment", sentiment_options)

    min_likes = st.sidebar.slider(
        "Minimum likes",
        min_value=0,
        max_value=int(df["likes"].max()) if not df.empty else 0,
        value=0,
    )

    search_text = st.sidebar.text_input("Search comment text")

    return filter_data(
        df=df,
        video_id=selected_video,
        sentiment=selected_sentiment,
        min_likes=min_likes,
        search_text=search_text,
    )


def main() -> None:
    require_login()
    logout_button()

    st.title("SocialPulse AI")
    st.subheader("YouTube Social Media Intelligence Platform")

    data_source_tab, overview_tab, video_tab, ai_tab, audience_tab, comments_tab = st.tabs(
        [
            "Data Sources",
            "Overview",
            "Video Analytics",
            "AI Analyst",
            "Audience Insights",
            "Comments Explorer",
        ]
    )

    with data_source_tab:
        uploaded_df = render_upload_csv()

        if not uploaded_df.empty:
            st.session_state["uploaded_df"] = uploaded_df

        st.divider()

        live_df = render_live_ingestion()

        if not live_df.empty:
            st.session_state["live_df"] = live_df

    df = build_combined_data()

    if df.empty:
        st.warning(
            "No data available. Upload a CSV, fetch live YouTube comments, or run your gold pipeline."
        )
        return

    filtered = sidebar_filters(df)

    if filtered.empty:
        st.warning("No comments match your selected filters.")
        return

    with overview_tab:
        render_kpis(filtered)
        st.divider()

        st.markdown("## 🤖 AI Insights Summary")

        for insight in generate_insights(filtered):
            st.info(insight)

        st.success(generate_recommendation(filtered))

    with video_tab:
        render_video_analytics(filtered)

    with ai_tab:
        render_ask_data(filtered)

    with audience_tab:
        render_charts(filtered)
        st.divider()
        render_engagement_segments(filtered)
        st.divider()
        render_keywords(filtered)

    with comments_tab:
        render_tables(filtered)
        render_export(filtered)


if __name__ == "__main__":
    main()