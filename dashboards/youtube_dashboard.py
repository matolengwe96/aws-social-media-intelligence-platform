from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


GOLD_COMMENTS_PATH = Path("data/gold/youtube/youtube_comments_enriched.parquet")
GOLD_SUMMARY_PATH = Path("data/gold/youtube/youtube_sentiment_summary.csv")


st.set_page_config(
    page_title="SocialPulse AI Dashboard",
    page_icon="📊",
    layout="wide",
)


@st.cache_data
def load_data():
    comments = pd.read_parquet(GOLD_COMMENTS_PATH)
    summary = pd.read_csv(GOLD_SUMMARY_PATH)

    comments["published_at"] = pd.to_datetime(
        comments["published_at"], errors="coerce"
    )

    return comments, summary


def main():
    st.title("SocialPulse AI")
    st.subheader("YouTube Social Media Intelligence Dashboard")

    if not GOLD_COMMENTS_PATH.exists() or not GOLD_SUMMARY_PATH.exists():
        st.error(
            "Gold data not found. Run: python src/transformation/youtube_to_gold.py"
        )
        return

    comments_df, summary_df = load_data()

    st.sidebar.title("Dashboard Filters")

    sentiment_options = ["all"] + sorted(
        comments_df["sentiment_label"].dropna().unique().tolist()
    )

    selected_sentiment = st.sidebar.selectbox(
        "Select sentiment",
        sentiment_options,
    )

    max_likes = int(comments_df["likes"].max())

    min_likes = st.sidebar.slider(
        "Minimum likes",
        min_value=0,
        max_value=max_likes,
        value=0,
    )

    filtered_df = comments_df.copy()

    if selected_sentiment != "all":
        filtered_df = filtered_df[
            filtered_df["sentiment_label"] == selected_sentiment
        ]

    filtered_df = filtered_df[filtered_df["likes"] >= min_likes]

    total_comments = len(filtered_df)
    avg_likes = filtered_df["likes"].mean() if total_comments > 0 else 0
    positive_count = len(
        filtered_df[filtered_df["sentiment_label"] == "positive"]
    )
    neutral_count = len(
        filtered_df[filtered_df["sentiment_label"] == "neutral"]
    )
    negative_count = len(
        filtered_df[filtered_df["sentiment_label"] == "negative"]
    )

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Total Comments", total_comments)
    col2.metric("Average Likes", round(avg_likes, 2))
    col3.metric("Positive", positive_count)
    col4.metric("Neutral", neutral_count)
    col5.metric("Negative", negative_count)

    st.divider()

    if filtered_df.empty:
        st.warning("No comments match your selected filters.")
        return

    sentiment_summary = (
        filtered_df.groupby("sentiment_label")
        .agg(
            total_comments=("clean_text", "count"),
            avg_likes=("likes", "mean"),
        )
        .reset_index()
    )

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Sentiment Distribution")
        fig = px.pie(
            sentiment_summary,
            names="sentiment_label",
            values="total_comments",
            title="Comment Sentiment Breakdown",
            hole=0.35,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Average Likes by Sentiment")
        fig = px.bar(
            sentiment_summary,
            x="sentiment_label",
            y="avg_likes",
            title="Average Likes per Sentiment",
            text_auto=True,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.subheader("Sentiment Over Time")

    time_df = filtered_df.dropna(subset=["published_at"]).copy()

    if not time_df.empty:
        sentiment_over_time = (
            time_df.groupby(
                [time_df["published_at"].dt.date, "sentiment_label"]
            )
            .size()
            .reset_index(name="comment_count")
        )

        fig = px.line(
            sentiment_over_time,
            x="published_at",
            y="comment_count",
            color="sentiment_label",
            markers=True,
            title="Comment Sentiment Trend Over Time",
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No valid dates available for sentiment-over-time chart.")

    st.divider()

    st.subheader("Most Liked Comments")

    top_comments = filtered_df.sort_values("likes", ascending=False)[
        [
            "author",
            "clean_text",
            "likes",
            "sentiment_label",
            "sentiment_score",
        ]
    ].head(10)

    st.dataframe(top_comments, use_container_width=True)

    st.divider()

    st.subheader("Explore Comments")

    st.dataframe(
        filtered_df[
            [
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


if __name__ == "__main__":
    main()