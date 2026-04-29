from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# Paths
GOLD_COMMENTS_PATH = Path("data/gold/youtube/youtube_comments_enriched.parquet")

st.set_page_config(
    page_title="SocialPulse AI Dashboard",
    page_icon="📊",
    layout="wide",
)


@st.cache_data
def load_data():
    df = pd.read_parquet(GOLD_COMMENTS_PATH)

    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
    df["date"] = df["published_at"].dt.date

    return df


def main():
    st.title("SocialPulse AI")
    st.subheader("YouTube Social Media Intelligence Dashboard")

    if not GOLD_COMMENTS_PATH.exists():
        st.error("Run gold transformation first.")
        return

    df = load_data()

    # =========================
    # SIDEBAR FILTERS
    # =========================
    st.sidebar.title("Dashboard Filters")

    sentiment_options = ["all"] + sorted(df["sentiment_label"].dropna().unique())

    selected_sentiment = st.sidebar.selectbox(
        "Select sentiment",
        sentiment_options,
    )

    min_likes = st.sidebar.slider(
        "Minimum likes",
        0,
        int(df["likes"].max()),
        0,
    )

    search_text = st.sidebar.text_input("Search comment text")

    # Apply filters
    filtered = df.copy()

    if selected_sentiment != "all":
        filtered = filtered[filtered["sentiment_label"] == selected_sentiment]

    filtered = filtered[filtered["likes"] >= min_likes]

    if search_text:
        filtered = filtered[
            filtered["clean_text"].str.contains(search_text, case=False, na=False)
        ]

    # =========================
    # METRICS
    # =========================
    total = len(filtered)
    avg_likes = filtered["likes"].mean() if total > 0 else 0

    pos = (filtered["sentiment_label"] == "positive").sum()
    neu = (filtered["sentiment_label"] == "neutral").sum()
    neg = (filtered["sentiment_label"] == "negative").sum()

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("Total Comments", total)
    c2.metric("Avg Likes", round(avg_likes, 2))
    c3.metric("Positive", pos)
    c4.metric("Neutral", neu)
    c5.metric("Negative", neg)

    st.divider()

    if filtered.empty:
        st.warning("No data after filtering.")
        return

    # =========================
    # SENTIMENT DISTRIBUTION
    # =========================
    sentiment_counts = filtered["sentiment_label"].value_counts().reset_index()
    sentiment_counts.columns = ["sentiment", "count"]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Sentiment Distribution")
        fig = px.pie(
            sentiment_counts,
            names="sentiment",
            values="count",
            hole=0.4,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Average Likes by Sentiment")

        avg_likes_sent = (
            filtered.groupby("sentiment_label")["likes"]
            .mean()
            .reset_index()
        )

        fig = px.bar(
            avg_likes_sent,
            x="sentiment_label",
            y="likes",
            text_auto=True,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # =========================
    # TIME SERIES (FIXED)
    # =========================
    st.subheader("Sentiment Over Time")

    time_df = filtered.dropna(subset=["date"])

    if not time_df.empty:
        trend = (
            time_df.groupby(["date", "sentiment_label"])
            .size()
            .reset_index(name="count")
        )

        fig = px.line(
            trend,
            x="date",
            y="count",
            color="sentiment_label",
            markers=True,
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No time data available.")

    st.divider()

    # =========================
    # TOP AUTHORS
    # =========================
    st.subheader("Top Commenters")

    top_authors = (
        filtered.groupby("author")
        .size()
        .reset_index(name="comments")
        .sort_values("comments", ascending=False)
        .head(10)
    )

    fig = px.bar(
        top_authors,
        x="author",
        y="comments",
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # =========================
    # ENGAGEMENT SCATTER
    # =========================
    st.subheader("Engagement vs Sentiment")

    fig = px.scatter(
        filtered,
        x="sentiment_score",
        y="likes",
        color="sentiment_label",
        hover_data=["author", "clean_text"],
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # =========================
    # TOP COMMENTS
    # =========================
    st.subheader("Most Liked Comments")

    top_comments = filtered.sort_values("likes", ascending=False).head(10)

    st.dataframe(
        top_comments[
            [
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

    # =========================
    # FULL TABLE
    # =========================
    st.subheader("Explore Comments")

    st.dataframe(
        filtered[
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