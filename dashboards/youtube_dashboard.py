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
    return comments, summary


st.title("SocialPulse AI")
st.subheader("YouTube Social Media Intelligence Dashboard")

comments_df, summary_df = load_data()

total_comments = len(comments_df)
avg_likes = comments_df["likes"].mean()
positive_count = len(comments_df[comments_df["sentiment_label"] == "positive"])
negative_count = len(comments_df[comments_df["sentiment_label"] == "negative"])

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Comments", total_comments)
col2.metric("Average Likes", round(avg_likes, 2))
col3.metric("Positive Comments", positive_count)
col4.metric("Negative Comments", negative_count)

st.divider()

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Sentiment Distribution")
    fig = px.pie(
        summary_df,
        names="sentiment_label",
        values="total_comments",
        title="Comment Sentiment Breakdown",
    )
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.subheader("Average Likes by Sentiment")
    fig = px.bar(
        summary_df,
        x="sentiment_label",
        y="avg_likes",
        title="Average Likes per Sentiment",
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()

st.subheader("Most Liked Comments")

top_comments = comments_df.sort_values("likes", ascending=False)[
    ["author", "clean_text", "likes", "sentiment_label", "sentiment_score"]
].head(10)

st.dataframe(top_comments, use_container_width=True)

st.divider()

st.subheader("Explore Comments")

sentiment_filter = st.selectbox(
    "Filter by sentiment",
    ["all", "positive", "neutral", "negative"],
)

filtered_df = comments_df.copy()

if sentiment_filter != "all":
    filtered_df = filtered_df[filtered_df["sentiment_label"] == sentiment_filter]

st.dataframe(
    filtered_df[["author", "clean_text", "likes", "sentiment_label", "published_at"]],
    use_container_width=True,
)