from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


GOLD_COMMENTS_PATH = Path("data/gold/youtube/youtube_comments_enriched.parquet")


st.set_page_config(
    page_title="SocialPulse AI Dashboard",
    page_icon="📊",
    layout="wide",
)


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_parquet(GOLD_COMMENTS_PATH)

    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
    df["date"] = df["published_at"].dt.to_period("M").dt.to_timestamp()

    df["clean_text"] = df["clean_text"].fillna("")
    df["author"] = df["author"].fillna("Unknown")
    df["likes"] = pd.to_numeric(df["likes"], errors="coerce").fillna(0)
    df["sentiment_score"] = pd.to_numeric(
        df["sentiment_score"], errors="coerce"
    ).fillna(0)

    return df


def safe_average_likes(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0

    capped_likes = df["likes"].clip(upper=df["likes"].quantile(0.95))
    return round(capped_likes.mean(), 2)


def generate_ai_insights(df: pd.DataFrame) -> list[str]:
    total_comments = len(df)

    if total_comments == 0:
        return ["No comments available for insight generation."]

    positive_count = (df["sentiment_label"] == "positive").sum()
    neutral_count = (df["sentiment_label"] == "neutral").sum()
    negative_count = (df["sentiment_label"] == "negative").sum()

    positive_pct = round((positive_count / total_comments) * 100, 2)
    neutral_pct = round((neutral_count / total_comments) * 100, 2)
    negative_pct = round((negative_count / total_comments) * 100, 2)

    avg_likes = safe_average_likes(df)
    max_likes = int(df["likes"].max())
    most_liked = df.sort_values("likes", ascending=False).iloc[0]

    insights = []

    insights.append(
        f"The dataset contains {total_comments} analyzed comments. "
        f"Sentiment is mostly neutral at {neutral_pct}%, followed by "
        f"positive at {positive_pct}% and negative at {negative_pct}%."
    )

    insights.append(
        f"Typical engagement is around {avg_likes} likes per comment after reducing "
        f"the effect of extreme outliers. The most liked comment is from "
        f"{most_liked['author']} with {max_likes:,} likes."
    )

    if max_likes > df["likes"].mean() * 10 and max_likes > 100:
        insights.append(
            "An unusual engagement spike was detected. One comment has significantly "
            "more likes than the rest, so average engagement should be interpreted carefully."
        )

    if negative_pct >= 20:
        insights.append(
            "Negative sentiment is relatively high. This may indicate dissatisfaction, "
            "controversy, or audience concern that should be investigated."
        )
    elif negative_pct > 0:
        insights.append(
            "Negative sentiment is present but low. Monitor the negative comments to check "
            "whether they are isolated complaints or early warning signals."
        )
    else:
        insights.append("No negative comments were detected in the current filtered dataset.")

    if positive_pct > negative_pct:
        insights.append(
            "Overall audience response appears healthier than risky because positive sentiment "
            "is stronger than negative sentiment."
        )
    elif negative_pct > positive_pct:
        insights.append(
            "Audience risk is elevated because negative sentiment is higher than positive sentiment."
        )
    else:
        insights.append(
            "Positive and negative sentiment are balanced, so the conversation is not clearly leaning either way."
        )

    top_negative = (
        df[df["sentiment_label"] == "negative"]
        .sort_values("sentiment_score", ascending=True)
        .head(3)
    )

    if not top_negative.empty:
        negative_examples = "; ".join(top_negative["clean_text"].tolist())
        insights.append(f"Sample negative feedback: {negative_examples}")

    top_positive = (
        df[df["sentiment_label"] == "positive"]
        .sort_values("sentiment_score", ascending=False)
        .head(3)
    )

    if not top_positive.empty:
        positive_examples = "; ".join(top_positive["clean_text"].tolist())
        insights.append(f"Sample positive feedback: {positive_examples}")

    return insights


def main() -> None:
    st.title("SocialPulse AI")
    st.subheader("YouTube Social Media Intelligence Dashboard")

    if not GOLD_COMMENTS_PATH.exists():
        st.error("Gold data not found. Run: python src/transformation/youtube_to_gold.py")
        return

    df = load_data()

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

    filtered = df.copy()

    if selected_sentiment != "all":
        filtered = filtered[filtered["sentiment_label"] == selected_sentiment]

    filtered = filtered[filtered["likes"] >= min_likes]

    if search_text:
        filtered = filtered[
            filtered["clean_text"].str.contains(search_text, case=False, na=False)
        ]

    total = len(filtered)
    avg_likes = safe_average_likes(filtered)
    pos = (filtered["sentiment_label"] == "positive").sum()
    neu = (filtered["sentiment_label"] == "neutral").sum()
    neg = (filtered["sentiment_label"] == "negative").sum()

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("Total Comments", total)
    c2.metric("Avg Likes", avg_likes)
    c3.metric("Positive", pos)
    c4.metric("Neutral", neu)
    c5.metric("Negative", neg)

    st.divider()

    if filtered.empty:
        st.warning("No data after filtering.")
        return

    st.subheader("AI Insights Summary")

    for insight in generate_ai_insights(filtered):
        st.info(insight)

    st.divider()

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
            .apply(lambda x: x.clip(upper=x.quantile(0.95)).mean())
            .reset_index(name="avg_likes")
        )

        fig = px.bar(
            avg_likes_sent,
            x="sentiment_label",
            y="avg_likes",
            text_auto=True,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

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

    st.divider()

    st.subheader("Export Filtered Data")

    csv_data = filtered.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download filtered comments as CSV",
        data=csv_data,
        file_name="socialpulse_filtered_comments.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()