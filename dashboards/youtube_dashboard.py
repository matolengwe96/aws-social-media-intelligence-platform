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
    df["month"] = df["published_at"].dt.to_period("M").dt.to_timestamp()

    df["video_id"] = df["video_id"].fillna("unknown")
    df["author"] = df["author"].fillna("Unknown")
    df["clean_text"] = df["clean_text"].fillna("")
    df["sentiment_label"] = df["sentiment_label"].fillna("neutral")

    df["likes"] = pd.to_numeric(df["likes"], errors="coerce").fillna(0)
    df["sentiment_score"] = pd.to_numeric(
        df["sentiment_score"], errors="coerce"
    ).fillna(0)

    return df


def adjusted_average(series: pd.Series) -> float:
    if series.empty:
        return 0.0

    upper_limit = series.quantile(0.95)
    return round(series.clip(upper=upper_limit).mean(), 2)


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

    top_video = (
        df.groupby("video_id")["likes"]
        .sum()
        .reset_index(name="total_likes")
        .sort_values("total_likes", ascending=False)
        .iloc[0]
    )

    insights = [
        (
            f"The current selection contains {total} analyzed comments across "
            f"{df['video_id'].nunique()} video(s). Neutral sentiment leads at "
            f"{neutral_pct}%, followed by positive sentiment at {positive_pct}% "
            f"and negative sentiment at {negative_pct}%."
        ),
        (
            f"Average likes including viral outliers is {raw_avg}, while typical "
            f"average likes is {typical_avg}. This indicates that engagement is "
            f"affected by one or more high-performing comments."
        ),
        (
            f"The most liked comment is from {most_liked['author']} with "
            f"{max_likes:,} likes."
        ),
        (
            f"The strongest video by total comment likes is `{top_video['video_id']}` "
            f"with {int(top_video['total_likes']):,} likes."
        ),
    ]

    if max_likes > df["likes"].mean() * 10 and max_likes > 100:
        insights.append(
            "A viral engagement outlier was detected. Raw engagement metrics should "
            "therefore be interpreted together with adjusted metrics."
        )

    if negative_pct >= 20:
        insights.append(
            "Negative sentiment is high enough to require investigation. This may "
            "indicate audience dissatisfaction, controversy, or reputational risk."
        )
    elif negative_pct > 0:
        insights.append(
            "Negative sentiment exists but is currently low. These comments should "
            "still be monitored as possible early warning signals."
        )
    else:
        insights.append("No negative sentiment was detected in the current selection.")

    top_negative = (
        df[df["sentiment_label"] == "negative"]
        .sort_values("sentiment_score")
        .head(2)
    )

    if not top_negative.empty:
        examples = "; ".join(top_negative["clean_text"].tolist())
        insights.append(f"Example negative feedback: {examples}")

    top_positive = (
        df[df["sentiment_label"] == "positive"]
        .sort_values("sentiment_score", ascending=False)
        .head(2)
    )

    if not top_positive.empty:
        examples = "; ".join(top_positive["clean_text"].tolist())
        insights.append(f"Example positive feedback: {examples}")

    return insights


def answer_data_question(df: pd.DataFrame, question: str) -> str:
    if not question:
        return ""

    question_lower = question.lower()

    if df.empty:
        return "No data is available for the selected filters."

    if "negative" in question_lower or "complain" in question_lower:
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

        return f"Here are the strongest negative comments:\n\n{examples}"

    if "positive" in question_lower or "like" in question_lower:
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

        return f"Here are the strongest positive comments:\n\n{examples}"

    if "video" in question_lower or "best" in question_lower or "perform" in question_lower:
        video_summary = build_video_summary(df).head(5)

        rows = "\n".join(
            f"- `{row['video_id']}`: {row['total_comments']} comments, "
            f"{row['total_likes']:,} likes, {row['positive_rate']}% positive"
            for _, row in video_summary.iterrows()
        )

        return f"Top performing videos:\n\n{rows}"

    if "summary" in question_lower or "overall" in question_lower:
        return "\n\n".join(generate_insights(df))

    return (
        "I can currently answer questions about negative comments, positive comments, "
        "top videos, video performance, and overall summaries."
    )


def build_video_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("video_id")
        .agg(
            total_comments=("clean_text", "count"),
            total_likes=("likes", "sum"),
            raw_avg_likes=("likes", "mean"),
            typical_likes=("likes", adjusted_average),
            avg_sentiment=("sentiment_score", "mean"),
            positive_comments=(
                "sentiment_label",
                lambda x: (x == "positive").sum(),
            ),
            neutral_comments=(
                "sentiment_label",
                lambda x: (x == "neutral").sum(),
            ),
            negative_comments=(
                "sentiment_label",
                lambda x: (x == "negative").sum(),
            ),
        )
        .reset_index()
    )

    summary["positive_rate"] = round(
        (summary["positive_comments"] / summary["total_comments"]) * 100, 2
    )

    summary["negative_rate"] = round(
        (summary["negative_comments"] / summary["total_comments"]) * 100, 2
    )

    summary["raw_avg_likes"] = summary["raw_avg_likes"].round(2)
    summary["avg_sentiment"] = summary["avg_sentiment"].round(4)

    return summary.sort_values("total_likes", ascending=False)


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
    c4.metric("Positive", positive)
    c5.metric("Neutral", neutral)
    c6.metric("Negative", negative)

    st.caption("Typical average likes reduces the effect of extreme viral outliers.")


def render_video_analytics(df: pd.DataFrame) -> None:
    st.subheader("Video-Level Analytics")

    video_summary = build_video_summary(df)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Top Performing Videos by Likes")
        fig = px.bar(
            video_summary.head(10),
            x="video_id",
            y="total_likes",
            title="Total Comment Likes by Video",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Sentiment Quality by Video")
        fig = px.bar(
            video_summary.head(10),
            x="video_id",
            y="positive_rate",
            title="Positive Comment Rate by Video",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Video Performance Table")
    st.dataframe(video_summary, use_container_width=True)


def render_charts(df: pd.DataFrame) -> None:
    sentiment_counts = df["sentiment_label"].value_counts().reset_index()
    sentiment_counts.columns = ["sentiment_label", "comments"]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Sentiment Distribution")
        fig = px.pie(
            sentiment_counts,
            names="sentiment_label",
            values="comments",
            hole=0.45,
            title="Comment Sentiment Breakdown",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Typical Likes by Sentiment")

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

    st.subheader("Sentiment Over Time")

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
        st.subheader("Top Commenters")

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
        st.subheader("Engagement vs Sentiment")

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


def render_tables(df: pd.DataFrame) -> None:
    st.subheader("Most Liked Comments")

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

    st.subheader("Explore Comments")

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


def render_ask_data(df: pd.DataFrame) -> None:
    st.subheader("Ask Your Data")

    question = st.text_input(
        "Ask a question",
        placeholder="Example: What are people complaining about?",
    )

    if question:
        answer = answer_data_question(df, question)
        st.markdown(answer)


def render_export(df: pd.DataFrame) -> None:
    st.divider()
    st.subheader("Export Filtered Data")

    csv_data = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download filtered comments as CSV",
        data=csv_data,
        file_name="socialpulse_filtered_comments.csv",
        mime="text/csv",
    )


def main() -> None:
    st.title("SocialPulse AI")
    st.subheader("YouTube Social Media Intelligence Dashboard")

    if not GOLD_COMMENTS_PATH.exists():
        st.error("Gold data not found. Run: python src/transformation/youtube_to_gold.py")
        return

    df = load_data()

    st.sidebar.title("Dashboard Filters")

    video_options = ["all"] + sorted(df["video_id"].dropna().unique().tolist())
    sentiment_options = ["all"] + sorted(
        df["sentiment_label"].dropna().unique().tolist()
    )

    selected_video = st.sidebar.selectbox("Select video", video_options)

    selected_sentiment = st.sidebar.selectbox(
        "Select sentiment",
        sentiment_options,
    )

    min_likes = st.sidebar.slider(
        "Minimum likes",
        min_value=0,
        max_value=int(df["likes"].max()),
        value=0,
    )

    search_text = st.sidebar.text_input("Search comment text")

    filtered = filter_data(
        df=df,
        video_id=selected_video,
        sentiment=selected_sentiment,
        min_likes=min_likes,
        search_text=search_text,
    )

    render_kpis(filtered)

    st.divider()

    if filtered.empty:
        st.warning("No comments match your selected filters.")
        return

    st.subheader("AI Insights Summary")

    for insight in generate_insights(filtered):
        st.info(insight)

    st.divider()

    render_video_analytics(filtered)

    st.divider()

    render_ask_data(filtered)

    st.divider()

    render_charts(filtered)

    st.divider()

    render_tables(filtered)

    render_export(filtered)


if __name__ == "__main__":
    main()