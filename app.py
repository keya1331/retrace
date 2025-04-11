import streamlit as st
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Custom modules
try:
    import preprocessor
    import helper
except ModuleNotFoundError:
    st.error("Missing custom modules: `preprocessor.py` and `helper.py` must be present in the same directory.")
    st.stop()

# Streamlit configuration
st.set_page_config(page_title="WhatsApp Chat Analyzer", page_icon="💬")

st.sidebar.title("Whatsapp Chat Analyzer")

# Header and description
st.markdown(
    """
    <div style="text-align: center;">
        <h1 style="color: #ff4b4b; font-family: Arial, sans-serif;">
            Welcome to the <span style="color: #ff914d;">WhatsApp Chat Analyzer</span> 📊
        </h1>
        <p style="font-size: 18px; font-family: 'Trebuchet MS', sans-serif; color: #ddd;">
            Upload your chat file to get insightful analytics on your conversations.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

uploaded_file = st.sidebar.file_uploader("Choose a chat text file", type=["txt"])

# Save chat file only if running in an environment where file system is available
UPLOAD_DIR = "uploaded_chats"
if uploaded_file is not None:
    try:
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"chat_{timestamp}.txt"
        file_path = os.path.join(UPLOAD_DIR, filename)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    except Exception as e:
        st.warning(f"Couldn't save file: {e}")

if uploaded_file is not None:
    st.success("File uploaded successfully! Processing...")

    # Decode uploaded file
    try:
        data = uploaded_file.getvalue().decode("utf-8")
        df = preprocessor.preprocess(data)
    except Exception as e:
        st.error(f"Error processing file: {e}")
        st.stop()

    st.dataframe(df)

    user_list = df['user'].dropna().unique().tolist()
    user_list = [user for user in user_list if "Messages and calls are end-to-end encrypted" not in user]
    user_list.sort()
    user_list.insert(0, "Overall")

    selected_user = st.sidebar.selectbox("SHOW ANALYSIS W.R.T.", user_list)

    if st.sidebar.button("SHOW ANALYSIS!"):
        st.header("DETAILED ANALYSIS OF THE Whatsapp CHAT!")

        # Basic Stats
        num_messages, total_words, media_shared, link_shared = helper.fetch_stats(selected_user, df)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Messages", num_messages)
        c2.metric("Words", total_words)
        c3.metric("Media", media_shared)
        c4.metric("Links", link_shared)

        # TF-IDF
        st.title("📝 Term Significance Analysis (TF-IDF)")
        tfidf_results, full_tfidf_df = helper.perform_tfidf_analysis(selected_user, df)
        tfidf_fig = helper.plot_tfidf_results(tfidf_results, selected_user)
        if tfidf_fig: st.pyplot(tfidf_fig)
        if full_tfidf_df is not None and not full_tfidf_df.empty:
            st.subheader("Detailed TF-IDF Scores")
            st.dataframe(full_tfidf_df.head(20), hide_index=True)

        # Monthly Timeline
        st.title("Monthly Timeline")
        timeline = helper.monthly_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(timeline['time'], timeline['user_message'], color='green')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # Daily Timeline
        st.title("Daily Timeline")
        daily = helper.daily_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(daily['only_date'], daily['user_message'], color='black')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # Activity Map
        st.title("ACTIVITY MAP")
        c1, c2 = st.columns(2)

        with c1:
            st.subheader("Busy Days")
            busy_day = helper.week_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_day.index, busy_day.values, color='purple')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        with c2:
            st.subheader("Busy Months")
            busy_month = helper.month_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values, color='orange')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        # Heatmap
        st.title("WEEKLY ACTIVITY MAP")
        heatmap = helper.activity_heatmap(selected_user, df)
        fig, ax = plt.subplots()
        sns.heatmap(heatmap, ax=ax)
        st.pyplot(fig)

        # Most Busy Users
        if selected_user == "Overall":
            st.title("Most Active Users")
            x, df1 = helper.fetch_most_busy_users(df)
            c1, c2 = st.columns(2)
            with c1:
                fig, ax = plt.subplots()
                ax.bar(x.index, x.values, color='red')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with c2:
                st.dataframe(df1)

        # Word Cloud
        st.title("Word Cloud")
        df_wc = helper.createwordcloud(selected_user, df)
        fig, ax = plt.subplots()
        ax.imshow(df_wc)
        st.pyplot(fig)

        # Common Words
        st.title("Most Common Words")
        most_common_df = helper.most_common_words(selected_user, df)
        fig, ax = plt.subplots()
        ax.barh(most_common_df[0], most_common_df[1])
        st.pyplot(fig)

        # Emoji Analysis
        st.title("Emoji Analysis")
        emoji_df = helper.emoji_helper(selected_user, df)
        c1, c2 = st.columns(2)
        with c1:
            st.dataframe(emoji_df)
        with c2:
            fig, ax = plt.subplots()
            if not emoji_df.empty and len(emoji_df.columns) > 1:
                ax.pie(emoji_df.iloc[:, 1].head(), labels=emoji_df.iloc[:, 0].head(), autopct="%0.2f")
                st.pyplot(fig)
            else:
                st.warning("Not enough emoji data to generate a pie chart.")

        # Sentiment Analysis
        st.title("💭 Sentiment Analysis")
        sentiment_counts, daily_sentiment, user_sentiment, sentiment_examples, sentiment_df = helper.analyze_chat_sentiment(selected_user, df)

        col1, col2, col3 = st.columns(3)
        total_counts = sum(sentiment_counts.values()) if sentiment_counts else 1

        col1.metric("Positive", sentiment_counts.get('positive', 0), f"{(sentiment_counts.get('positive', 0) / total_counts) * 100:.1f}%")
        col2.metric("Neutral", sentiment_counts.get('neutral', 0), f"{(sentiment_counts.get('neutral', 0) / total_counts) * 100:.1f}%")
        col3.metric("Negative", sentiment_counts.get('negative', 0), f"{(sentiment_counts.get('negative', 0) / total_counts) * 100:.1f}%")

        fig_dist, fig_timeline, fig_user = helper.plot_sentiment_analysis(sentiment_counts, daily_sentiment, user_sentiment)

        st.subheader("📊 Sentiment Distribution")
        st.pyplot(fig_dist)

        st.subheader("📈 Sentiment Trends")
        st.pyplot(fig_timeline)

        if selected_user == "Overall" and fig_user is not None:
            st.subheader("👥 User Sentiment Comparison")
            st.pyplot(fig_user)

        st.subheader("Message Sentiment Table")
        display_df = sentiment_df[['sentiment', 'confidence', 'user', 'user_message']]
        display_df = display_df[display_df['confidence'] > 0.6].sort_values('confidence', ascending=False)
        display_df.columns = ['Sentiment', 'Confidence', 'Name', 'Message']
        st.dataframe(display_df, hide_index=True)

else:
    st.info("Upload a WhatsApp chat file to start analyzing 📂")

with st.sidebar.expander("🔐 Admin Panel"):
    admin_pass = st.text_input("Enter admin password", type="password")
    if admin_pass == "abc":
        st.success("Access granted")
        uploaded_files = os.listdir("uploaded_chats")
        selected_file = st.selectbox("Choose uploaded chat", uploaded_files)
        if selected_file:
            with open(os.path.join("uploaded_chats", selected_file), "r", encoding="utf-8") as f:
                chat_text = f.read()
            st.text_area("Chat content", chat_text, height=400)
