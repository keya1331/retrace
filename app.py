import streamlit as st
import preprocessor
import helper
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

st.set_page_config(page_title="WhatsApp Chat Analyzer", page_icon="💬")

st.sidebar.title("Whatsapp Chat Analyzer")

st.markdown(
    """
    <div style="text-align: center;">
        <h1 style="color: #ff4b4b; font-family: Arial, sans-serif;">
            Welcome to the <span style="color: #ff914d;">WhatsApp Chat Analyzer</span> 📊
        </h1>
        <p style="font-size: 18px; font-family: 'Trebuchet MS', sans-serif; color: #ddd;">
            Upload your chat file to get insightful analytics on your conversations.
        </p>
         <p style="font-size: 18px; font-family: 'Trebuchet MS', sans-serif; color: #ddd;">
            Brought to you by Mohammed Fowzan.
        </p>
    </div>
    
    <style>
    .footer {
        position: fixed;
        bottom: 10px;
        right: 10px;
        font-size: 24px;
        color: WHITE;
    }
    </style>
    <div class="footer">© Fowzan</div>
    """,
    unsafe_allow_html=True
)


uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
    st.success("File uploaded successfully! Processing...")
    bytes_data = uploaded_file.getvalue()
    data=bytes_data.decode("utf-8")
    # st.text(data)
    df = preprocessor.preprocess(data)

    st.dataframe(df)
    #fetching unique users
    user_list=df['user'].unique().tolist()
    user_list.remove("3rd yr CSE divA announcement")
    user_list.sort()
    user_list.insert(0,"Overall")
    selected_user=st.sidebar.selectbox("SHOW ANALYSIS W.R.T. ",user_list)

    if st.sidebar.button("SHOW ANALYSIS!"):
        st.header("DETAILED ANALYSIS OF THE Whatsapp CHAT!")
        num_messages,total_words,media_shared,link_shared = helper.fetch_stats(selected_user,df)
        c1,c2,c3,c4 = st.columns(4)

        with c1:
            st.header("TOTAL MESSAGES:")
            st.title(num_messages)
        with c2:
            st.header("TOTAL WORDS:")
            st.title(total_words)
        with c3:
            st.header("MEDIA SHARED:")
            st.title(media_shared)
        with c4:
            st.header("LINKS SHARED:")
            st.title(link_shared)

        
        
        st.title("📝 Term Significance Analysis (TF-IDF)")
        st.markdown("""
            Term Frequency-Inverse Document Frequency (TF-IDF) helps identify the most unique 
            and important words for each person in the chat.
        """)

        # Perform TF-IDF analysis
        tfidf_results, full_tfidf_df = helper.perform_tfidf_analysis(selected_user, df)

        # Visualize results
        tfidf_fig = helper.plot_tfidf_results(tfidf_results, selected_user)
        st.pyplot(tfidf_fig)

        # Show the detailed data
        if not isinstance(full_tfidf_df, type(None)) and not full_tfidf_df.empty:
            st.subheader("Detailed TF-IDF Scores")
            st.markdown("""
                This table shows the significance scores of terms in the chat. Higher scores indicate 
                terms that are more unique to a specific user and used frequently by them.
            """)
            st.dataframe(full_tfidf_df.head(20), hide_index=True)
        else:
            st.info("Not enough data for detailed TF-IDF analysis")


        
        st.title("Monthly Timeline")
        timeline = helper.monthly_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(timeline['time'], timeline['user_message'], color='green')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # daily timeline
        st.title("Daily Timeline")
        daily_timeline = helper.daily_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(daily_timeline['only_date'], daily_timeline['user_message'], color='black')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)



        # activity map
        st.title('ACTIVITY MAP')
        c1, c2 = st.columns(2)

        with c1:
            st.header("MOST BUSY DAY")
            busy_day = helper.week_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_day.index, busy_day.values, color='purple')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        with c2:
            st.header("MOST BUSY MONTH")
            busy_month = helper.month_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values, color='orange')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        st.title("WEEKLY ACTIVITY MAP")
        user_heatmap = helper.activity_heatmap(selected_user, df)
        fig, ax = plt.subplots()
        ax = sns.heatmap(user_heatmap)
        st.pyplot(fig)



        if selected_user=="Overall":
            st.title("MOST BUSY USERS")
            x,df1=helper.fetch_most_busy_users(df)
            fig,ax=plt.subplots()
            c1,c2=st.columns(2)
            with c1:
                ax.bar(x.index, x.values, color='red')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with c2:
                st.dataframe(df1)



        #wordcloud
        st.title("WORD CLOUD")
        df_wc=helper.createwordcloud(selected_user, df)
        fig,ax=plt.subplots()
        ax.imshow(df_wc)
        st.pyplot(fig)

        #MOST COMMON WORDS
        st.title("MOST COMMON WORDS")
        most_common_df=helper.most_common_words(selected_user, df)
        fig,ax=plt.subplots()
        ax.barh(most_common_df[0],most_common_df[1])
        st.pyplot(fig)
        plt.xticks(rotation="vertical")
        # st.dataframe(most_common_df)


        #emoji
        emoji_df = helper.emoji_helper(selected_user, df)

        st.title("Emoji Analysis")
        c1, c2 = st.columns(2)
        with c1:
            st.dataframe(emoji_df)
        with c2:
            fig, ax = plt.subplots()
            if isinstance(emoji_df, list):
                emoji_df = pd.DataFrame(emoji_df)

            if not emoji_df.empty and len(emoji_df.columns) > 1:
                ax.pie(emoji_df.iloc[:, 1].head(), labels=emoji_df.iloc[:, 0].head(), autopct="%0.2f")
            else:
                st.warning("Not enough emoji data to generate a pie chart.")

            st.pyplot(fig)

        # Add this in the if st.sidebar.button("SHOW ANALYSIS!") block
        # After your existing analysis sections

        st.title("💭 Sentiment Analysis")

        # Get sentiment analysis results
        sentiment_counts, daily_sentiment, user_sentiment, sentiment_examples, sentiment_df = helper.analyze_chat_sentiment(
            selected_user, df)

        # Display overall metrics
        col1, col2, col3 = st.columns(3)
        col1, col2, col3 = st.columns(3)
        with col1:
            total_counts = sum(sentiment_counts.values()) if sentiment_counts else 1
            positive_pct = (sentiment_counts.get('positive', 0) / total_counts * 100) if total_counts > 0 else 0
            st.metric("Positive Messages",
                    f"{sentiment_counts.get('positive', 0)}",
                    f"{positive_pct:.1f}%")
        with col2:
            neutral_pct = (sentiment_counts.get('neutral', 0) / total_counts * 100) if total_counts > 0 else 0
            st.metric("Neutral Messages",
                    f"{sentiment_counts.get('neutral', 0)}",
                    f"{neutral_pct:.1f}%")
        with col3:
            negative_pct = (sentiment_counts.get('negative', 0) / total_counts * 100) if total_counts > 0 else 0
            st.metric("Negative Messages",
                    f"{sentiment_counts.get('negative', 0)}",
                    f"{negative_pct:.1f}%")

        # Plot visualizations
        fig_dist, fig_timeline, fig_user = helper.plot_sentiment_analysis(
            sentiment_counts, daily_sentiment, user_sentiment)

        # Display visualizations
        st.subheader("📊 Sentiment Distribution")
        st.pyplot(fig_dist)

        st.subheader("📈 Sentiment Trends")
        st.pyplot(fig_timeline)

        if selected_user == "Overall" and fig_user is not None:
            st.subheader("👥 User Sentiment Comparison")
            st.pyplot(fig_user)

        # Display message examples in a dataframe
        st.subheader("Message Sentiment Analysis")
        display_df = sentiment_df[[ 'sentiment', 'confidence','user','user_message']]
        display_df = display_df[display_df['confidence'] > 0.6].sort_values('confidence', ascending=False)
        display_df.columns = ['Sentiment', 'Confidence','Name', 'Message']
        st.dataframe(display_df, hide_index=True)
        
        
        
else:
    st.info("Upload a WhatsApp chat file to start analyzing 📂")
