from urlextract import URLExtract
from wordcloud import WordCloud
from collections import Counter
import pandas as pd
extract=URLExtract()
import numpy as np
import emoji
import matplotlib.pyplot as plt

try:
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    # Check if vader_lexicon is downloaded and download if needed
    try:
        nltk.data.find('vader_lexicon')
    except LookupError:
        nltk.download('vader_lexicon')
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("NLTK not installed. Please run: pip install nltk")

def fetch_stats(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    # fetch the number of messages
    num_messages = df.shape[0]

    # fetch the total number of words
    words = []
    for message in df['user_message']:
        words.extend(message.split())

    # fetch number of media messages
    num_media_messages = df[df['user_message'] == '<Media omitted>\n'].shape[0]

    # fetch number of links shared
    links = []
    for message in df['user_message']:
        links.extend(extract.find_urls(message))

    return num_messages,len(words),num_media_messages,len(links)


def fetch_most_busy_users(df):
    x=df['user'].value_counts().head()
    df=round((df['user'].value_counts() /df.shape[0])*100,2).reset_index().rename(columns={'index':'name','user':'percentage'})
    return x,df


def createwordcloud(selected_user,df):
    f = open('stopword.txt', 'r')
    stop_words = f.read()
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    temp = df[df['user'] != '3rd yr CSE divA announcement']
    temp = temp[temp['user_message'] != '<image omitted>\n']

    def remove_stop_words(message):
        y = []
        for word in message.lower().split():
            if word not in stop_words:
                y.append(word)
        return " ".join(y)

    wc=WordCloud(width=500,height=500,min_font_size=10,background_color='white')
    temp['user_message'] = temp['user_message'].apply(remove_stop_words)
    df_wc = wc.generate(temp['user_message'].str.cat(sep=" "))
    return df_wc



def most_common_words(selected_user,df):
    f = open('stopword.txt','r')
    stop_words = f.read()
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != '3rd yr CSE divA announcement']
    temp = temp[temp['user_message'] != '<image omitted>\n']

    words = []
    for message in temp['user_message']:
        for word in message.lower().split():
            if word not in stop_words:
                words.append(word)

    most_common_df = pd.DataFrame(Counter(words).most_common(20))
    return most_common_df


def emoji_helper(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    emojis = []
    for message in df['user_message']:
        x={e: emoji.EMOJI_DATA[e]['en'] for e in emoji.EMOJI_DATA.keys()}
        emojis.extend([c for c in message if c in x])
    emoji_df = pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis))))
    if emoji_df.empty:
        print("emoji_df is empty")
        return []
    return emoji_df



def monthly_timeline(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    timeline = df.groupby(['year', 'month_num', 'month']).count()['user_message'].reset_index()
    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i] + "-" + str(timeline['year'][i]))
    timeline['time'] = time
    return timeline



def daily_timeline(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    daily_timeline = df.groupby('only_date').count()['user_message'].reset_index()
    return daily_timeline


def week_activity_map(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    return df['day_name'].value_counts()



def month_activity_map(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    return df['month'].value_counts()


def activity_heatmap(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    user_heatmap = df.pivot_table(index='day_name', columns='period', values='user_message', aggfunc='count').fillna(0)
    return user_heatmap



def analyze_chat_sentiment(selected_user, df):
   
    default_counts = {'positive': 1, 'neutral': 1, 'negative': 1}
    default_daily = pd.DataFrame()
    default_user = None
    default_examples = {}
    default_df = pd.DataFrame({
        'user': ['System'],
        'user_message': ['Placeholder message'],
        'sentiment': ['neutral'],
        'confidence': [0.0]
    })
     
    if not NLTK_AVAILABLE:
        print("NLTK is not available. Using default values.")
        return default_counts, default_daily, default_user, default_examples, default_df
     
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
     
    if df.empty:
        print("No data available for selected user.")
        return default_counts, default_daily, default_user, default_examples, default_df
     
    df = df[df['user'] != '3rd yr CSE divA announcement']
    df = df[~df['user_message'].str.contains('<Media omitted>|<image omitted>', na=False)]
     
    if df.empty:
        print("No valid messages for sentiment analysis.")
        return default_counts, default_daily, default_user, default_examples, default_df
     
    try:
        sia = SentimentIntensityAnalyzer()
    except Exception as e:
        print(f"Error initializing SentimentIntensityAnalyzer: {e}")
        return default_counts, default_daily, default_user, default_examples, default_df
     
    results = []
    
    # Analyze each message
    for index, row in df.iterrows():
        message = row['user_message']
         
        if pd.isna(message) or message == '':
            continue
            
        try:
           
            message_str = str(message)
            sentiment = sia.polarity_scores(message_str)
            
            if sentiment['compound'] >= 0.05:
                sentiment_category = 'positive'
            elif sentiment['compound'] <= -0.05:
                sentiment_category = 'negative'
            else:
                sentiment_category = 'neutral'
                 
            results.append({
                'user': row['user'],
                'user_message': message,
                'only_date': row['only_date'] if 'only_date' in row else pd.Timestamp.now().date(),
                'sentiment': sentiment_category,
                'compound': sentiment['compound'],
                'positive': sentiment['pos'],
                'negative': sentiment['neg'],
                'neutral': sentiment['neu'],
                'confidence': abs(sentiment['compound'])
            })
        except Exception as e:
            print(f"Error analyzing message: {e}")
            continue
     
    if not results:
        print("No sentiment results generated.")
        return default_counts, default_daily, default_user, default_examples, default_df
        
    sentiment_df = pd.DataFrame(results)
     
    sentiment_counts = dict(Counter(sentiment_df['sentiment']))
     
    for category in ['positive', 'negative', 'neutral']:
        if category not in sentiment_counts:
            sentiment_counts[category] = 1
        elif sentiment_counts[category] == 0:
            sentiment_counts[category] = 1
     
    if 'only_date' in sentiment_df.columns and not sentiment_df['only_date'].isna().all():
        daily_sentiment = sentiment_df.groupby(['only_date', 'sentiment']).size().unstack(fill_value=0)
         
        for category in ['positive', 'negative', 'neutral']:
            if category not in daily_sentiment.columns:
                daily_sentiment[category] = 0
    else:
        daily_sentiment = pd.DataFrame()
     
    if selected_user == 'Overall':
        user_sentiment = sentiment_df.groupby(['user', 'sentiment']).size().unstack(fill_value=0)
         
        for category in ['positive', 'negative', 'neutral']:
            if category not in user_sentiment.columns:
                user_sentiment[category] = 0
                 
        user_sentiment['total'] = user_sentiment.sum(axis=1)
        user_sentiment.loc[user_sentiment['total'] == 0, 'total'] = 1
        
        for category in ['positive', 'negative', 'neutral']:
            user_sentiment[f'{category}_pct'] = user_sentiment[category] / user_sentiment['total'] * 100
    else:
        user_sentiment = None
     
    sentiment_examples = {}
    for category in ['positive', 'negative', 'neutral']:
        category_df = sentiment_df[sentiment_df['sentiment'] == category]
        if not category_df.empty:
            sentiment_examples[category] = category_df.nlargest(3, 'confidence')[['user', 'user_message', 'compound']]
    
    return sentiment_counts, daily_sentiment, user_sentiment, sentiment_examples, sentiment_df

def plot_sentiment_analysis(sentiment_counts, daily_sentiment, user_sentiment):
  
    fig_dist = plt.figure(figsize=(10, 6))
    
    total_count = sum(sentiment_counts.values())
    if not sentiment_counts or total_count < 3:   
        plt.text(0.5, 0.5, "Not enough sentiment data available", 
                horizontalalignment='center', verticalalignment='center', 
                transform=plt.gca().transAxes, fontsize=14)
    else:
        colors = ['#2ecc71', '#95a5a6', '#e74c3c']  # positive, neutral, negative
        plt.bar(sentiment_counts.keys(), sentiment_counts.values(), color=colors)
        plt.title('Distribution of Message Sentiments', fontsize=16)
        plt.ylabel('Number of Messages', fontsize=12)
        plt.xticks(fontsize=12)
    
    plt.tight_layout()
    
    # Sentiment over time
    fig_timeline = plt.figure(figsize=(12, 6))
    
    if daily_sentiment is None or daily_sentiment.empty or daily_sentiment.shape[0] < 2:
        plt.text(0.5, 0.5, "Not enough data for sentiment timeline", 
                horizontalalignment='center', verticalalignment='center', 
                transform=plt.gca().transAxes, fontsize=14)
    else:
        try:
             
            rolling_window = min(7, len(daily_sentiment))
            daily_sentiment_smooth = daily_sentiment.rolling(rolling_window, min_periods=1).mean()
             
            if not daily_sentiment_smooth.empty and not daily_sentiment_smooth.isna().all().all():
                plt.plot(daily_sentiment_smooth.index, daily_sentiment_smooth['positive'], 
                        color='#2ecc71', label='Positive', linewidth=2)
                plt.plot(daily_sentiment_smooth.index, daily_sentiment_smooth['neutral'], 
                        color='#95a5a6', label='Neutral', linewidth=2)
                plt.plot(daily_sentiment_smooth.index, daily_sentiment_smooth['negative'], 
                        color='#e74c3c', label='Negative', linewidth=2)
                
                plt.title('Sentiment Trends Over Time', fontsize=16)
                plt.xlabel('Date', fontsize=12)
                plt.ylabel('Number of Messages (7-day rolling avg)', fontsize=12)
                plt.legend(fontsize=12)
                plt.xticks(rotation=45)
            else:
                plt.text(0.5, 0.5, "Insufficient data for timeline", 
                        horizontalalignment='center', verticalalignment='center', 
                        transform=plt.gca().transAxes, fontsize=14)
        except Exception as e:
            print(f"Error plotting timeline: {e}")
            plt.text(0.5, 0.5, "Error creating sentiment timeline", 
                    horizontalalignment='center', verticalalignment='center', 
                    transform=plt.gca().transAxes, fontsize=14)
    
    plt.tight_layout()
    
    # User sentiment comparison
    fig_user = None
    
    if user_sentiment is not None and not user_sentiment.empty and user_sentiment['total'].sum() > 3:
        try:
            valid_users = user_sentiment[user_sentiment['total'] > 1]
            if valid_users.empty or len(valid_users) < 2:
                fig_user = plt.figure(figsize=(10, 6))
                plt.text(0.5, 0.5, "Not enough users for sentiment comparison", 
                        horizontalalignment='center', verticalalignment='center', 
                        transform=plt.gca().transAxes, fontsize=14)
                plt.tight_layout()
            else:
                top_users = valid_users.nlargest(min(5, len(valid_users)), 'total')
                fig_user, ax = plt.subplots(figsize=(12, 6))
                 
                bottom = np.zeros(len(top_users))
                
                for category, color in zip(['positive', 'neutral', 'negative'], 
                                          ['#2ecc71', '#95a5a6', '#e74c3c']):
                    if f'{category}_pct' in top_users.columns:
                        ax.bar(top_users.index, top_users[f'{category}_pct'], 
                              bottom=bottom, label=category.capitalize(), color=color)
                        bottom += top_users[f'{category}_pct']
               
                for i, (user, row) in enumerate(top_users.iterrows()):
                    ax.text(i, 105, f"{int(row['total'])} msgs", 
                           ha='center', va='bottom', fontsize=10)
                
                ax.set_ylim(0, 115)   
                ax.set_ylabel('Percentage of Messages', fontsize=12)
                ax.set_title('Sentiment Distribution by User', fontsize=16)
                ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), 
                         ncol=3, fancybox=True, shadow=True)
                plt.xticks(rotation=30, ha='right')
                plt.tight_layout()
        except Exception as e:
            print(f"Error creating user sentiment plot: {e}")
            fig_user = plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, "Error creating user sentiment comparison", 
                    horizontalalignment='center', verticalalignment='center', 
                    transform=plt.gca().transAxes, fontsize=14)
            plt.tight_layout()
    else:
        fig_user = plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, "Not enough data for user sentiment comparison", 
                horizontalalignment='center', verticalalignment='center', 
                transform=plt.gca().transAxes, fontsize=14)
        plt.tight_layout()
    
    return fig_dist, fig_timeline, fig_user


from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def perform_tfidf_analysis(selected_user, df):
    # Filter system messages and media
    df = df[df['user'] != '3rd yr CSE divA announcement']
    df = df[~df['user_message'].str.contains('<Media omitted>|<image omitted>', na=False)]
    df = df[df['user_message'].notna() & (df['user_message'] != '')]
    
    if df.empty or len(df) < 3:
        return None, pd.DataFrame()
    
   
    stop_words = ['the', 'is', 'in', 'to', 'and', 'of', 'a', 'for', 'on', 'with', 'as', 'by', 'at', 'from', 'this']
     
    if selected_user != 'Overall':
        user_df = df[df['user'] == selected_user]
        if user_df.empty:
            return None, pd.DataFrame()
        
        document = ' '.join(user_df['user_message'])
        corpus = [document]
         
        tfidf = TfidfVectorizer(stop_words=stop_words, max_features=100, min_df=1)
         
        try:
            tfidf_matrix = tfidf.fit_transform(corpus)
            feature_names = np.array(tfidf.get_feature_names_out())
            
            scores = np.array(tfidf_matrix.toarray()).flatten()
             
            indices = np.argsort(scores)[::-1]
            top_n = min(30, len(indices))
             
            tfidf_df = pd.DataFrame({
                'term': feature_names[indices[:top_n]],
                'score': scores[indices[:top_n]]
            })
            
            return tfidf_df, tfidf_df
        except Exception as e:
            print(f"Error in TF-IDF calculation for single user: {e}")
            return None, pd.DataFrame()
    else:
        try:
            user_messages = {}
            for user in df['user'].unique():
                if user not in user_messages:
                    user_messages[user] = ""
                user_df = df[df['user'] == user]
                user_messages[user] = ' '.join(user_df['user_message'])
            
            corpus = list(user_messages.values())
            users = list(user_messages.keys())
            
            if not corpus or all(not doc for doc in corpus):
                return None, pd.DataFrame()
             
            tfidf = TfidfVectorizer(stop_words=stop_words, max_features=100, min_df=1)
             
            tfidf_matrix = tfidf.fit_transform(corpus)
            feature_names = np.array(tfidf.get_feature_names_out())
             
            all_results = []
            
            for i, user in enumerate(users):
                user_scores = tfidf_matrix[i].toarray().flatten()
                top_indices = np.argsort(user_scores)[::-1][::-1]  
                
                for idx in top_indices:
                    if user_scores[idx] > 0:   
                        all_results.append({
                            'user': user,
                            'term': feature_names[idx],
                            'score': user_scores[idx]
                        })
             
            tfidf_df = pd.DataFrame(all_results)
             
            tfidf_df = tfidf_df.sort_values(by='score', ascending=False)
             
            top_overall = tfidf_df.nlargest(30, 'score')
            
            return top_overall, tfidf_df
        except Exception as e:
            print(f"Error in TF-IDF calculation for all users: {e}")
            return None, pd.DataFrame()

def plot_tfidf_results(tfidf_df, selected_user):
     
    if tfidf_df is None or tfidf_df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "Not enough data for TF-IDF analysis", 
                horizontalalignment='center', verticalalignment='center', 
                transform=ax.transAxes, fontsize=14)
        plt.tight_layout()
        return fig
    
    try:
        if selected_user != 'Overall':
            plot_df = tfidf_df.nlargest(15, 'score')
            
            fig, ax = plt.subplots(figsize=(12, 8))
            bars = ax.barh(plot_df['term'], plot_df['score'], color='skyblue')
             
            for bar in bars:
                width = bar.get_width()
                ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                        f'{width:.3f}', ha='left', va='center')
            
            plt.title(f'Most Significant Terms for {selected_user}', fontsize=16)
            plt.xlabel('TF-IDF Score', fontsize=12)
            plt.ylabel('Terms', fontsize=12)
            plt.tight_layout()
            
        else:
            top_users = tfidf_df['user'].value_counts().nlargest(5).index.tolist()
            
            if len(top_users) <= 1:
                plot_df = tfidf_df.nlargest(15, 'score')
                
                fig, ax = plt.subplots(figsize=(12, 8))
                bars = ax.barh(plot_df['term'], plot_df['score'], color='skyblue')
                
                for bar in bars:
                    width = bar.get_width()
                    ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                            f'{width:.3f}', ha='left', va='center')
                
                plt.title('Most Significant Terms Overall', fontsize=16)
                plt.xlabel('TF-IDF Score', fontsize=12)
                plt.ylabel('Terms', fontsize=12)
                plt.tight_layout()
            else:
                fig, axes = plt.subplots(len(top_users), 1, figsize=(12, 4 * len(top_users)))
                
                for i, user in enumerate(top_users):
                    user_df = tfidf_df[tfidf_df['user'] == user]
                    user_df = user_df.nlargest(8, 'score')   
                    
                    if len(top_users) == 1:
                        ax = axes
                    else:
                        ax = axes[i]
                    
                    bars = ax.barh(user_df['term'], user_df['score'], color=plt.cm.Accent(i/len(top_users)))
                    
                   
                    for bar in bars:
                        width = bar.get_width()
                        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                                f'{width:.3f}', ha='left', va='center')
                    
                    ax.set_title(f'Significant Terms for {user}', fontsize=14)
                    ax.set_xlabel('TF-IDF Score')
                
                plt.tight_layout()
        
        return fig
    
    except Exception as e:
        print(f"Error plotting TF-IDF results: {e}")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "Error creating TF-IDF visualization", 
                horizontalalignment='center', verticalalignment='center', 
                transform=ax.transAxes, fontsize=14)
        plt.tight_layout()
        return fig