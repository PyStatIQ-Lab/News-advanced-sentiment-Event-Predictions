import streamlit as st
import datetime
from collections import defaultdict, Counter
import requests
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Configure plot style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("Set2")
st.set_page_config(layout="wide", page_title="Stock News Analytics Dashboard", page_icon="📈")

# Define event types with keywords
EVENT_TYPES = [
    ("acquisition", ["acquire", "acquisition", "buys", "takeover"]),
    ("partnership", ["partner", "partners", "teams up"]),
    ("agreement", ["agreement", "signs", "deal"]),
    ("investment", ["invest", "funding", "raise capital"]),
    ("launch", ["launch", "introduce", "release"]),
]

# Initialize sentiment analyzer
analyzer = SentimentIntensityAnalyzer()
finance_lexicon = {
    'bullish': 4.0, 'bearish': -4.0, 'dividend': 3.0,
    'growth': 3.5, 'rally': 3.0, 'plunge': -3.5
}
analyzer.lexicon.update(finance_lexicon)

@st.cache_data(ttl=3600)
def fetch_news_data():
    try:
        url = "https://service.upstox.com/content/open/v5/news/sub-category/news/list//market-news/stocks?page=1&pageSize=100"
        response = requests.get(url, timeout=10)
        return response.json()['data']
    except:
        st.error("Failed to fetch news data")
        return []

@st.cache_data(ttl=3600)
def get_stock_data(symbol, exchange):
    suffix = '.NS' if exchange.lower() == 'nse' else '.BO'
    ticker = symbol + suffix
    
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1mo")
        if hist.empty:
            return None, None, None, None
        
        current_price = hist['Close'].iloc[-1]
        pct_change = ((current_price - hist['Close'].iloc[0]) / hist['Close'].iloc[0]) * 100
        
        if pct_change > 5:
            trend = "strong_up"
        elif pct_change > 1:
            trend = "moderate_up"
        elif pct_change < -5:
            trend = "strong_down"
        elif pct_change < -1:
            trend = "moderate_down"
        else:
            trend = "neutral"
            
        return current_price, pct_change, trend, hist
    except:
        return None, None, None, None

def extract_event_type(headline):
    headline_lower = headline.lower()
    for event_type, keywords in EVENT_TYPES:
        for kw in keywords:
            if kw in headline_lower:
                return event_type
    return "other"

def generate_word_cloud(news_data):
    headlines = [item['headline'] for item in news_data if 'headline' in item]
    if not headlines:
        return None
    text = " ".join(headlines)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    return fig

def enhanced_sentiment_analysis(news_data):
    sentiment_results = []
    for item in news_data:
        if 'headline' not in item:
            continue
        text = item['headline'] + " " + item.get('summary', '')
        vs = analyzer.polarity_scores(text)
        sentiment_results.append({
            'negative': vs['neg'],
            'neutral': vs['neu'],
            'positive': vs['pos'],
            'compound': vs['compound']
        })
    return pd.DataFrame(sentiment_results)

def generate_sentiment_plot(sentiment_df):
    if sentiment_df.empty:
        return None
    sentiment_df['category'] = pd.cut(sentiment_df['compound'], 
                                     bins=[-1, -0.5, 0.5, 1],
                                     labels=['Negative', 'Neutral', 'Positive'])
    sentiment_counts = sentiment_df['category'].value_counts()
    return px.pie(sentiment_counts, values=sentiment_counts.values, 
                 names=sentiment_counts.index, 
                 title='News Sentiment Distribution')

def main():
    st.title("📈 Stock News Analytics Dashboard")
    
    # Load news data
    news_data = fetch_news_data()
    if not news_data:
        st.error("No news data available")
        return
    
    # Sentiment analysis
    sentiment_df = enhanced_sentiment_analysis(news_data)
    
    # Overview metrics
    st.subheader("Market Overview")
    cols = st.columns(4)
    cols[0].metric("Total News", len(news_data))
    cols[1].metric("Positive Sentiment", f"{len(sentiment_df[sentiment_df['compound'] > 0.5])} articles")
    cols[2].metric("Negative Sentiment", f"{len(sentiment_df[sentiment_df['compound'] < -0.5])} articles")
    
    try:
        nifty = yf.Ticker("^NSEI").history(period='1d')
        nifty_change = ((nifty['Close'][0] - nifty['Open'][0]) / nifty['Open'][0]) * 100
        cols[3].metric("Nifty 50", f"{nifty['Close'][0]:.2f}", f"{nifty_change:.2f}%")
    except:
        pass
    
    # Visualizations
    st.subheader("News Analytics")
    col1, col2 = st.columns(2)
    
    with col1:
        wc = generate_word_cloud(news_data)
        if wc:
            st.pyplot(wc)
    
    with col2:
        sentiment_plot = generate_sentiment_plot(sentiment_df)
        if sentiment_plot:
            st.plotly_chart(sentiment_plot, use_container_width=True)
    
    # News display
    st.subheader("Latest Market News")
    for item in news_data[:10]:  # Show first 10 news items
        if 'headline' not in item:
            continue
            
        with st.expander(item['headline']):
            if 'summary' in item:
                st.write(item['summary'])
            if 'contentUrl' in item:
                st.markdown(f"[Read more]({item['contentUrl']})")

if __name__ == "__main__":
    main()
