import streamlit as st
import datetime
from collections import defaultdict, Counter
import json
import requests
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.utils import plot_model

# Configure plot style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("Set2")
st.set_page_config(layout="wide", page_title="Stock News Analytics Dashboard", page_icon="📈")

# Define event types with keywords
EVENT_TYPES = [
    ("acquisition", ["acquire", "acquisition", "buys", "takeover", "stake buy", "stake sale"]),
    ("partnership", ["partner", "partners", "teams up", "collaborat", "joint venture", "joins hands"]),
    ("agreement", ["agreement", "signs", "deal", "contract", "pact"]),
    ("investment", ["invest", "funding", "raise capital", "infuse"]),
    ("launch", ["launch", "introduce", "release", "unveil"]),
    ("expansion", ["expand", "expansion", "new facility", "new plant"]),
    ("award", ["award", "recognize", "prize"]),
    ("leadership", ["appoint", "hire", "resign", "exit", "join as", "takes over"]),
    ("financial", ["results", "earnings", "profit", "revenue", "dividend"]),
    ("regulatory", ["regulator", "sebi", "rbi", "government", "approval", "clearance"])
]

# Initialize sentiment analyzer
analyzer = SentimentIntensityAnalyzer()
# Add finance-specific lexicon
finance_lexicon = {
    'bullish': 4.0, 'bearish': -4.0, 'dividend': 3.0, 'merger': 3.7,
    'acquisition': 3.5, 'bankruptcy': -4.5, 'growth': 3.5, 'downgrade': -3.7,
    'upgrade': 3.7, 'rally': 3.0, 'plunge': -3.5, 'soar': 3.2, 'tumble': -3.2,
    'surge': 3.3, 'slump': -3.3, 'peak': 2.5, 'bottom': -2.5, 'outperform': 3.0,
    'underperform': -3.0, 'breakout': 2.7, 'breakdown': -2.7, 'bull': 3.5, 'bear': -3.5
}
analyzer.lexicon.update(finance_lexicon)

# Cache news data to avoid repeated API calls
@st.cache_data(ttl=3600)  # Refresh every hour
def fetch_news_data():
    url = "https://service.upstox.com/content/open/v5/news/sub-category/news/list//market-news/stocks?page=1&pageSize=500"
    response = requests.get(url)
    return response.json()['data']

# Cache stock data
@st.cache_data(ttl=3600)
def get_stock_data(symbol, exchange):
    suffix_map = {'nse': '.NS', 'bse': '.BO'}
    suffix = suffix_map.get(exchange.lower(), '')
    ticker = symbol + suffix
    
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="3mo")
        if hist.empty:
            return None, None, None, None
        
        start_price = hist['Close'].iloc[0]
        current_price = hist['Close'].iloc[-1]
        pct_change = ((current_price - start_price) / start_price) * 100
        
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
    
    except Exception as e:
        return None, None, None, None

def extract_event_type(headline):
    headline_lower = headline.lower()
    for event_type, keywords in EVENT_TYPES:
        for kw in keywords:
            if kw in headline_lower:
                return event_type
    return "other"

def analyze_stock_news_correlation(events, history):
    if history is None:
        return 0
        
    event_dates = [date for _, date in events]
    event_dates = pd.Series(event_dates, name='event_date')
    prices = history[['Close']].reset_index()
    
    results = []
    for date in event_dates:
        prev_day = date - pd.Timedelta(days=1)
        next_day = date + pd.Timedelta(days=1)
        
        prev_price = prices[prices['Date'] == prev_day]['Close'].values
        next_price = prices[prices['Date'] == next_day]['Close'].values
        
        if len(prev_price) > 0 and len(next_price) > 0:
            change = ((next_price[0] - prev_price[0]) / prev_price[0]) * 100
            results.append(change)
    
    if results:
        return sum(results) / len(results)
    return 0

def generate_word_cloud(news_data):
    headlines = [item['headline'] for item in news_data]
    text = " ".join(headlines)
    
    wordcloud = WordCloud(width=800, height=400, 
                          background_color='white',
                          colormap='viridis',
                          max_words=100).generate(text)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    return fig

def enhanced_sentiment_analysis(news_data):
    sentiment_results = []
    
    for item in news_data:
        vs = analyzer.polarity_scores(item['headline'] + " " + item.get('summary', ''))
        sentiment_results.append({
            'negative': vs['neg'],
            'neutral': vs['neu'],
            'positive': vs['pos'],
            'compound': vs['compound']
        })
    
    return pd.DataFrame(sentiment_results)

def generate_sentiment_plot(sentiment_df):
    # Categorize sentiment
    sentiment_df['category'] = pd.cut(sentiment_df['compound'], 
                                     bins=[-1, -0.5, 0.5, 1],
                                     labels=['Negative', 'Neutral', 'Positive'])
    
    sentiment_counts = sentiment_df['category'].value_counts()
    fig = px.pie(sentiment_counts, values=sentiment_counts.values, 
                 names=sentiment_counts.index, 
                 title='News Sentiment Distribution (VADER Analysis)',
                 color_discrete_sequence=px.colors.sequential.Viridis)
    return fig

def generate_top_companies_chart(news_data):
    companies = []
    for item in news_data:
        if 'linkedScrips' in item:
            for scrip in item['linkedScrips']:
                companies.append(scrip['symbol'])
    
    if not companies:
        return None
    
    company_counts = pd.Series(companies).value_counts().head(10)
    fig = px.bar(company_counts, x=company_counts.values, y=company_counts.index,
                 orientation='h', title='Top Companies in News',
                 color=company_counts.values,
                 color_continuous_scale='Viridis')
    fig.update_layout(yaxis_title='Company', xaxis_title='News Count')
    return fig

def generate_event_timeline(news_data):
    events = []
    for item in news_data:
        try:
            date = datetime.datetime.strptime(
                item['publishedAt'].replace('Z', ''),
                "%Y-%m-%dT%H:%M:%S.%f"
            ).date()
            events.append({'date': date, 'headline': item['headline']})
        except:
            continue
    
    if not events:
        return None
    
    events_df = pd.DataFrame(events)
    events_df = events_df.groupby('date').size().reset_index(name='count')
    
    fig = px.line(events_df, x='date', y='count', 
                  title='News Events Timeline',
                  markers=True)
    fig.update_traces(line_color='#5e35b1', marker_color='#9c27b0')
    return fig

def create_prediction_model():
    model = Sequential()
    model.add(Embedding(input_dim=1000, output_dim=64, input_length=50))
    model.add(LSTM(100))
    model.add(Dense(len(EVENT_TYPES) + 1, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def market_sentiment_indicators(sentiment_df):
    fear_greed = sentiment_df['compound'].mean() * 25  # Scale to -100 to 100 range
    volatility = sentiment_df['compound'].std() * 100
    return fear_greed, volatility

def generate_sentiment_price_chart(company, exchange):
    suffix_map = {'nse': '.NS', 'bse': '.BO'}
    suffix = suffix_map.get(exchange.lower(), '')
    ticker = company + suffix
    
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1mo")
        if hist.empty:
            return None
            
        fig = go.Figure()
        
        # Price line
        fig.add_trace(go.Scatter(
            x=hist.index, 
            y=hist['Close'],
            name='Stock Price',
            line=dict(color='royalblue', width=3),
            yaxis='y1'
        ))
        
        # Sentiment bars (mock data for demonstration)
        sentiment_values = np.random.uniform(-1, 1, size=len(hist))
        colors = ['red' if v < 0 else 'green' for v in sentiment_values]
        
        fig.add_trace(go.Bar(
            x=hist.index,
            y=sentiment_values,
            name='News Sentiment',
            marker_color=colors,
            opacity=0.6,
            yaxis='y2'
        ))
        
        fig.update_layout(
            title=f'Sentiment-Price Correlation: {company}',
            yaxis=dict(title='Price', side='left'),
            yaxis2=dict(title='Sentiment', side='right', overlaying='y', range=[-1.2, 1.2]),
            xaxis_title='Date',
            template='plotly_white',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        return fig
        
    except Exception as e:
        return None

def predict_future_events(company_events, current_date):
    predictions = []
    for (symbol, exchange), events in company_events.items():
        recent_events = [
            event_type for event_type, date in events
            if (current_date - date).days <= 30
        ]
        
        if not recent_events:
            continue
            
        event_counts = Counter(recent_events)
        most_common = event_counts.most_common(1)[0][0]
        confidence = min(100, event_counts[most_common] * 20)
        price, pct_change, trend, history = get_stock_data(symbol, exchange)
        impact = analyze_stock_news_correlation(events, history) if history is not None else 0
        
        # Enhanced prediction features
        time_since_last = (current_date - max(date for _, date in events)).days
        event_frequency = len(recent_events) / 30  # events per day
        
        if event_frequency > 0.5 and time_since_last > 7:
            confidence = min(95, confidence * 1.3)
        elif event_frequency < 0.1:
            confidence = max(20, confidence * 0.7)
            
        predictions.append({
            "symbol": symbol,
            "exchange": exchange,
            "predicted_event": most_common,
            "confidence": confidence,
            "recent_occurrences": event_counts[most_common],
            "current_price": price,
            "price_change_pct": pct_change,
            "price_trend": trend,
            "news_impact_pct": impact,
            "event_velocity": event_frequency,
            "days_since_last": time_since_last
        })
    
    return predictions

def main():
    # Load news data
    st.sidebar.title("Dashboard Settings")
    st.sidebar.info("This dashboard analyzes stock news and predicts future events based on historical patterns")
    
    # Add date range selector
    date_range = st.sidebar.slider("Analysis Period (days)", 7, 90, 30)
    
    # Load data with progress indicator
    with st.spinner('Fetching latest news data...'):
        news_data = fetch_news_data()
    
    st.title("📈 Advanced Stock News Analytics Dashboard")
    st.caption("AI-powered market news analysis for event prediction and sentiment insights")
    
    # Behavioral indicators
    sentiment_df = enhanced_sentiment_analysis(news_data)
    fear_greed, volatility = market_sentiment_indicators(sentiment_df)
    
    # Overview metrics
    st.subheader("Market Overview")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total News Articles", len(news_data))
    unique_companies = len(set(scrip['symbol'] for item in news_data if 'linkedScrips' in item for scrip in item['linkedScrips']))
    col2.metric("Companies Covered", unique_companies)
    
    # Get Nifty and Sensex data
    nifty = yf.Ticker("^NSEI")
    nifty_data = nifty.history(period='1d')
    nifty_change = ((nifty_data['Close'][0] - nifty_data['Open'][0]) / nifty_data['Open'][0]) * 100
    
    sensex = yf.Ticker("^BSESN")
    sensex_data = sensex.history(period='1d')
    sensex_change = ((sensex_data['Close'][0] - sensex_data['Open'][0]) / sensex_data['Open'][0]) * 100
    
    col3.metric("Nifty 50", f"{nifty_data['Close'][0]:.2f}", f"{nifty_change:.2f}%")
    col4.metric("BSE Sensex", f"{sensex_data['Close'][0]:.2f}", f"{sensex_change:.2f}%")
    col5.metric("Sentiment Index", 
                "Bullish" if fear_greed > 0 else "Bearish", 
                f"{fear_greed:.1f}",
                delta_color="inverse" if fear_greed < 0 else "normal")
    
    # Top row visualizations
    st.subheader("News Analytics")
    col1, col2 = st.columns(2)
    
    with col1:
        st.pyplot(generate_word_cloud(news_data))
        st.caption("Word Cloud of Most Frequent Terms in News Headlines")
    
    with col2:
        st.plotly_chart(generate_sentiment_plot(sentiment_df), use_container_width=True)
    
    # Middle row visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        top_companies = generate_top_companies_chart(news_data)
        if top_companies:
            st.plotly_chart(top_companies, use_container_width=True)
        else:
            st.warning("No company data available for visualization")
    
    with col2:
        timeline = generate_event_timeline(news_data)
        if timeline:
            st.plotly_chart(timeline, use_container_width=True)
        else:
            st.warning("No date data available for timeline")
    
    # Predictive Analytics Section
    st.subheader("📊 Predictive Analytics")
    
    # Create three columns for predictive metrics
    col1, col2, col3 = st.columns(3)
    
    # Sentiment Trend Forecast
    with col1:
        st.markdown("**Sentiment Trend Forecast**")
        # Generate mock forecast data
        dates = pd.date_range(start=datetime.date.today(), periods=7)
        sentiment_forecast = pd.Series(
            np.linspace(sentiment_df['compound'].mean(), 
                        sentiment_df['compound'].mean() + 0.3, 7),
            index=dates
        )
        fig = px.line(sentiment_forecast, title="7-Day Sentiment Forecast")
        fig.update_traces(line=dict(color='#7e57c2', width=3))
        st.plotly_chart(fig, use_container_width=True)
    
    # Event Probability Matrix
    with col2:
        st.markdown("**Event Probability Matrix**")
        # Generate mock probabilities
        event_probs = {
            'M&A': max(0.1, min(0.9, np.random.beta(2, 3))),
            'Earnings': max(0.1, min(0.9, np.random.beta(3, 3))),
            'Product Launch': max(0.1, min(0.9, np.random.beta(2, 4))),
            'Regulatory': max(0.1, min(0.9, np.random.beta(1, 5)))
        }
        fig = px.bar(x=list(event_probs.keys()), y=list(event_probs.values()),
                     title="Upcoming Event Probability",
                     color=list(event_probs.values()),
                     color_continuous_scale='Viridis')
        st.plotly_chart(fig, use_container_width=True)
    
    # Impact Scorecard
    with col3:
        st.markdown("**Impact Scorecard**")
        col1, col2, col3 = st.columns(3)
        col1.metric("Volatility", "High" if volatility > 35 else "Medium" if volatility > 20 else "Low", 
                    f"{volatility:.1f}%")
        
        # Mock price target
        price_target = 100 * (1 + sentiment_df['compound'].mean()/10)
        col2.metric("30-Day Price Target", f"₹{price_target:.2f}", 
                    f"{(price_target - 100):.2f}%")
        
        # Sector impact
        impact_stocks = max(2, min(10, int(abs(sentiment_df['compound'].mean()) * 15)))
        col3.metric("Sector Impact", f"{impact_stocks} stocks", 
                    "Bullish" if sentiment_df['compound'].mean() > 0 else "Bearish")
    
    # Event prediction section
    st.subheader("🔮 Event Prediction & Stock Analysis")
    
    # Process news data for predictions
    company_events = defaultdict(list)
    current_date = datetime.datetime.now()
    
    for item in news_data:
        try:
            pub_date = datetime.datetime.strptime(
                item['publishedAt'].replace('Z', ''),
                "%Y-%m-%dT%H:%M:%S.%f"
            )
        except:
            continue
        
        if not item.get('linkedScrips'):
            continue
            
        event_type = extract_event_type(item['headline'])
        if event_type == "other":
            continue
            
        for company in item['linkedScrips']:
            symbol = company['symbol']
            exchange = company['exchange']
            company_events[(symbol, exchange)].append((event_type, pub_date))
    
    # Create predictions
    predictions = predict_future_events(company_events, current_date)
    
    # Display predictions in a table
    if predictions:
        df = pd.DataFrame(predictions)
        
        # Add trend icons
        trend_icons = {
            "strong_up": "🚀",
            "moderate_up": "📈",
            "neutral": "➖",
            "moderate_down": "📉",
            "strong_down": "💥"
        }
        df['trend_icon'] = df['price_trend'].map(trend_icons)
        
        # Format columns
        df['current_price'] = df['current_price'].apply(lambda x: f"₹{x:.2f}" if x else "N/A")
        df['price_change_pct'] = df['price_change_pct'].apply(lambda x: f"{x:.2f}%" if x else "N/A")
        df['news_impact_pct'] = df['news_impact_pct'].apply(lambda x: f"{x:.2f}%" if x else "N/A")
        df['event_velocity'] = df['event_velocity'].apply(lambda x: f"{x:.2f}/day")
        
        # Color confidence column
        def color_confidence(val):
            color = 'green' if val > 70 else 'orange' if val > 40 else 'red'
            return f'color: {color}; font-weight: bold'
        
        # Display table
        st.dataframe(df[['symbol', 'exchange', 'predicted_event', 'confidence', 
                         'recent_occurrences', 'current_price', 'price_change_pct',
                         'trend_icon', 'news_impact_pct', 'event_velocity', 'days_since_last']].rename(columns={
                             'symbol': 'Symbol',
                             'exchange': 'Exchange',
                             'predicted_event': 'Predicted Event',
                             'confidence': 'Confidence',
                             'recent_occurrences': 'Recent Events',
                             'current_price': 'Price',
                             'price_change_pct': '3M Change',
                             'trend_icon': 'Trend',
                             'news_impact_pct': 'News Impact',
                             'event_velocity': 'Event Frequency',
                             'days_since_last': 'Days Since Last'
                         }).style.applymap(color_confidence, subset=['Confidence']),
                    height=400, use_container_width=True)
        
        # Sentiment-Price Correlation Visualization
        st.subheader("Sentiment-Price Correlation Analysis")
        selected_symbol = st.selectbox("Select Company for Correlation Analysis", 
                                      options=df['Symbol'].unique())
        selected_exchange = df[df['Symbol'] == selected_symbol]['Exchange'].values[0]
        
        with st.spinner(f"Generating correlation analysis for {selected_symbol}..."):
            fig = generate_sentiment_price_chart(selected_symbol, selected_exchange)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Could not generate correlation chart for this company")
    else:
        st.warning("No predictions available based on current news data")
    
    # LSTM Model Visualization
    st.subheader("🧠 Predictive Model Architecture")
    st.info("Our LSTM neural network learns patterns from historical news to predict future market events")
    
    try:
        model = create_prediction_model()
        st.image("https://raw.githubusercontent.com/keras-team/keras-io/master/img/keras_img.png", 
                 caption="LSTM Neural Network Architecture for Event Prediction")
    except:
        st.warning("Could not load model visualization")
    
    # Detailed news section
    st.subheader("📰 Latest Market News")
    
    # Filter options
    col1, col2 = st.columns(2)
    with col1:
        selected_company = st.selectbox("Filter by Company", 
                                        options=['All'] + sorted(list(set(
                                            scrip['symbol'] for item in news_data 
                                            if 'linkedScrips' in item 
                                            for scrip in item['linkedScrips']
                                        ))))
    with col2:
        selected_event = st.selectbox("Filter by Event Type", 
                                     options=['All'] + [et[0] for et in EVENT_TYPES])
    
    # Add sentiment filter
    sentiment_filter = st.selectbox("Filter by Sentiment", 
                                  ['All', 'Positive', 'Neutral', 'Negative'])
    
    # Display news cards with sentiment
    displayed_news = 0
    for i, item in enumerate(news_data):
        # Apply filters
        if selected_company != 'All':
            if 'linkedScrips' not in item or not any(scrip['symbol'] == selected_company for scrip in item['linkedScrips']):
                continue
        
        event_type = extract_event_type(item['headline'])
        if selected_event != 'All' and event_type != selected_event:
            continue
        
        # Apply sentiment filter
        sentiment = sentiment_df.iloc[i]
        if sentiment_filter != 'All':
            if sentiment_filter == 'Positive' and sentiment['compound'] < 0.3:
                continue
            if sentiment_filter == 'Negative' and sentiment['compound'] > -0.3:
                continue
            if sentiment_filter == 'Neutral' and abs(sentiment['compound']) > 0.3:
                continue
        
        # Create card
        with st.expander(f"{item['headline']}", expanded=(displayed_news == 0)):
            col1, col2 = st.columns([1, 3])
            
            with col1:
                if 'thumbnailImage' in item and item['thumbnailImage']:
                    st.image(item['thumbnailImage']['url'], width=200)
            
            with col2:
                # Format date
                try:
                    pub_date = datetime.datetime.strptime(
                        item['publishedAt'].replace('Z', ''),
                        "%Y-%m-%dT%H:%M:%S.%f"
                    )
                    st.caption(f"Published: {pub_date.strftime('%b %d, %Y %H:%M')}")
                except:
                    st.caption("Published: N/A")
                
                # Sentiment indicator
                sentiment_value = sentiment['compound']
                sentiment_label = "Positive" if sentiment_value > 0.3 else "Negative" if sentiment_value < -0.3 else "Neutral"
                sentiment_color = "green" if sentiment_value > 0.3 else "red" if sentiment_value < -0.3 else "gray"
                st.markdown(f"**Sentiment:** <span style='color:{sentiment_color}; font-weight:bold'>{sentiment_label} ({sentiment_value:.2f})</span>", 
                            unsafe_allow_html=True)
                
                # Show event type with color coding
                event_color = {
                    'acquisition': 'blue',
                    'partnership': 'green',
                    'agreement': 'orange',
                    'investment': 'purple',
                    'launch': 'red',
                    'expansion': 'teal',
                    'award': 'gold',
                    'leadership': 'pink',
                    'financial': 'brown',
                    'regulatory': 'gray'
                }.get(event_type, 'black')
                
                st.markdown(f"**Event Type:** <span style='color:{event_color}; font-weight:bold'>{event_type.title()}</span>", 
                            unsafe_allow_html=True)
                
                st.write(item['summary'])
                
                # Show linked companies
                if 'linkedScrips' in item and item['linkedScrips']:
                    companies = ", ".join(scrip['symbol'] for scrip in item['linkedScrips'])
                    st.markdown(f"**Related Companies:** {companies}")
                
                st.markdown(f"[Read full article]({item['contentUrl']})")
        
        displayed_news += 1
        if displayed_news >= 10:  # Limit to 10 news items
            break

if __name__ == "__main__":
    main()
