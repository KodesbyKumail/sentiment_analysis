import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor, XGBClassifier
from sklearn.pipeline import Pipeline
from textblob import TextBlob
import re

# Set page configuration
st.set_page_config(
    page_title="Laptop Review Sentiment Analyzer",
    page_icon="💻",
    layout="wide"
)


def preprocess_text(text):
    """Clean text while preserving sentiment indicators"""
    if pd.isna(text):
        return ""

    text = str(text).lower()
    # Keep punctuation that matters for sentiment
    text = re.sub(r'[^a-zA-Z0-9\s!?.]', '', text)
    text = ' '.join(text.split())
    return text


def get_emotion_features(text):
    """Extract emotional indicators from text"""
    if pd.isna(text) or text == "":
        return {'textblob_polarity': 0, 'textblob_subjectivity': 0, 'exclamation_ratio': 0}

    text = str(text)
    blob = TextBlob(text)

    # Count emotional punctuation
    exclamation_ratio = text.count('!') / len(text.split()) if text.split() else 0

    return {
        'textblob_polarity': blob.sentiment.polarity,
        'textblob_subjectivity': blob.sentiment.subjectivity,
        'exclamation_ratio': exclamation_ratio
    }


def create_better_sentiment_labels(df):
    """Create better sentiment labels using text emotion + rating"""
    df['processed_review'] = df['review'].apply(preprocess_text)

    # Get emotional features
    emotion_features = df['review'].apply(get_emotion_features)
    feature_df = pd.DataFrame(emotion_features.tolist())

    # Combine text emotion with rating for better labels
    enhanced_sentiment = []

    for i in range(len(df)):
        text_polarity = feature_df['textblob_polarity'].iloc[i]
        rating = df['rating'].iloc[i]

        # If text and rating agree strongly
        if text_polarity > 0.1 and rating >= 4:
            enhanced_sentiment.append('positive')
        elif text_polarity < -0.1 and rating <= 2:
            enhanced_sentiment.append('negative')
        # If text emotion is strong, trust it over rating
        elif text_polarity > 0.3:
            enhanced_sentiment.append('positive')
        elif text_polarity < -0.3:
            enhanced_sentiment.append('negative')
        # Otherwise use rating with better thresholds
        elif rating >= 3.5:
            enhanced_sentiment.append('positive')
        else:
            enhanced_sentiment.append('negative')

    df['enhanced_sentiment'] = enhanced_sentiment

    # Add emotion features to dataframe
    for col in feature_df.columns:
        df[f'emotion_{col}'] = feature_df[col]

    return df


# Load and prepare data
@st.cache_data
def load_data():
    df = pd.read_csv("laptops_dataset_final_600.csv").dropna(subset=['review', 'rating'])
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce').dropna()
    df = create_better_sentiment_labels(df)
    return df


# Train models
@st.cache_resource
def get_models():
    df = load_data()

    emotion_features = [col for col in df.columns if col.startswith('emotion_')]

    label_encoder = LabelEncoder()
    df['sentiment_encoded'] = label_encoder.fit_transform(df['enhanced_sentiment'])

    # Enhanced preprocessor with emotion features
    preprocessor = ColumnTransformer([
        ('text', TfidfVectorizer(
            max_features=1500,
            ngram_range=(1, 2),
            stop_words='english'
        ), 'processed_review'),
        ('product', OneHotEncoder(handle_unknown='ignore'), ['product_name']),
        ('emotions', 'passthrough', emotion_features)
    ])

    rating_model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', XGBRegressor(objective='reg:squarederror', n_estimators=100))
    ])

    sentiment_model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(eval_metric='logloss', n_estimators=100))
    ])

    X = df[['processed_review', 'product_name'] + emotion_features]
    rating_model.fit(X, df['rating'])
    sentiment_model.fit(X, df['sentiment_encoded'])

    return rating_model, sentiment_model, label_encoder, df


def analyze_emotion(text, product_name, models):
    """Analyze emotion in the review text"""
    processed_text = preprocess_text(text)
    emotion_features = get_emotion_features(text)

    # Create input dataframe
    input_data = pd.DataFrame({
        'processed_review': [processed_text],
        'product_name': [product_name]
    })

    # Add emotion features
    for key, value in emotion_features.items():
        input_data[f'emotion_{key}'] = [value]

    rating_model, sentiment_model, label_encoder, _ = models

    # Get predictions
    sentiment_proba = sentiment_model.predict_proba(input_data)[0]
    sentiment_pred = label_encoder.inverse_transform(sentiment_model.predict(input_data))[0]
    confidence = max(sentiment_proba)

    return sentiment_pred, confidence, emotion_features['textblob_polarity']


# Load models and data
try:
    models = get_models()
    rating_model, sentiment_model, label_encoder, df = models
except FileNotFoundError:
    st.error("Dataset file 'laptops_dataset_final_600.csv' not found.")
    st.stop()

# App title
st.title("💻 Laptop Review Sentiment Analyzer")
st.markdown("Enhanced emotion detection to better understand reviewer feelings!")
st.image("https://images.unsplash.com/photo-1496181133206-80ce9b88a853?w=600&auto=format&fit=crop&q=60&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8M3x8bGFwdG9wfGVufDB8fDB8fHww", width=400)


# Create two columns
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📝 Enter Review Details")

    review_text = st.text_area(
        "Review Text:",
        placeholder="Enter your laptop review here...",
        height=150,
        key="review_input"
    )

    product_names = ['generic_product'] + df['product_name'].unique().tolist()
    product_name = st.selectbox("Product Name:", product_names, key="product_select")

    predict_button = st.button("🔍 Analyze Review", type="primary", key="analyze_button")

with col2:
    st.subheader("📊 Results")

    if predict_button and review_text.strip():
        # Prepare input for rating prediction
        processed_text = preprocess_text(review_text)
        emotion_features = get_emotion_features(review_text)

        input_data = pd.DataFrame({
            'processed_review': [processed_text],
            'product_name': [product_name]
        })

        # Add emotion features
        for key, value in emotion_features.items():
            input_data[f'emotion_{key}'] = [value]

        # Make predictions
        rating_pred = rating_model.predict(input_data)[0]
        rating_clipped = max(1, min(5, round(rating_pred)))

        sentiment, confidence, text_emotion = analyze_emotion(review_text, product_name, models)

        # Display rating with stars
        st.markdown("### 🌟 Rating")
        st.markdown(f"**{rating_clipped}/5**")
        stars = "⭐" * rating_clipped + "☆" * (5 - rating_clipped)
        st.markdown(stars)

        # Display sentiment with emotion indicator
        st.markdown("### 💭 Sentiment")
        color = '#28a745' if sentiment.lower() == 'positive' else '#dc3545'

        # Add emotion indicator
        if text_emotion > 0.3:
            emotion_indicator = "😊 (Very Positive)"
        elif text_emotion > 0.1:
            emotion_indicator = "🙂 (Positive)"
        elif text_emotion < -0.3:
            emotion_indicator = "😞 (Very Negative)"
        elif text_emotion < -0.1:
            emotion_indicator = "😐 (Negative)"
        else:
            emotion_indicator = "😐 (Neutral)"

        st.markdown(
            f'<p style="color: {color}; font-size: 20px; font-weight: bold;">'
            f'{sentiment.upper()}</p>',
            unsafe_allow_html=True
        )
        st.markdown(f"**Emotion:** {emotion_indicator}")
        st.markdown(f"**Confidence:** {confidence:.1%}")

    elif predict_button and not review_text.strip():
        st.warning("Please enter a review text to analyze.")

# Additional info
st.markdown("---")
with st.expander("ℹ️ About This App"):
    st.markdown("""
    This app uses enhanced sentiment analysis that considers:
    - **Text Emotion**: Analyzes the actual emotional tone of words
    - **Rating Context**: Combines rating with text sentiment
    - **Language Patterns**: Uses advanced text processing
    - **Confidence Scoring**: Shows prediction certainty

    The emotion detection helps identify when a review sounds positive/negative 
    regardless of the rating given.
    """)

with st.expander("📈 Dataset Information"):
    st.write(f"Total reviews: {len(df)}")
    st.write(f"Average rating: {df['rating'].mean():.2f}")

    if 'enhanced_sentiment' in df.columns:
        sentiment_counts = df['enhanced_sentiment'].value_counts()
        for sentiment, count in sentiment_counts.items():
            st.write(f"{sentiment.title()}: {count} ({count / len(df) * 100:.1f}%)")