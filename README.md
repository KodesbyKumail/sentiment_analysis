## Sentiment Analysis ## 
An advanced sentiment analysis application that combines machine learning with emotional text analysis to provide accurate sentiment predictions and rating estimations for laptop reviews.

🎯 Purpose

This project addresses the common challenge of understanding true customer sentiment from product reviews. Traditional sentiment analysis often falls short when reviewers give conflicting signals - for example, writing negatively while giving a high rating, or vice versa. Our enhanced approach combines multiple data sources to provide more nuanced and accurate sentiment analysis.
Key Features

**Enhanced Sentiment Detection:** Combines text emotion analysis with numerical ratings for improved accuracy

**Real-time Analysis:** Interactive web interface for immediate sentiment prediction

**Confidence Scoring:** Provides prediction certainty metrics

**Multi-dimensional Analysis:** Considers text polarity, subjectivity, and linguistic patterns

**Product-aware Predictions:** Takes specific laptop models into account for context

# Tech Stacks Used and their Purpose #
# Frontend #

**Streamlit** - Interactive web application framework
**HTML/CSS** - Styling and layout enhancement

# Machine Learning & Data Processing# 

**scikit-learn** - Machine learning pipeline and preprocessing

**XGBoost** - Gradient boosting for regression and classification

**pandas** - Data manipulation and analysis

**numpy** - Numerical computing support

# Natural Language Processing #

**TextBlob** - Sentiment polarity and subjectivity analysis

**TF-IDF Vectorizer** - Text feature extraction

**Regular Expressions (re)** - Text preprocessing and cleaning

# Starting the Program #

## Install dependencies ##
pip install streamlit pandas numpy scikit-learn xgboost textblob

## Run the application ##
streamlit run app.py

# Use Cases #
**E-commerce Platforms:** Automated review analysis for product insights

**Market Research:** Understanding customer sentiment trends

**Quality Assurance:** Identifying products with sentiment-rating mismatches

**Customer Service:** Prioritizing reviews that need attention
