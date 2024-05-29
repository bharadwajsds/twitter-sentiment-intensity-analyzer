import sqlite3
from textblob import TextBlob
import pandas as pd
import streamlit as st
import cleantext
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
import emoji
from googletrans import Translator
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

nltk.download('vader_lexicon')
st.set_option('deprecation.showPyplotGlobalUse', False)

conn = sqlite3.connect('analysis_results.db')
c = conn.cursor()

c.execute('''CREATE TABLE IF NOT EXISTS SentimentResults
             (ID INTEGER PRIMARY KEY AUTOINCREMENT,
             TextColumn TEXT,
             SentimentScore REAL)''')

st.title('Twitter Sentiment Analysis, Emotion Classification, and Topic Modeling')



# Sentiment analysis using TextBlob
with st.expander('Analyze Text Sentiment'):
    text = st.text_input('Text here: ')
    if text:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        if polarity > 0:
            emoji_sentiment = emoji.emojize(":grinning_face_with_smiling_eyes:")
        elif polarity < 0:
            emoji_sentiment = emoji.emojize(":disappointed_face:")
        else:
            emoji_sentiment = emoji.emojize(":neutral_face:")
        
        st.write('Sentiment Polarity:', round(polarity, 2), emoji_sentiment)
        st.write('Sentiment Subjectivity:', round(subjectivity, 2))

# Emotion classification using NLTK's VADER
with st.expander('Analyze Text Emotion'):
    sia = SentimentIntensityAnalyzer()
    text_for_emotion = st.text_input('Text for Emotion Analysis: ')
    if text_for_emotion:
        scores = sia.polarity_scores(text_for_emotion)
        if scores['compound'] >= 0.05:
            emotion = 'Positive'
            emoji_emotion = emoji.emojize(":smiling_face_with_smiling_eyes:")
        elif scores['compound'] <= -0.05:
            emotion = 'Negative'
            emoji_emotion = emoji.emojize(":pensive_face:")
        else:
            emotion = 'Neutral'
            emoji_emotion = emoji.emojize(":expressionless_face:")
        st.write('Emotion:', emotion, emoji_emotion)

# Language Translation and Sentiment Analysis on Translated Text
with st.expander('Translate and Analyze Text Sentiment'):
    translator = Translator()
    text_to_translate = st.text_input('Text to Translate:')
    source_language = st.selectbox('Select Source Language:', ('te', 'hi', 'ta','en'))  
    
    target_language = st.selectbox('Select Target Language:', ('en', 'te', 'hi', 'de'))
    
    if text_to_translate:
        translated_text = translator.translate(text_to_translate, dest=target_language).text
        st.write('Translated Text:', translated_text)
        
        blob_translated = TextBlob(translated_text)
        polarity_translated = blob_translated.sentiment.polarity
        subjectivity_translated = blob_translated.sentiment.subjectivity
        
        if polarity_translated > 0:
            emoji_sentiment_translated = emoji.emojize(":grinning_face_with_smiling_eyes:")  # Positive emoji
        elif polarity_translated < 0:
            emoji_sentiment_translated = emoji.emojize(":disappointed_face:")  # Negative emoji
        else:
            emoji_sentiment_translated = emoji.emojize(":neutral_face:")  # Neutral emoji
        
        st.write('Sentiment Polarity (Translated):', round(polarity_translated, 2), emoji_sentiment_translated)
        st.write('Sentiment Subjectivity (Translated):', round(subjectivity_translated, 2))


# Clean text input
pre = st.text_input('Clean Text:')
if pre:
    st.write(cleantext.clean(pre, clean_all=False, extra_spaces=True,
                             stopwords=True, lowercase=True, numbers=True, punct=True))

# Analyzing CSV file for sentiment analysis and LDA
def perform_sentiment_analysis(text):
    analyzer = SentimentIntensityAnalyzer()
    if isinstance(text, str):
        sentiment_score = analyzer.polarity_scores(text)
        return sentiment_score
    else:
        return {'compound': 0.0}  # Return neutral score for non-string values

def perform_lda(text_data, num_topics=5):
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    dtm = vectorizer.fit_transform(text_data)
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(dtm)
    return lda, vectorizer

def plot_top_words(lda_model, vectorizer, num_topics=5, num_words=10):
    for index, topic in enumerate(lda_model.components_):
        st.write(f"Topic {index + 1}:")
        top_words_idx = topic.argsort()[-num_words:][::-1]
        top_words = [vectorizer.get_feature_names_out()[i] for i in top_words_idx]
        st.write(', '.join(top_words))

        # Plotting the top words for each topic
        plt.figure(figsize=(8, 6))
        sns.barplot(x=top_words, y=topic[top_words_idx], palette='viridis')
        plt.title(f"Topic {index + 1} - Top {num_words} words")
        plt.xlabel("Words")
        plt.ylabel("Weights")
        plt.xticks(rotation=45)
        st.pyplot()


# Streamlit app for CSV analysis
st.write('Sentiment Analysis, Emotion Classification, and Topic Modeling on CSV File')

num_topics_lda = 5  # Define the number of topics for LDA

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Preview of the CSV file")
    st.write(df.head())

    text_column = st.selectbox("Select the text column for analysis", df.columns)
    num_rows = st.number_input("Enter the number of rows to analyze", min_value=1, max_value=len(df), value=len(df))

    if st.button("Perform Analysis"):
        with st.spinner('Performing Analysis...'):
            df_subset = df.head(num_rows)
            df_subset['Sentiment Score'] = df_subset[text_column].apply(lambda x: perform_sentiment_analysis(str(x))['compound'])
            text_data_for_lda = df_subset[text_column].dropna()
            lda_model, vectorizer = perform_lda(text_data_for_lda)
            st.success('Analysis completed for {} rows!'.format(num_rows))

            # Inserting data into the database table
        for index, row in df_subset.head(num_rows).iterrows():
            c.execute("INSERT INTO SentimentResults (TextColumn, SentimentScore) VALUES (?, ?)", (row[text_column], row['Sentiment Score']))
            conn.commit()

        st.subheader("Resultant DataFrame with Sentiment Scores")
        st.write(df_subset)
        
        st.subheader(f"Top {num_topics_lda} Topics from LDA")
        plot_top_words(lda_model, vectorizer, num_topics_lda)

        st.subheader("Bar Graph: Sentiment Distribution")
        plt.figure(figsize=(8, 6))
        sns.countplot(x=df_subset['Sentiment Score'].apply(lambda score: 'Positive' if score >= 0.05 else ('Negative' if score <= -0.05 else 'Neutral')))
        plt.xlabel('Sentiment')
        plt.ylabel('Count')
        st.pyplot()

        st.subheader("Pie Chart: Sentiment Distribution")
        sentiment_counts = df_subset['Sentiment Score'].apply(lambda score: 'Positive' if score >= 0.05 else ('Negative' if score <= -0.05 else 'Neutral')).value_counts()
        plt.figure(figsize=(8, 6))
        plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=140)
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        st.pyplot()


c.execute("SELECT * FROM SentimentResults")
data_from_db = c.fetchall()
st.write("Data retrieved from the database:")
st.write(data_from_db)

conn.close()
