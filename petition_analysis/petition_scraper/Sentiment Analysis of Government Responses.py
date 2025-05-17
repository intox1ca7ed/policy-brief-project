import pandas as pd
import numpy as np
from textblob import TextBlob
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=r"C:\Users\82154\Desktop\Digital public spheres\policy-brief-project\petition_analysis\petition_scraper\analysis.log"
)

# Step 1: Load CSV
def load_data(csv_path=r"C:\Users\82154\Desktop\Digital public spheres\policy-brief-project\petition_analysis\petition_scraper\petitions_with_response_json.csv",
              framing_path=r"C:\Users\82154\Desktop\Digital public spheres\policy-brief-project\petition_analysis\petition_scraper\framing_analysis.csv"):
    try:
        df = pd.read_csv(csv_path)
        required_columns = ['petition_id', 'title', 'text', 'response_text']
        if not all(col in df.columns for col in required_columns):
            raise ValueError("CSV missing required columns")
        # Convert date to readable format
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        
        # Try to load framing analysis for similarity scores
        try:
            framing_df = pd.read_csv(framing_path)
            df = df.merge(framing_df[['petition_id', 'similarity_score']], on='petition_id', how='left')
        except FileNotFoundError:
            logging.warning("framing_analysis.csv not found, computing similarity scores")
            df = compute_similarity(df)
        
        return df
    except Exception as e:
        logging.error(f"Error loading CSV: {e}")
        return None

# Step 2: Compute similarity scores (if framing_analysis.csv is missing)
def compute_similarity(df):
    try:
        df['text'] = df['text'].fillna('').astype(str)
        df['response_text'] = df['response_text'].fillna('').astype(str)
        documents = df['text'].tolist() + df['response_text'].tolist()
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        tfidf_matrix = vectorizer.fit_transform(documents)
        n = len(df)
        petition_tfidf = tfidf_matrix[:n]
        response_tfidf = tfidf_matrix[n:]
        similarities = [round(cosine_similarity(petition_tfidf[i], response_tfidf[i])[0][0], 4) for i in range(n)]
        df['similarity_score'] = similarities
        return df
    except Exception as e:
        logging.error(f"Error computing similarity: {e}")
        return df

# Step 3: Perform sentiment analysis
def analyze_sentiment(df):
    try:
        def get_sentiment(text):
            blob = TextBlob(text)
            return blob.sentiment.polarity, blob.sentiment.subjectivity
        
        # Apply sentiment analysis
        sentiments = df['response_text'].apply(get_sentiment)
        df['sentiment_polarity'] = [s[0] for s in sentiments]
        df['sentiment_subjectivity'] = [s[1] for s in sentiments]
        
        # Classify tone
        def classify_tone(polarity, subjectivity):
            if polarity > 0.2 and subjectivity > 0.3:
                return 'Supportive'
            elif polarity < 0 or (polarity <= 0.2 and subjectivity < 0.3):
                return 'Dismissive'
            else:
                return 'Neutral'
        
        df['tone'] = df.apply(lambda x: classify_tone(x['sentiment_polarity'], x['sentiment_subjectivity']), axis=1)
        return df
    except Exception as e:
        logging.error(f"Error in sentiment analysis: {e}")
        return None

# Step 4: Generate bar plot
def plot_sentiment_distribution(df, output_path=r"C:\Users\82154\Desktop\Digital public spheres\policy-brief-project\petition_analysis\petition_scraper\sentiment_distribution.png"):
    try:
        tone_counts = df['tone'].value_counts()
        plt.figure(figsize=(8, 6))
        tone_counts.plot(kind='bar', edgecolor='black', alpha=0.7)
        plt.title('Distribution of Sentiment Tones in Government Responses (30 Petitions)')
        plt.xlabel('Tone')
        plt.ylabel('Number of Petitions')
        plt.grid(True, axis='y', alpha=0.3)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"Sentiment distribution plot saved to {output_path}")
    except Exception as e:
        logging.error(f"Error generating bar plot: {e}")

# Step 5: Save results
def save_results(df, output_path=r"C:\Users\82154\Desktop\Digital public spheres\policy-brief-project\petition_analysis\petition_scraper\sentiment_analysis.csv"):
    try:
        df[['petition_id', 'title', 'similarity_score', 'sentiment_polarity', 'sentiment_subjectivity', 'tone']].to_csv(output_path, index=False)
        logging.info(f"Sentiment analysis saved to {output_path}")
    except Exception as e:
        logging.error(f"Error saving results: {e}")

# Step 6: Print summary
def print_summary(df):
    try:
        print("\nSentiment Analysis Summary (30 Petitions):")
        print(f"Mean Polarity: {df['sentiment_polarity'].mean():.3f}")
        print(f"Mean Subjectivity: {df['sentiment_subjectivity'].mean():.3f}")
        print("\nTone Distribution:")
        print(df['tone'].value_counts().to_string())
        print("\nCorrelation between Similarity Score and Polarity:")
        correlation = df['similarity_score'].corr(df['sentiment_polarity'])
        print(f"Pearson Correlation: {correlation:.3f}")
        print("\nTop 3 Supportive Responses:")
        supportive = df[df['tone'] == 'Supportive'][['petition_id', 'title', 'sentiment_polarity', 'similarity_score']].sort_values(by='sentiment_polarity', ascending=False).head(3)
        for _, row in supportive.iterrows():
            print(f"ID: {row['petition_id']}, Title: {row['title']}, Polarity: {row['sentiment_polarity']:.3f}, Similarity: {row['similarity_score']:.3f}")
        print("\nTop 3 Dismissive Responses:")
        dismissive = df[df['tone'] == 'Dismissive'][['petition_id', 'title', 'sentiment_polarity', 'similarity_score']].sort_values(by='sentiment_polarity').head(3)
        for _, row in dismissive.iterrows():
            print(f"ID: {row['petition_id']}, Title: {row['title']}, Polarity: {row['sentiment_polarity']:.3f}, Similarity: {row['similarity_score']:.3f}")
    except Exception as e:
        logging.error(f"Error printing summary: {e}")

# Main execution
def main():
    logging.info("Starting sentiment analysis on May 17, 2025")
    df = load_data()
    if df is not None:
        df = analyze_sentiment(df)
        if df is not None:
            save_results(df)
            plot_sentiment_distribution(df)
            print_summary(df)
            print("\nResults saved to sentiment_analysis.csv")
            print("Bar plot saved to sentiment_distribution.png")
        else:
            print("Failed to perform sentiment analysis")
    else:
        print("Failed to load CSV")

if __name__ == "__main__":
    main()