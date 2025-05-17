import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=r"C:\Users\82154\Desktop\Digital public spheres\policy-brief-project\petition_analysis\petition_scraper\analysis.log"
)

# Step 1: Load CSV
def load_data(csv_path=r"C:\Users\82154\Desktop\Digital public spheres\policy-brief-project\petition_analysis\petition_scraper\petitions_with_response_json.csv"):
    try:
        df = pd.read_csv(csv_path)
        required_columns = ['petition_id', 'title', 'text', 'response_text']
        if not all(col in df.columns for col in required_columns):
            raise ValueError("CSV missing required columns")
        # Convert date to readable format for reference
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        return df
    except Exception as e:
        logging.error(f"Error loading CSV: {e}")
        return None

# Step 2: Compute TF-IDF and cosine similarity
def compute_similarity(df):
    try:
        # Ensure text and response_text are strings, handle NaN
        df['text'] = df['text'].fillna('').astype(str)
        df['response_text'] = df['response_text'].fillna('').astype(str)
        
        # Combine text and response_text for TF-IDF
        documents = df['text'].tolist() + df['response_text'].tolist()
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        tfidf_matrix = vectorizer.fit_transform(documents)
        
        # Split into petition and response matrices
        n = len(df)
        petition_tfidf = tfidf_matrix[:n]
        response_tfidf = tfidf_matrix[n:]
        
        # Compute cosine similarity
        similarities = []
        for i in range(n):
            sim = cosine_similarity(petition_tfidf[i], response_tfidf[i])[0][0]
            similarities.append(round(sim, 4))  # Round for readability
        
        df['similarity_score'] = similarities
        return df
    except Exception as e:
        logging.error(f"Error computing similarity: {e}")
        return None

# Step 3: Generate histogram
def plot_histogram(df, output_path=r"C:\Users\82154\Desktop\Digital public spheres\policy-brief-project\petition_analysis\petition_scraper\similarity_histogram.png"):
    try:
        scores = df['similarity_score']
        plt.figure(figsize=(8, 6))
        plt.hist(scores, bins=10, edgecolor='black', alpha=0.7)
        plt.title('Distribution of Similarity Scores (Petition vs. Government Response)')
        plt.xlabel('Cosine Similarity Score')
        plt.ylabel('Number of Petitions')
        plt.grid(True, alpha=0.3)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"Histogram saved to {output_path}")
    except Exception as e:
        logging.error(f"Error generating histogram: {e}")

# Step 4: Save results
def save_results(df, output_path=r"C:\Users\82154\Desktop\Digital public spheres\policy-brief-project\petition_analysis\petition_scraper\framing_analysis.csv"):
    try:
        df[['petition_id', 'title', 'similarity_score']].to_csv(output_path, index=False)
        logging.info(f"Framing analysis saved to {output_path}")
    except Exception as e:
        logging.error(f"Error saving results: {e}")

# Step 5: Print summary
def print_summary(df):
    try:
        scores = df['similarity_score']
        print("\nFraming Consistency Analysis Summary (30 Petitions):")
        print(f"Median Similarity Score: {scores.median():.3f}")
        print(f"Mean Similarity Score: {scores.mean():.3f}")
        print(f"Min Similarity Score: {scores.min():.3f}")
        print(f"Max Similarity Score: {scores.max():.3f}")
        print(f"Standard Deviation: {scores.std():.3f}")
        print("\nTop 3 Petitions by Similarity Score:")
        top_3 = df[['petition_id', 'title', 'similarity_score']].sort_values(by='similarity_score', ascending=False).head(3)
        for _, row in top_3.iterrows():
            print(f"ID: {row['petition_id']}, Title: {row['title']}, Score: {row['similarity_score']:.3f}")
        print("\nBottom 3 Petitions by Similarity Score:")
        bottom_3 = df[['petition_id', 'title', 'signature_count', 'similarity_score']].sort_values(by='similarity_score').head(3)
        for _, row in bottom_3.iterrows():
            print(f"ID: {row['petition_id']}, Title: {row['title']}, Score: {row['similarity_score']:.3f}")
    except Exception as e:
        logging.error(f"Error printing summary: {e}")

# Main execution
def main():
    logging.info("Starting framing consistency analysis with histogram on May 16, 2025")
    df = load_data()
    if df is not None:
        df = compute_similarity(df)
        if df is not None:
            save_results(df)
            plot_histogram(df)
            print_summary(df)
            print("\nResults saved to framing_analysis.csv")
            print("Histogram saved to similarity_histogram.png")
        else:
            print("Failed to compute similarity scores")
    else:
        print("Failed to load CSV")

if __name__ == "__main__":
    main()