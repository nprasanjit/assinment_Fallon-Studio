import pandas as pd
import numpy as np
from transformers import pipeline, DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import re

# Load and preprocess the dataset
def preprocess_data(df):
    # Clean text
    def clean_text(text):
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    df = df.dropna(subset=['review_content', 'rating']).copy()
    df['review_content'] = df['review_content'].apply(clean_text)
    
    # Create sentiment labels from ratings
    def get_sentiment(rating):
        if rating >= 4.0:
            return 'positive'
        elif rating == 3.0:
            return 'neutral'
        else:
            return 'negative'
    
    df['true_sentiment'] = df['rating'].apply(get_sentiment)
    return df

# Load dataset and take a sample
df = pd.read_csv('amazon_d.csv')
df = preprocess_data(df)
df_sample = df.sample(n=500, random_state=42)  # Sample 500 reviews for evaluation

# Initialize the sentiment analysis model
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(model_name)
sentiment_analyzer = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Function to classify sentiment with neutral detection
def classify_sentiment(review):
    if not review:
        return "neutral", 0.5
    result = sentiment_analyzer(review)[0]
    score = result['score']
    label = result['label'].lower()
    
    # Threshold for neutral: confidence score between 0.4 and 0.6
    if 0.4 <= score <= 0.6:
        sentiment = "neutral"
    else:
        sentiment = label
    
    return sentiment, score

# Function to generate explanations based on prompt type
def generate_explanation(review, sentiment, score, product_name, category, prompt_type):
    if prompt_type == "direct":
        return f"Sentiment: {sentiment.capitalize()}\nExplanation: The review expresses a {sentiment} opinion based on its tone and content."
    elif prompt_type == "score":
        key_phrases = " and ".join(review.split()[:3])  # Simplified key phrase extraction
        return f"Sentiment Score: {score:.2f}\nKey Phrases: {key_phrases}"
    elif prompt_type == "contextual":
        return f"Sentiment: {sentiment.capitalize()}\nExplanation: The review reflects a {sentiment} experience with the {product_name} in the {category} category, focusing on its features."

# Evaluate model on the dataset
results = []
prompt_types = ["direct", "score", "contextual"]

for idx, row in df_sample.iterrows():
    review = row['review_content']
    product_name = row['product_name']
    category = row['category']
    true_sentiment = row['true_sentiment']
    
    # Get sentiment prediction
    predicted_sentiment, score = classify_sentiment(review)
    
    # Generate explanations for all prompt types
    for prompt_type in prompt_types:
        explanation = generate_explanation(review, predicted_sentiment, score, product_name, category, prompt_type)
        results.append({
            'Review': review,
            'Product': product_name,
            'Category': category,
            'True Sentiment': true_sentiment,
            'Predicted Sentiment': predicted_sentiment,
            'Score': score,
            'Prompt Type': prompt_type,
            'Explanation': explanation
        })

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Compute metrics (using the direct prompt predictions for simplicity)
true_labels = results_df[results_df['Prompt Type'] == 'direct']['True Sentiment']
predicted_labels = results_df[results_df['Prompt Type'] == 'direct']['Predicted Sentiment']
accuracy = accuracy_score(true_labels, predicted_labels)
precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='weighted', zero_division=0)

# Print metrics
print("Evaluation Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# Save results to CSV
results_df.to_csv('sentiment_evaluation_results.csv', index=False)

# Qualitative analysis: Print a few sample outputs
print("\nSample Outputs for Qualitative Analysis:")
for prompt_type in prompt_types:
    print(f"\nPrompt Type: {prompt_type.capitalize()}")
    sample = results_df[results_df['Prompt Type'] == prompt_type].head(2)
    for _, row in sample.iterrows():
        print(f"Review: {row['Review'][:100]}...")
        print(f"True Sentiment: {row['True Sentiment']}")
        print(f"Predicted Sentiment: {row['Predicted Sentiment']}")
        print(f"{row['Explanation']}\n")