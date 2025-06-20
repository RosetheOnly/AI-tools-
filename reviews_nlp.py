
import spacy
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

# Load spaCy model
nlp = spacy.load("en_core_web_sm")
# Sample Amazon reviews (in practice, you'd load from a dataset)
sample_reviews = [
     "I love this iPhone 14 Pro! The camera quality is amazing and Apple really outdid themselves.",
     "Samsung Galaxy S23 is okay but the battery life could be better. Samsung needs to improve.",
     "The Nike Air Max shoes are comfortable but expensive. Nike products are always pricey.",
     "This MacBook Pro M2 is incredibly fast. Apple's M2 chip is revolutionary for professionals.",
     "Amazon Echo Dot works well with Alexa. Amazon has great smart home products.",
     "The Sony WH-1000XM4 headphones have excellent noise cancellation. Sony audio quality is top-notch.",
     "Tesla Model 3 is an amazing electric vehicle. Tesla's innovation in EVs is unmatched.",
     "Microsoft Surface Pro 9 is versatile but heavy. Microsoft could make it lighter.",
     "Google Pixel 7 has a great camera but poor battery. Google phones are hit or miss.",
     "Dell XPS 13 laptop is reliable and lightweight. Dell makes quality business laptops."
 ]
# Create a DataFrame 
df = pd.DataFrame({'review': sample_reviews})

def extract_entities_and_sentiment(text):
    """Extract named entities and analyze sentiment"""
    doc = nlp(text)
   
    # Extract entities
    entities = []
    for ent in doc.ents:
        entities.append({
            'text': ent.text,
            'label': ent.label_,
            'description': spacy.explain(ent.label_)
        })
   
    # Rule-based sentiment analysis
    positive_words = ['love', 'amazing', 'excellent', 'great', 'good', 'fantastic', 'wonderful',
                     'awesome', 'outstanding', 'perfect', 'incredible', 'top-notch', 'revolutionary']
    negative_words = ['hate', 'terrible', 'awful', 'bad', 'poor', 'horrible', 'disappointing',
                     'expensive', 'pricey', 'heavy', 'slow', 'unreliable']
   
    # Convert to lowercase for matching
    text_lower = text.lower()
   
    positive_count = sum(1 for word in positive_words if word in text_lower)
    negative_count = sum(1 for word in negative_words if word in text_lower)
   
    # Determine sentiment
    if positive_count > negative_count:
        sentiment = 'Positive'
        confidence = positive_count / (positive_count + negative_count) if (positive_count + negative_count) > 0 else 0.5
    elif negative_count > positive_count:
        sentiment = 'Negative'
        confidence = negative_count / (positive_count + negative_count) if (positive_count + negative_count) > 0 else 0.5
    else:
        sentiment = 'Neutral'
        confidence = 0.5
   
    return entities, sentiment, confidence

# Process all reviews
results = []
all_entities = []
sentiments = []

print("Processing Reviews:")
print("=" * 50)

for i, review in enumerate(df['review']):
    entities, sentiment, confidence = extract_entities_and_sentiment(review)
   
    # Store results
    results.append({
        'review_id': i,
        'review': review,
        'entities': entities,
        'sentiment': sentiment,
        'confidence': confidence
    })
   
    # Collect all entities for analysis
    all_entities.extend(entities)
    sentiments.append(sentiment)
   
    # Print results
    print(f"Review {i+1}:")
    print(f"Text: {review}")
    print(f"Sentiment: {sentiment} (Confidence: {confidence:.2f})")
    print("Entities found:")
    for ent in entities:
        print(f"  - {ent['text']} ({ent['label']}: {ent['description']})")
    print("-" * 30)

# Analyze extracted entities
print("\nEntity Analysis:")
print("=" * 30)

# Count entity types
entity_labels = [ent['label'] for ent in all_entities]
entity_counts = Counter(entity_labels)

print("Entity Types Found:")
for label, count in entity_counts.most_common():
    print(f"  {label}: {count} ({spacy.explain(label)})")

# Count specific entities (brands/products)
entity_texts = [ent['text'] for ent in all_entities]
entity_text_counts = Counter(entity_texts)

print(f"\nMost Mentioned Entities:")
for entity, count in entity_text_counts.most_common(10):
    print(f"  {entity}: {count}")

# Sentiment distribution
sentiment_counts = Counter(sentiments)
print(f"\nSentiment Distribution:")
for sentiment, count in sentiment_counts.items():
    print(f"  {sentiment}: {count}")

# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Entity types distribution
axes[0, 0].bar(entity_counts.keys(), entity_counts.values())
axes[0, 0].set_title('Entity Types Distribution')
axes[0, 0].set_xlabel('Entity Type')
axes[0, 0].set_ylabel('Count')
axes[0, 0].tick_params(axis='x', rotation=45)

# Sentiment distribution
axes[0, 1].pie(sentiment_counts.values(), labels=sentiment_counts.keys(), autopct='%1.1f%%')
axes[0, 1].set_title('Sentiment Distribution')

# Top entities
top_entities = dict(entity_text_counts.most_common(8))
axes[1, 0].bar(top_entities.keys(), top_entities.values())
axes[1, 0].set_title('Most Mentioned Entities')
axes[1, 0].set_xlabel('Entity')
axes[1, 0].set_ylabel('Count')
axes[1, 0].tick_params(axis='x', rotation=45)

# Review length vs sentiment
review_lengths = [len(review.split()) for review in df['review']]
colors = ['green' if s == 'Positive' else 'red' if s == 'Negative' else 'gray' for s in sentiments]
axes[1, 1].scatter(review_lengths, range(len(review_lengths)), c=colors, alpha=0.6)
axes[1, 1].set_title('Review Length vs Sentiment')
axes[1, 1].set_xlabel('Word Count')
axes[1, 1].set_ylabel('Review Index')

plt.tight_layout()
plt.show()
