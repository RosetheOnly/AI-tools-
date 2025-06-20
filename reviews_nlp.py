
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