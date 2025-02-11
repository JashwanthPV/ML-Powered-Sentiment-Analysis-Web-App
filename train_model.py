import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# Sample training data
data = [
    ("I love this product!", 1),
    ("This is the best experience I've had!", 1),
    ("Amazing service and fantastic support!", 1),
    ("This is terrible and disappointing.", 0),
    ("I hate this place!", 0),
    ("Worst experience ever!", 0),
]

# Separate features and labels
texts, labels = zip(*data)

# Build a pipeline with a vectorizer and model
model_pipeline = make_pipeline(
    CountVectorizer(),
    MultinomialNB()
)

# Train the model
model_pipeline.fit(texts, labels)

# Save the model pipeline
with open("model/sentiment_model.pkl", "wb") as f:
    pickle.dump(model_pipeline, f)

print("Model training complete and saved to model/sentiment_model.pkl")
