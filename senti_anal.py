import nltk
from nltk.corpus import movie_reviews
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report

# Download movie reviews dataset
nltk.download('movie_reviews')

# Load data and labels
docs = [' '.join(movie_reviews.words(fileid)) for fileid in movie_reviews.fileids()]
labels = [1 if fileid.startswith('pos') else 0 for fileid in movie_reviews.fileids()]

# Split data
X_train, X_test, y_train, y_test = train_test_split(docs, labels, test_size=0.2, random_state=0)

# Create model pipeline
model = make_pipeline(CountVectorizer(), MultinomialNB())

# Train model
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Predict sentiment on custom input
while True:
    user_input = input("Enter text to analyze (or type 'exit'): ")
    if user_input.lower() == 'exit':
        break
    prediction = model.predict([user_input])[0]
    print("Sentiment:", "Positive 😊" if prediction == 1 else "Negative 😞")
