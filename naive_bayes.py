import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Read and preprocess data
def read_data(file_path):
    # Load data into a Pandas DataFrame
    df = pd.read_csv(file_path)
    
    # Select relevant columns
    negative_reviews = df[df['Negative_Review'] != 'No Negative'][['Negative_Review']]
    negative_reviews['sentiment'] = 0  # Label negative reviews

    positive_reviews = df[df['Positive_Review'] != 'No Positive'][['Positive_Review']]
    positive_reviews['sentiment'] = 1  # Label positive reviews

    # Combine positive and negative reviews
    reviews = pd.concat([negative_reviews.rename(columns={'Negative_Review': 'text'}),
                         positive_reviews.rename(columns={'Positive_Review': 'text'})])
    
    # Shuffle data and reset index
    reviews = reviews.sample(frac=1, random_state=42).reset_index(drop=True)
    return reviews['text'], reviews['sentiment']

# Tokenize text
def tokenize(text):
    return text.lower().split()

# Train Naive Bayes classifier
def train_naive_bayes(X_train, y_train):
    word_counts = {0: {}, 1: {}}
    class_counts = {0: 0, 1: 0}
    vocabulary = set()

    for text, label in zip(X_train, y_train):
        tokens = tokenize(text)
        class_counts[label] += len(tokens)
        for token in tokens:
            vocabulary.add(token)
            if token not in word_counts[label]:
                word_counts[label][token] = 0
            word_counts[label][token] += 1

    return word_counts, class_counts, vocabulary

# Predict using Naive Bayes
def predict_naive_bayes(text, word_counts, class_counts, vocabulary):
    tokens = tokenize(text)
    total_words = sum(class_counts.values())
    log_probabilities = {0: 0, 1: 0}

    for label in [0, 1]:
        log_probabilities[label] = class_counts[label] / total_words  # Prior probability
        for token in tokens:
            word_probability = (
                word_counts[label].get(token, 0) + 1
            ) / (class_counts[label] + len(vocabulary))  # Laplace smoothing
            log_probabilities[label] *= word_probability

    return max(log_probabilities, key=log_probabilities.get)

# Evaluate the model
def evaluate_model(X_test, y_test, word_counts, class_counts, vocabulary):
    correct = 0
    for text, label in zip(X_test, y_test):
        prediction = predict_naive_bayes(text, word_counts, class_counts, vocabulary)
        if prediction == label:
            correct += 1
    return correct / len(y_test)

# Main script
file_path = os.path.join(os.path.dirname(__file__), 'datasets', 'Hotel_Reviews.csv')
X, y = read_data(file_path)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
word_counts, class_counts, vocabulary = train_naive_bayes(X_train, y_train)

# Evaluate on test set
accuracy = evaluate_model(X_test, y_test, word_counts, class_counts, vocabulary)

print(f'Accuracy: {accuracy:.2f}')
