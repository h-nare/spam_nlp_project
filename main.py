import pandas as pd
from preprocess import clean_text

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, classification_report

# Load dataset
print("Loading dataset...")
data = pd.read_csv("data/spam.csv", encoding="latin-1")
print("Dataset loaded")

data = data[["v1", "v2"]]
data.columns = ["label", "text"]

# NLP preprocessing
data["clean_text"] = data["text"].apply(clean_text)

# Encode labels
data["label"] = data["label"].map({"ham": 0, "spam": 1})

# Feature extraction (TF-IDF)
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data["clean_text"])
y = data["label"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model: Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)

cv_score = cross_val_score(model, X, y, cv=5).mean()

print("\n================ RESULTS ================")

print("Accuracy:", accuracy)
print("Cross-validation score:", cv_score)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
