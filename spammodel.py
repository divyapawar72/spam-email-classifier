import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv("email.csv")

print("\nDataset loaded successfully!")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumns:")
print(df.columns)


# ============================================================
# 2. CLEAN DATA
# ============================================================

# Keep only the columns we need
df = df[["Category", "Message"]]

# Remove missing values
df = df.dropna()

# Remove duplicate messages
df = df.drop_duplicates()

print("\nDataset after cleaning:")
print(df.shape)

print("\nClass distribution:")
print(df["Category"].value_counts())


# ============================================================
# 3. CONVERT LABELS
# ============================================================

# ham = 0
# spam = 1

# ============================================================
# 3. CONVERT LABELS
# ============================================================

# Clean the Category column first
df["Category"] = (
    df["Category"]
    .astype(str)
    .str.strip()
    .str.lower()
)

# Convert:
# ham  -> 0
# spam -> 1
df["label"] = df["Category"].map({
    "ham": 0,
    "spam": 1
})

# Check if any labels could not be converted
print("\nMissing labels after conversion:")
print(df["label"].isna().sum())

# Remove rows where the label could not be converted
df = df.dropna(subset=["label"])

# Convert labels to integers
df["label"] = df["label"].astype(int)

print("\nLabel distribution:")
print(df["label"].value_counts())


# ============================================================
# 4. INPUT AND OUTPUT
# ============================================================

X = df["Message"]
y = df["label"]


# ============================================================
# 5. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 6. TF-IDF
# ============================================================

tfidf = TfidfVectorizer(
    lowercase=True,
    strip_accents="unicode",
    sublinear_tf=True,
    ngram_range=(1, 2),
    min_df=1,
    max_df=0.98,
    max_features=50000
)

X_train_tfidf = tfidf.fit_transform(X_train)

X_test_tfidf = tfidf.transform(X_test)

print("\nTF-IDF complete!")

print("Training matrix shape:")
print(X_train_tfidf.shape)

print("Testing matrix shape:")
print(X_test_tfidf.shape)


# ============================================================
# 7. LOGISTIC REGRESSION
# ============================================================

model = LogisticRegression(
    C=5,
    max_iter=2000,
    class_weight="balanced",
    random_state=42
)

print("\nTraining Logistic Regression...")

model.fit(
    X_train_tfidf,
    y_train
)

print("Training completed!")


# ============================================================
# 8. PREDICTION
# ============================================================

y_pred = model.predict(X_test_tfidf)


# ============================================================
# 9. ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n================================")
print("MODEL PERFORMANCE")
print("================================")

print(
    f"Accuracy: {accuracy * 100:.2f}%"
)


# ============================================================
# 10. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Ham", "Spam"]
    )
)


# ============================================================
# 11. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")

print(cm)


# ============================================================
# 12. CONFUSION MATRIX VISUALIZATION
# ============================================================

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=["Ham", "Spam"],
    yticklabels=["Ham", "Spam"]
)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Spam Detection - Confusion Matrix")

plt.tight_layout()

plt.savefig(
    "confusion_matrix.png",
    dpi=300
)

plt.show()


# ============================================================
# 13. TEST CUSTOM MESSAGES
# ============================================================

custom_messages = [
    "Congratulations! You have won a free prize. Click now!",
    "Hey, are we meeting for lunch today?",
    "URGENT! You have won £1000. Call now!",
    "Can you send me the notes from today's class?"
]

custom_tfidf = tfidf.transform(custom_messages)

custom_predictions = model.predict(custom_tfidf)

print("\n================================")
print("CUSTOM MESSAGE PREDICTIONS")
print("================================")

for message, prediction in zip(
    custom_messages,
    custom_predictions
):

    if prediction == 1:
        result = "SPAM"
    else:
        result = "HAM"

    print("\nMessage:")
    print(message)

    print("Prediction:", result)