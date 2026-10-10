from pathlib import Path
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "email.csv"
MODEL_PATH = BASE_DIR / "spam_model.joblib"
MATRIX_PATH = BASE_DIR / "confusion_matrix.png"


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Place email.csv beside spammodel.py."
        )

    df = pd.read_csv(DATA_PATH)

    required_columns = {"Category", "Message"}
    if not required_columns.issubset(df.columns):
        raise ValueError(
            "email.csv must contain Category and Message columns."
        )

    df = df[["Category", "Message"]].dropna().copy()
    df["Category"] = df["Category"].astype(str).str.strip().str.lower()
    df["Message"] = df["Message"].astype(str).str.strip()
    df = df[df["Category"].isin(["ham", "spam"])]
    df = df[df["Message"] != ""]
    df = df.drop_duplicates()

    df["label"] = df["Category"].map({"ham": 0, "spam": 1})

    if df["label"].nunique() != 2:
        raise ValueError("The dataset must contain both ham and spam messages.")

    X = df["Message"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=42
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            sublinear_tf=True,
            ngram_range=(1, 2),
            max_df=0.98,
            max_features=50000
        )),
        ("classifier", LogisticRegression(
            C=5,
            max_iter=2000,
            class_weight="balanced",
            random_state=42
        ))
    ])

    print("Training spam detection model...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"\nDataset rows after cleaning: {len(df)}")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"\nTest accuracy: {accuracy * 100:.2f}%")

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Ham", "Spam"],
        zero_division=0
    ))

    cm = confusion_matrix(y_test, predictions, labels=[0, 1])

    print("\nConfusion Matrix:")
    print(cm)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=["Ham", "Spam"],
        yticklabels=["Ham", "Spam"],
        cmap="rocket_r"
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.title("Spam Detection - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(MATRIX_PATH, dpi=300)
    plt.close()

    joblib.dump(model, MODEL_PATH)

    print(f"\nSaved trained model: {MODEL_PATH.name}")
    print(f"Saved confusion matrix: {MATRIX_PATH.name}")

    examples = [
        "Congratulations! You have won a free prize. Click now!",
        "Hey, are we meeting for lunch today?",
        "URGENT! You have won £1000. Call now!",
        "Can you send me the notes from today's class?"
    ]

    print("\nSample predictions:")
    for message, prediction in zip(examples, model.predict(examples)):
        label = "SPAM" if prediction == 1 else "HAM"
        print(f"{label}: {message}")


if __name__ == "__main__":
    main()
