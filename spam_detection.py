import pandas as pd
import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv("email.csv")

# Test with only 20 messages


print("\nDataset loaded successfully!")
print(df.head(20))
df = pd.read_csv("email.csv")

# Test with only 20 messages
df = df.head(20)

print("\nDataset loaded successfully!")
print(df.head(20))

print("\nColumns in dataset:")
print(df.columns)

print("\nColumns in dataset:")
print(df.columns)


# ============================================================
# 2. CLASSIFICATION FUNCTION
# ============================================================

def classify_texts(
    df,
    text_col,
    model_path="cja5553/deberta-Twitter-spam-classification",
    batch_size=8
):

    """
    Classifies texts as Quality or Spam
    using a pre-trained Hugging Face model.
    """

    # --------------------------------------------------------
    # Select CPU or GPU automatically
    # --------------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("\nUsing device:", device)


    # --------------------------------------------------------
    # Load tokenizer
    # --------------------------------------------------------

    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path
    )


    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("Loading model...")

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path
    )

    model = model.to(device)

    model.eval()


    # --------------------------------------------------------
    # Check text column
    # --------------------------------------------------------

    if text_col not in df.columns:

        raise ValueError(
            f"Column '{text_col}' not found. "
            f"Available columns are: {list(df.columns)}"
        )


    # --------------------------------------------------------
    # Prepare text
    # --------------------------------------------------------

    df = df.copy()

    df["text"] = df[text_col].astype(str)


    # --------------------------------------------------------
    # Convert Pandas DataFrame to Hugging Face Dataset
    # --------------------------------------------------------

    text_dataset = Dataset.from_pandas(df)


    # --------------------------------------------------------
    # Tokenization
    # --------------------------------------------------------

    def tokenize_function(example):

        return tokenizer(
            example["text"],
            padding="max_length",
            truncation=True,
            max_length=512
        )


    print("\nTokenizing text...")

    text_dataset = text_dataset.map(
        tokenize_function,
        batched=True
    )


    # --------------------------------------------------------
    # Convert to PyTorch format
    # --------------------------------------------------------

    text_dataset.set_format(
        type="torch",
        columns=[
            "input_ids",
            "attention_mask"
        ]
    )


    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    text_loader = DataLoader(
        text_dataset,
        batch_size=batch_size
    )


    # --------------------------------------------------------
    # Make predictions
    # --------------------------------------------------------

    predictions = []

    print("\nMaking predictions...")

    with torch.no_grad():

        for batch in tqdm(text_loader):

            input_ids = batch["input_ids"].to(device)

            attention_mask = batch["attention_mask"].to(device)


            # Model prediction

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )


            # Get logits

            logits = outputs.logits


            # Get predicted class

            preds = torch.argmax(
                logits,
                dim=-1
            ).cpu().numpy()


            predictions.extend(preds)


    # --------------------------------------------------------
    # Convert prediction numbers to labels
    # --------------------------------------------------------

    id2label = {
        0: "Quality",
        1: "Spam"
    }


    predicted_labels = [
        id2label[int(pred)]
        for pred in predictions
    ]


    # --------------------------------------------------------
    # Add predictions to DataFrame
    # --------------------------------------------------------

    df["spam_prediction"] = predicted_labels


    return df


# ============================================================
# 3. RUN CLASSIFICATION
# ============================================================

spam_df_classification = classify_texts(
    df,
    "Message"
)


# ============================================================
# 4. DISPLAY RESULTS
# ============================================================

print("\n========================================")
print("SPAM CLASSIFICATION RESULTS")
print("========================================\n")

print(
    spam_df_classification[
        ["Message", "spam_prediction"]
    ].to_string(index=False)
)
# ==========================================
# EVALUATE THE MODEL
# ==========================================

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# Convert original labels to the same format as model predictions
spam_df_classification["actual_label"] = (
    spam_df_classification["Category"]
    .map({
        "ham": "Quality",
        "spam": "Spam"
    })
)


# ------------------------------------------
# 1. Accuracy
# ------------------------------------------

accuracy = accuracy_score(
    spam_df_classification["actual_label"],
    spam_df_classification["spam_prediction"]
)

print("\n==============================")
print("MODEL ACCURACY")
print("==============================")

print("Accuracy:", accuracy)


# ------------------------------------------
# 2. Classification Report
# ------------------------------------------

print("\n==============================")
print("CLASSIFICATION REPORT")
print("==============================")

print(
    classification_report(
        spam_df_classification["actual_label"],
        spam_df_classification["spam_prediction"]
    )
)


# ------------------------------------------
# 3. Confusion Matrix
# ------------------------------------------

cm = confusion_matrix(
    spam_df_classification["actual_label"],
    spam_df_classification["spam_prediction"],
    labels=["Quality", "Spam"]
)

print("\n==============================")
print("CONFUSION MATRIX")
print("==============================")

print(cm)


# ------------------------------------------
# 4. Display Confusion Matrix
# ------------------------------------------

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=["Quality", "Spam"],
    yticklabels=["Quality", "Spam"]
)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("DeBERTa Spam Classification")

plt.show()