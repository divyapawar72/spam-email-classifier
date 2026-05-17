# Simple Spam Classifier

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Step 1: Load dataset (make sure spam.csv is in same folder)
df = pd.read_csv("spam.csv", encoding='latin-1')

# Step 2: Keep only required columns
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# Step 3: Convert labels to numbers
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Step 4: Split data
X = df['message']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Step 5: Convert text into numbers
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Step 6: Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 7: Test with your own message
print("\n--- Test Your Message ---")
user_input = input("Enter a message: ")

user_vec = vectorizer.transform([user_input])
prediction = model.predict(user_vec)

if prediction[0] == 1:
    print("Result: Spam 🚫")
else:
    print("Result: Not Spam ✅")
