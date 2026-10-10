# Spam Email Classifier

## About the Project

This project uses Machine Learning and Natural Language Processing (NLP) to classify messages as Spam or Ham (legitimate messages).

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* TF-IDF Vectorization
* Matplotlib
* Seaborn

## Features

* Classifies messages as spam or legitimate.
* Converts text into numerical features using TF-IDF.
* Evaluates model performance using a confusion matrix.
* Supports testing new text messages.

## Model Evaluation

The confusion matrix from my current test run shows:

| Actual Label | Predicted Ham | Predicted Spam |
| ------------ | ------------: | -------------: |
| Ham          |           891 |             13 |
| Spam         |            16 |            112 |

## Project Files

* `spammodel.py` — Spam detection model.
* `spam_detection.py` — Spam detection script.
* `spam_detection.ipynb` — Jupyter Notebook.
* `confusion_matrix.png` — Confusion matrix visualization.
* `email.csv` — Dataset file.

## How to Run

1. Install Python.
2. Install the required libraries.
3. Keep the dataset in the project folder.
4. Run the Python script.

## Future Improvements

* Connect the model to an email application.
* Improve spam detection performance.
* Test the model on more real-world emails.

## Author

Divya Pawar
