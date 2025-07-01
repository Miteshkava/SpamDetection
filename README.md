# SpamDetection
# 📧 Spam Detection using Machine Learning

This project focuses on building a **Spam Detection** system using Natural Language Processing (NLP) and machine learning techniques. The model is trained to classify text messages (SMS or emails) as either **Spam** or **Ham (Not Spam)**.

---

## 📁 Project Structure

spam-detection/
├── data/
│ └── spam.csv
├── spam_detection.py
├── requirements.txt
└── README.md

yaml
Copy
Edit

---

## 📌 Dataset Description

The dataset contains a collection of labeled SMS messages. Each entry has:

- `label`: Indicates whether the message is "spam" or "ham" (not spam)
- `message`: The actual text of the message

Example:
| label | message                       |
|-------|-------------------------------|
| ham   | I'm going to the store now.   |
| spam  | WINNER! Claim your prize now! |

---

## 🎯 Project Objectives

- Load and preprocess text data
- Perform text cleaning: lowercase, punctuation removal, stopword filtering
- Convert text to numerical features using TF-IDF or CountVectorizer
- Train classification models to detect spam
- Evaluate performance using accuracy, precision, recall, and F1-score

---

## 🛠️ Tools and Libraries Used

- Python
- Pandas
- NumPy
- NLTK / re (Regular Expressions)
- Scikit-learn (for model building and evaluation)
- Matplotlib / Seaborn (for visualizations)

---

## ▶️ How to Run the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/spam-detection.git
   cd spam-detection
Install required packages:

bash
Copy
Edit
pip install -r requirements.txt
Run the script:

bash
Copy
Edit
python spam_detection.py
Or open the notebook version:

bash
Copy
Edit
jupyter notebook
🤖 Models Trained
Multinomial Naive Bayes

Logistic Regression

Support Vector Machine (SVM)

Random Forest (optional)

📊 Evaluation Metrics
Confusion Matrix

Accuracy Score

Precision, Recall, F1-Score

ROC-AUC Score

🔍 Sample Insight
Multinomial Naive Bayes performs exceptionally well on text classification tasks like spam detection due to the probabilistic nature of word frequencies.
