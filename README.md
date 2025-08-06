
# 📧 Email Spam Classifier

This project is a Machine Learning-powered **Email Spam Classifier** that detects whether a given email is **Spam** or **Not Spam** based on its content. It uses NLP preprocessing techniques and an SVM classifier trained on real-world email data.

🔗 **[Click here to try the live app](https://email-spam-classification-1atq.onrender.com)**

---

## 🚀 Features

- Clean and simple **Streamlit web interface**
- Powerful **Natural Language Processing** pipeline:
  - HTML tag removal
  - URL & email cleaning
  - Lowercasing, tokenization, stopword removal
  - POS tagging and lemmatization
- TF-IDF vectorization
- Trained with **Support Vector Machine (SVM)** classifier
- Confidence-based prediction with probability bars
- Real-time text classification for testing your own email content

---

## 🛠️ Technologies Used

- Python
- pandas, NumPy
- scikit-learn
- NLTK
- BeautifulSoup
- Streamlit
- Matplotlib

---

## 📁 Project Structure

```
📦 Email Spam Classifier
├── app.py                     # Streamlit app
├── emails.csv                 # Dataset used
├── pipeline.pkl               # Trained model pipeline
├── nltk_data/                 # Pre-downloaded NLTK data
├── download_nltk_data.py      # Script to download NLTK resources
├── setup.sh                   # Streamlit & NLTK setup for Render
├── requirements.txt           # Required Python packages
└── EmailClassifier.ipynb      # Jupyter notebook (exploratory data analysis + model building)
```




## 📊 Dataset

- Source: `emails.csv`
- Contains labeled email messages with `spam` (1 = spam, 0 = not spam)

---

## 👨‍💻 Author

Made by Sunil Limbu



