import streamlit as st
import joblib 
import re
import os
import nltk

# Set explicit paths for NLTK data
nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
os.environ['NLTK_DATA'] = nltk_data_path 
nltk.data.path.append(nltk_data_path)   

# Debugging: Print current paths (remove in production)
print("Current working directory:", os.getcwd())
print("NLTK search paths:", nltk.data.path)

# Download required resources with explicit paths
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading punkt...")
    nltk.download('punkt', download_dir=nltk_data_path)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("Downloading stopwords...")
    nltk.download('stopwords', download_dir=nltk_data_path)

# Special handling for punkt_tab
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    print("punkt_tab not found - using punkt instead")
    # Workaround: Copy punkt to punkt_tab if needed
    punkt_path = os.path.join(nltk_data_path, 'tokenizers', 'punkt')
    punkt_tab_path = os.path.join(nltk_data_path, 'tokenizers', 'punkt_tab')
    
    if os.path.exists(punkt_path) and not os.path.exists(punkt_tab_path):
        print("Creating symlink from punkt to punkt_tab")
        os.symlink(punkt_path, punkt_tab_path)

# Now import other NLTK components
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt

# Load the trained pipeline
pipeline = joblib.load('pipeline.pkl')

# Streamlit app config
st.set_page_config(page_title="Spam Email Classifier", page_icon="📧", layout="centered")

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #252421;
        padding: 2rem;
        border-radius: 10px;
    }
    section[data-testid="stSidebar"] {
        background-color: #dbeaf5;
    }
    .stTextArea textarea {
        height: 200px !important;
    }
    .title {
        color: #333333;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# App header
st.markdown("<h1 class='title'>📧 Email Spam Classifier</h1>", unsafe_allow_html=True)
st.write("🔍 Paste your email below and let the model detect if it's **Spam** or **Not Spam**.")

# Text input
email = st.text_area('✉️ Enter your email content here:')

# Preprocessing function
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  

def pre_process(text):
    text = text.lower()
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'_+', ' ', text)
    text = re.sub(r'\b\d+\b', ' <NUM> ', text)
    text = re.sub(r'[^\w\s]', '', text)

    tokens = word_tokenize(text)

    custom_stopwords = {'_', 'subject', 'com', 'http', 'mail', 'e', 'u', '000', 'www','NUM'}
    stop_words = set(stopwords.words('english')).union(custom_stopwords)
    tokens = [word for word in tokens if word not in stop_words and len(word) > 1]

    tagged_tokens = pos_tag(tokens)
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word, get_wordnet_pos(pos)) for word, pos in tagged_tokens]

    return " ".join(tokens)

# button
if st.button("Predict"):
    if email.strip() == "":
        st.warning("Please enter an email to classify.")
    else:
        processed_email = pre_process(email)
        prediction = pipeline.predict([processed_email])[0]
        proba = pipeline.predict_proba([processed_email])[0]

        spam_score = round(proba[1] * 100, 2)
        ham_score = round(proba[0] * 100, 2)

        if prediction == 1:
            st.error(f"🔴 Prediction: This is **SPAM** ({spam_score}%)")
        else:
            st.success(f"🟢 Prediction: This is **NOT SPAM** ({ham_score}%)")

        # Horizontal bar chart
        st.markdown("### 📊 Prediction Confidence")
        fig, ax = plt.subplots(figsize=(6, 2))
        categories = ['Not Spam', 'Spam']
        scores = [ham_score, spam_score]
        ax.barh(categories, scores, color=['green', 'red'])
        ax.set_xlim(0, 100)
        ax.set_xlabel('Probability (%)')
        for i, v in enumerate(scores):
            ax.text(v + 1, i, f"{v}%", color='black', va='center')
        st.pyplot(fig)

# Sidebar info
st.sidebar.title("📘 About This App")
st.sidebar.markdown("""
This **Spam Email Classifier** uses a machine learning model to predict whether an email is **Spam** or **Not Spam** based on its content.

###  Steps Followed:

1. **Data Cleaning & Preprocessing**  
   - Removed HTML tags, URLs, emails, digits, punctuation  
   - Lowercased the text  
   - Removed stopwords and irrelevant tokens  
   - Tokenized, POS-tagged and **lemmatized** each word for better generalization

2. **Feature Extraction**  
   - Converted cleaned text into numerical vectors using **TF-IDF**

3. **Model Training**  
   - Trained an **SVM classifier** on labeled spam data

4. **Model Packaging**  
   - Combined preprocessing and classification in a `Pipeline`  
   - Saved with `joblib` and integrated with this Streamlit app

---

###  Tools Used:
- Python 
- Numpy
- Pandas                 
- scikit-learn  
- NLTK  
- BeautifulSoup  
- Streamlit

Feel free to test with your own email content! 💬
""")
