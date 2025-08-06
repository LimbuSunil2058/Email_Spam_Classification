#!/bin/bash

# Step 1: Create Streamlit config
mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"your-email@example.com\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[server]\n\
headless = true\n\
enableCORS = false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml

# Step 2: Install Python dependencies
pip install -r requirements.txt

# Step 3: Download NLTK resources to local folder
python download_nltk_data.py

