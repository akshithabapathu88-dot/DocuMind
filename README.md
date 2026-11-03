# DocuMind AI

An AI-powered document understanding application built using
Python, Streamlit, Groq LLM and Retrieval-Augmented Generation.

## Features

- PDF Chat
- AI Document Summarization
- AI Quiz Generator
- PDF Report Generator

## Run locally

1. Install Python 3.10 or later.
2. Install packages:

   pip install -r requirements.txt

3. Set your Groq API key.

   Windows PowerShell:
   $env:GROQ_API_KEY="your_api_key"

4. Run:

   streamlit run app.py

## Deployment

Upload app.py and requirements.txt to a GitHub repository.
Deploy the repository using Streamlit Community Cloud.

Add this secret in Streamlit Cloud:

GROQ_API_KEY = "your_actual_groq_api_key"

Never upload your API key to GitHub.
