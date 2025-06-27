# ai_powered_course_recommendation
An intelligent, NLP-powered web app built using Streamlit that recommends personalized course modules based on users' natural language queries like:

"I want a beginner course on supervised learning."

## 🚀 Features
Natural Language Understanding using spaCy for lemmatization and stopword removal.

TF-IDF + Logistic Regression model to predict the course level from user queries.

Custom Keyword Rules to detect interests like "supervised", "deep learning", or intent like "beginner".

Clean Streamlit Interface for real-time interaction.

## 🧠 How it Works
Preprocessing: Lemmatizes and cleans text using spaCy.

Vectorization: Uses TfidfVectorizer to convert course module descriptions into feature vectors.

Classification: Trains a Logistic Regression model to predict the course level (Beginner, Intermediate, Advanced).

Keyword Filtering: Applies filters like 'supervised', 'deep learning', 'beginner' based on the query.

Rule-Based + ML Pipeline: Combines both keyword matching and fallback ML prediction for best results.

## 🛠️ Tech Stack
Frontend: Streamlit

NLP: spaCy (en_core_web_sm)

ML: Scikit-learn (TF-IDF, Logistic Regression)

Data: Custom courses.csv file with module names, levels, and keyword tags.

