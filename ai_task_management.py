
import streamlit as st
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib


st.set_page_config(page_title="Smart AI Course Recommender", layout="wide")
st.title("🎓 AI-Powered Course Recommendation System")
st.subheader("Enter your learning interest or question below:")

user_input = st.text_input("💬 Ask me something like 'I want a beginner course on supervised learning'")

# Load data
modules_df = pd.read_csv(r'C:\Users\dhima\Downloads\courses.csv')

# Load NLP model
nlp = spacy.load("en_core_web_sm")

# Preprocess function
def preprocess(text):
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return " ".join(tokens)

# Preprocess and vectorize
modules_df['processed_text'] = modules_df['Module'].astype(str).apply(preprocess)
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(modules_df['processed_text'])
y = modules_df['Course_Level']

# Train a classifier
clf = LogisticRegression(max_iter=1000)
clf.fit(X, y)

# Keyword filters
keyword_filters = {
    'supervised': lambda df: df[df['Keywords_Tags_Skills_Interests_Categories']
                                .str.contains('supervised', case=False, na=False)],
    'unsupervised': lambda df: df[df['Keywords_Tags_Skills_Interests_Categories']
                                  .str.contains('unsupervised', case=False, na=False)],
    'beginner': lambda df: df[df['Course_Level'].str.lower() == 'beginner'],
    'intermediate': lambda df: df[df['Course_Level'].str.lower() == 'intermediate'],
    'advanced': lambda df: df[df['Course_Level'].str.lower() == 'advanced'],
    'machine learning': lambda df: df[df['Keywords_Tags_Skills_Interests_Categories']
                                .str.contains('machine learning', case=False, na=False)],
    'deep learning': lambda df: df[df['Keywords_Tags_Skills_Interests_Categories']
                                  .str.contains('deep learning', case=False, na=False)],
}

beginner_phrases = ["don't", "new", "no" ,"idea", "start", "started","basics", "first","dont"]

# Smart Predictor


def smart_predict(question):
    question_lower = question.lower().split()

    # Step 1: Handle beginner intent heuristics
    if any(phrase in question_lower for phrase in beginner_phrases):
        print("Rule: Detected beginner intent → Showing beginner courses")
        return modules_df[modules_df['Course_Level'].str.lower() == 'beginner'].head()

    # Step 2: Keyword filters
    matched_keywords = [kw for kw in keyword_filters if kw in question_lower]
    if matched_keywords:
        print(f"Detected keywords: {matched_keywords}")
        df_filtered = modules_df
        for kw in matched_keywords:
            df_filtered = keyword_filters[kw](df_filtered)
        return df_filtered.head()

    # Step 3: Fallback ML prediction
    print("No keywords matched — using ML classifier.")
    question_proc = preprocess(question)
    vec = vectorizer.transform([question_proc])
    predicted_level = clf.predict(vec)[0]
    print(f"ML Predicted Level: {predicted_level}")
    return modules_df[modules_df['Course_Level'] == predicted_level].head()



# Streamlit UI


if user_input:
    st.markdown("### 🔍 Recommended Modules")
    recommendations = smart_predict(user_input)
    st.dataframe(recommendations[['Module', 'Course_Level', 'Keywords_Tags_Skills_Interests_Categories']], use_container_width=True) 