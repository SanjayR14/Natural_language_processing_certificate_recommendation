from flask import Flask, request, render_template
import spacy
from collections import Counter
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import numpy as np

# Sample dataset (same as before)
courses_data = [
    {
        "Name Of The Course": "Introduction to Machine Learning",
        "Description": "Learn the basics of machine learning including supervised and unsupervised learning techniques.",
        "Institute": "Alison",
        "Link": "https://alison.com/course/introduction-to-machine-learning"
    },
    {
        "Name Of The Course": "Artificial Intelligence in Practice",
        "Description": "Explore practical applications of artificial intelligence in various industries.",
        "Institute": "Alison",
        "Link": "https://alison.com/course/artificial-intelligence-in-practice"
    },
    # ... (rest of the courses)
]

# Convert to DataFrame
courses_df = pd.DataFrame(courses_data)

# Load the spaCy model
nlp = spacy.load('en_core_web_sm')

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer()

# Function to preprocess course descriptions
def preprocess_data(courses_df):
    courses_df['Name Of The Course'] = courses_df['Name Of The Course'].str.lower().str.strip()
    courses_df['Description'] = courses_df['Description'].str.lower().str.strip()
    courses_df['Label'] = courses_df['Description'].apply(lambda x: 1 if 'machine learning' in x else 0)
    return courses_df

# Function to train the classifier
def train_classifier(courses_df):
    X = vectorizer.fit_transform(courses_df['Description'].fillna(''))
    y = courses_df['Label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    classifier = DecisionTreeClassifier(max_depth=None)
    classifier.fit(X_train, y_train)
    return classifier

# Preprocess data and train classifier
courses_df = preprocess_data(courses_df)
classifier = train_classifier(courses_df)

def extract_noun_phrases(text):
    doc = nlp(text)
    noun_phrases = [chunk.text for chunk in doc.noun_chunks]
    return list(set(noun_phrases))

def recommend_courses(keywords):
    recommended_courses = pd.DataFrame()

    if 'Name Of The Course' not in courses_df.columns or 'Description' not in courses_df.columns:
        return recommended_courses

    if not keywords:
        return pd.DataFrame()

    keywords = [keyword.lower().strip() for keyword in keywords if keyword.strip()]
    input_vector = vectorizer.transform([' '.join(keywords)])
    predicted = classifier.predict(input_vector)

    if 'ai' in keywords:
        recommended_courses = courses_df[courses_df['Description'].str.contains('artificial intelligence|ai', na=False, case=False)]
    elif 'ml' in keywords:
        recommended_courses = courses_df[courses_df['Description'].str.contains('machine learning|ml', na=False, case=False)]
    else:
        cosine_similarities = cosine_similarity(input_vector, vectorizer.transform(courses_df['Description'].fillna(''))).flatten()
        top_indices = np.argsort(cosine_similarities)[-20:][::-1]
        recommended_courses = courses_df.iloc[top_indices]
    
    recommended_courses = recommended_courses.drop_duplicates()
    return recommended_courses[['Name Of The Course', 'Description', 'Institute', 'Link']]

def display_syntax_semantics(text):
    doc = nlp(text)
    result = "<h2>--- Syntax and Semantics ---</h2>"
    result += f"<strong>Original Text:</strong> {text}<br><br>"

    result += "<strong>Part of Speech Tagging:</strong><br>"
    for token in doc:
        result += f"{token.text} | POS: {token.pos_} | Lemma: {token.lemma_} | Dependency: {token.dep_}<br>"

    result += "<br><strong>Named Entities:</strong><br>"
    for ent in doc.ents:
        result += f"{ent.text} ({ent.label_})<br>"

    result += "<br><strong>Noun Phrases:</strong><br>"
    for chunk in doc.noun_chunks:
        result += f"<li>{chunk.text}</li>"

    return result

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_input = request.form['user_input']
        keywords = extract_noun_phrases(user_input)
        recommended_courses = recommend_courses(keywords)
        syntax_semantics = display_syntax_semantics(user_input)
        return render_template('index.html', recommended_courses=recommended_courses, syntax_semantics=syntax_semantics)
    return render_template('index.html', recommended_courses=pd.DataFrame(), syntax_semantics='')


