from flask import Flask, render_template, request, redirect, session, url_for, flash
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash
import spacy
from collections import Counter
from textblob import TextBlob
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'supersecretkey'

# MongoDB connection
client = MongoClient('mongodb+srv://vgugan16:gugan2004@cluster0.qyh1fuo.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0')
db = client['flask_app']
users_collection = db['users']
queries_collection = db['queries']

# Load the spaCy model
nlp = spacy.load('en_core_web_sm')

# Load the courses data
courses_df = pd.read_csv(r"alison.csv", encoding='ISO-8859-1')
courses_df.columns = courses_df.columns.str.strip()  # Normalize column names

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer()

# Predefined keyword mappings
keyword_mappings = {
    'ai': 'artificial intelligence',
    'ml': 'machine learning',
    'aiml': 'artificial intelligence and machine learning'
}

def preprocess_data(courses_df):
    courses_df['Name Of The Course'] = courses_df['Name Of The Course'].str.lower().str.strip()
    courses_df['Description'] = courses_df['Description'].str.lower().str.strip()
    courses_df['Label'] = courses_df['Description'].apply(lambda x: 1 if 'machine learning' in x else 0)
    return courses_df

def train_classifier(courses_df):
    X = vectorizer.fit_transform(courses_df['Description'].fillna(''))
    y = courses_df['Label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    classifier = DecisionTreeClassifier(max_depth=3)
    classifier.fit(X_train, y_train)
    
    return classifier

# Train classifier
courses_df = preprocess_data(courses_df)
classifier = train_classifier(courses_df)

def extract_noun_phrases(text):
    text = text.lower()
    for abbr, full_form in keyword_mappings.items():
        text = text.replace(abbr, full_form)

    doc = nlp(text)
    noun_phrases = [chunk.text for chunk in doc.noun_chunks]
    noun_phrase_counts = Counter(noun_phrases)

    filtered_noun_phrases = []
    for phrase in noun_phrase_counts:
        if "certificate" not in phrase.lower():
            parts = [part.strip() for part in phrase.split('and')]
            filtered_noun_phrases.extend(parts)

    meaningful_keywords = [
        token.text for token in doc
        if token.pos_ in ["NOUN", "PROPN", "ADJ"] and "certificate" not in token.text.lower()
    ]

    named_entities = [ent.text for ent in doc.ents if ent.label_ in ["ORG", "GPE", "PRODUCT", "EVENT"]]

    final_keywords = list(set(filtered_noun_phrases + meaningful_keywords + named_entities))
    return final_keywords

def recommend_courses(keywords):
    recommended_courses = pd.DataFrame()

    if not keywords:
        return pd.DataFrame()

    keywords = [keyword.lower().strip() for keyword in keywords if keyword.strip()]
    input_vector = vectorizer.transform([' '.join(keywords)])
    predicted = classifier.predict(input_vector)

    if predicted[0] == 1:
        recommended_courses = courses_df[courses_df['Description'].str.contains('machine learning', na=False)]
    else:
        cosine_similarities = cosine_similarity(input_vector, vectorizer.transform(courses_df['Description'].fillna(''))).flatten()
        top_indices = np.argsort(cosine_similarities)[-10:][::-1]
        recommended_courses = courses_df.iloc[top_indices]
    
    recommended_courses = recommended_courses.drop_duplicates()
    return recommended_courses[['Name Of The Course', 'Description', 'Institute', 'Link']]

def save_to_mongo(query, recommendations):
    # Normalize the query by stripping whitespace and converting it to lowercase
    normalized_query = query.lower().strip()

    # Check if the query already exists in the database
    existing_record = queries_collection.find_one({"query": normalized_query})

    if not existing_record:
        # If no existing record is found, insert the new query and recommendations
        record = {
            "query": normalized_query,  # Store query in normalized form
            "recommendations": recommendations,
            "timestamp": datetime.now()
        }
        queries_collection.insert_one(record)
        print(f"Inserted new query: {query}")
    else:
        print(f"Query '{query}' already exists in the database.")
@app.route('/change_password', methods=['POST'])
def change_password():
    if 'username' not in session:
        print('Please log in to change your password.', 'warning')
        return redirect(url_for('login'))

    username = session['username']
    user = users_collection.find_one({'username': username})

    if user:
        current_password = request.form['current_password']
        new_password = request.form['new_password']

        # Check if the current password is correct
        if check_password_hash(user['password'], current_password):
            # Update the password in the database
            hashed_new_password = generate_password_hash(new_password)
            users_collection.update_one({'username': username}, {'$set': {'password': hashed_new_password}})
            print('Password changed successfully!', 'success')
        else:
            print('Current password is incorrect.', 'danger')
    else:
        print('User not found.', 'danger')

    return redirect(url_for('profile'))

@app.route('/profile')
def profile():
    if 'username' not in session:
        flash('Please log in to view your profile.', 'warning')
        return redirect(url_for('login'))

    # Fetch user details from the database
    username = session['username']
    user = users_collection.find_one({'username': username})

    if user:
        return render_template('profile.html', username=user['username'])
    else:
        flash('User not found.', 'danger')
        return redirect(url_for('home'))

def display_syntax_semantics(text):
    doc = nlp(text)
    result = f"<h2>--- Syntax and Semantics ---</h2><strong>Original Text:</strong> {text}<br><br>"

    result += "<strong>Part of Speech Tagging:</strong><br>"
    for token in doc:
        result += f"{token.text} | POS: {token.pos_} | Lemma: {token.lemma_} | Dependency: {token.dep_}<br>"

    result += "<br><strong>Named Entities:</strong><br>"
    for ent in doc.ents:
        result += f"{ent.text} ({ent.label_})<br>"

    result += "<br><strong>Noun Phrases:</strong><br>"
    for chunk in doc.noun_chunks:
        result += f"<li>{chunk.text}</li>"

    blob = TextBlob(text)
    sentiment = blob.sentiment
    polarity = sentiment.polarity
    subjectivity = sentiment.subjectivity

    result += f"<br><strong>Sentiment Analysis:</strong><br>Polarity: {polarity:.2f}<br>Subjectivity: {subjectivity:.2f}<br>"
    result += f"Sentiment: {'Positive' if polarity >= 0 else 'Negative'}<br>"

    return result
  # Redirect after logout

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = users_collection.find_one({'username': username})
        if user and check_password_hash(user['password'], password):
            session['username'] = username
            print('Login successful!', 'success')
            return redirect(url_for('home'))
        print('Invalid username or password.', 'danger')
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if users_collection.find_one({'username': username}):
            flash('Username already exists. Please log in.', 'warning')
            return redirect(url_for('login'))
        hashed_password = generate_password_hash(password)
        users_collection.insert_one({'username': username, 'password': hashed_password})
        flash('Signup successful! Please log in.', 'success')
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/home', methods=['GET', 'POST'])
def home():
    if 'username' not in session:
        flash('Please log in to access this page.', 'warning')
        return redirect(url_for('login'))

    recommendation_result = ""
    syntax_result = ""

    if request.method == "POST":
        topic = request.form.get("topic")

        if topic.strip():
            syntax_result = display_syntax_semantics(topic)
            keywords = extract_noun_phrases(topic)
            recommended_courses = recommend_courses(keywords)

            if not recommended_courses.empty:
                recommendation_result = "<h2>Recommended Courses:</h2>"
                recommendations_list = []
                for index, row in recommended_courses.iterrows():
                    course_info = {
                        "course_name": row['Name Of The Course'],
                        "description": row['Description'],
                        "institution": row['Institute'],
                        "link": row['Link']
                    }
                    recommendations_list.append(course_info)
                    recommendation_result += f"<strong>Course Name:</strong> {row['Name Of The Course']}<br>"
                    recommendation_result += f"<strong>Description:</strong> {row['Description']}<br>"
                    recommendation_result += f"<strong>Institution:</strong> {row['Institute']}<br>"
                    recommendation_result += f"<strong>Link:</strong> <a href='{row['Link']}'>{row['Link']}</a><br><br>"
                
                save_to_mongo(topic, recommendations_list)

    return render_template("index.html", recommendation_result=recommendation_result, syntax_result=syntax_result)
from flask import Flask, send_from_directory

@app.route('/static/<path:filename>')
def send_static(filename):
    return send_from_directory('static', filename)

@app.route('/logout')
def logout():
    session.pop('username', None)
    print('You have been logged out.', 'info')
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
