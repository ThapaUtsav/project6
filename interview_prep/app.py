import re
import os
import random
import sqlite3
from datetime import datetime
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash, session, abort
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from sentence_transformers import SentenceTransformer, InputExample, losses, util
from torch.utils.data import DataLoader
from functools import wraps

os.environ["WANDB_DISABLED"] = "true"

app = Flask(__name__)
app.secret_key = 'your_secret_key'

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin):
    def __init__(self, id_, username, password, is_admin=False):
        self.id = id_
        self.username = username
        self.password = password
        self.is_admin = is_admin

@login_manager.user_loader
def load_user(user_id):
    conn = sqlite3.connect('database.db')
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    row = cur.fetchone()
    conn.close()
    if row:
        return User(id_=row[0], username=row[1], password=row[2], is_admin=bool(row[3]))
    return None

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or not current_user.is_admin:
            abort(403)
        return f(*args, **kwargs)
    return decorated_function

def init_db():
    conn = sqlite3.connect('database.db')
    cur = conn.cursor()
    cur.execute('''CREATE TABLE IF NOT EXISTS users 
                   (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT NOT NULL UNIQUE, password TEXT NOT NULL, is_admin INTEGER DEFAULT 0)''')
    cur.execute('''CREATE TABLE IF NOT EXISTS user_answers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            question TEXT,
            submitted_answer TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )''')

    cur.execute("SELECT * FROM users WHERE username = 'admin'")
    if not cur.fetchone():
        hashed_pw = generate_password_hash('password')
        cur.execute("INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)", ('admin', hashed_pw, 1))

    conn.commit()
    conn.close()

init_db()

desc_url = "https://raw.githubusercontent.com/ThapaUtsav/project6/main/Software%20Questions.csv"
desc_data = pd.read_csv(desc_url, encoding='ISO-8859-1', engine='python', on_bad_lines='skip')
desc_data.columns = desc_data.columns.str.strip()

mcq_url = "https://raw.githubusercontent.com/ThapaUtsav/project6/main/MCQ.csv"
mcq_data = pd.read_csv(mcq_url, on_bad_lines='skip')
mcq_data.columns = mcq_data.columns.str.strip()
mcq_questions = mcq_data.to_dict(orient='records')

QUESTIONS_PER_PAGE = 5

model = SentenceTransformer('all-MiniLM-L6-v2')
train_data = [InputExample(texts=[row['Question'], row['Answer']], label=1.0) for _, row in desc_data.iterrows()]
train_dataloader = DataLoader(train_data, batch_size=8)
train_loss = losses.CosineSimilarityLoss(model)
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=10)

def save_user_answer(user_id, question, answer):
    conn = sqlite3.connect('database.db')
    cur = conn.cursor()
    cur.execute("INSERT INTO user_answers (user_id, question, submitted_answer, timestamp) VALUES (?, ?, ?, ?)",
                (user_id, question, answer, datetime.now()))
    conn.commit()
    conn.close()

def is_plagiarized(user_id, answer, question, threshold=0.85):
    answer_tokens = set(re.findall(r'\w+', answer.lower()))
    conn = sqlite3.connect('database.db')
    cur = conn.cursor()
    cur.execute("SELECT submitted_answer FROM user_answers WHERE user_id != ? AND question = ?", (user_id, question))
    previous_answers = [row[0] for row in cur.fetchall()]
    conn.close()
    for prev in previous_answers:
        prev_tokens = set(re.findall(r'\w+', prev.lower()))
        union = len(answer_tokens | prev_tokens)
        if union == 0:
            continue
        jaccard = len(answer_tokens & prev_tokens) / union
        if jaccard > threshold:
            return True, jaccard
    return False, 0

def process_answer(ref_answer, answer, user_id, question, threshold=0.95):
    embeddings = model.encode([ref_answer, answer], convert_to_tensor=True)
    similarity_score = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
    similarity_percentage = int(similarity_score * 100)
    if similarity_score >= threshold:
        return ("⚠️ Your answer is too similar to the reference. Please write in your own words.", similarity_percentage, True, False)
    ai_warning = False
    if similarity_score >= 0.98 and len(answer.split()) > 30:
        ai_warning = True
    feedback = []
    if similarity_percentage > 85:
        feedback.append("✅ Excellent answer! Covers most or all key ideas.")
    elif similarity_percentage > 70:
        feedback.append("👍 Good job. Try to expand on details.")
    elif similarity_percentage > 50:
        feedback.append("⚠️ Decent start, but missing several key points.")
    else:
        feedback.append("❌ Weak answer. Needs major improvement.")
    if len(answer.split()) < len(ref_answer.split()) * 0.6:
        feedback.append("📉 Your answer is too short compared to the reference. Add more detail.")
    ref_words = set(ref_answer.lower().split())
    ans_words = set(answer.lower().split())
    missed = ref_words - ans_words
    if len(missed) > 5:
        feedback.append(f"🧠 Consider including these important words: {', '.join(list(missed)[:5])}")
    return "\n".join(feedback), similarity_percentage, False, ai_warning

@app.route('/')
def home():
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = generate_password_hash(request.form['password'])
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        try:
            cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
            conn.commit()
        except sqlite3.IntegrityError:
            flash("Username already exists.", "danger")
            return redirect('/register')
        conn.close()
        flash("Registration successful. Please login.", "success")
        return redirect('/login')
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = sqlite3.connect('database.db')
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE username = ?", (username,))
        row = cur.fetchone()
        conn.close()
        if row and check_password_hash(row[2], password):
            user = User(id_=row[0], username=row[1], password=row[2], is_admin=bool(row[3]))
            login_user(user)
            if user.is_admin:
                return redirect(url_for('admin'))
            else:
                return redirect(url_for('select_test'))
        flash("Invalid username or password.", "danger")
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/select_test')
@login_required
def select_test():
    return render_template('test_selection.html')

@app.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():
    categories = sorted(desc_data['Category'].unique())
    difficulties = sorted(desc_data['Difficulty'].unique())
    selected_category = request.form.get('category', categories[0] if categories else '')
    selected_difficulty = request.form.get('difficulty', difficulties[0] if difficulties else '')
    filtered_data = desc_data[
        (desc_data['Category'] == selected_category) &
        (desc_data['Difficulty'] == selected_difficulty)
    ]
    questions = filtered_data.to_dict(orient='records')
    feedback = None
    star_rating = 0
    reference_answer = ''
    answer = ''
    selected_question = ''
    if request.method == 'POST' and 'answer' in request.form:
        selected_question = request.form.get('question')
        answer = request.form.get('answer', '').strip()
        question_row = filtered_data[filtered_data['Question'] == selected_question]
        if not question_row.empty:
            reference_answer = question_row.iloc[0]['Answer']
            feedback, similarity_percentage, is_plagiarized, ai_warning = process_answer(reference_answer, answer, current_user.id, selected_question)
            star_rating = max(1, min(5, int(similarity_percentage // 20)))
            if is_plagiarized:
                flash("⚠️ Your answer is too similar to the reference answer. Please rewrite in your own words.", "danger")
            else:
                save_user_answer(current_user.id, selected_question, answer)
                if ai_warning:
                    flash("⚠️ This answer may have been generated by an AI tool. Please ensure it is your own work.", "warning")
    return render_template('dashboard.html',
                           categories=categories,
                           difficulties=difficulties,
                           selected_category=selected_category,
                           selected_difficulty=selected_difficulty,
                           questions=questions,
                           selected_question=selected_question,
                           answer=answer,
                           feedback=feedback,
                           star_rating=star_rating,
                           reference_answer=reference_answer)

@app.route('/mcq_test', methods=['GET', 'POST'])
@login_required
def mcq_test():
    page = int(request.args.get('page', 1))
    total_questions = len(mcq_questions)
    total_pages = (total_questions + QUESTIONS_PER_PAGE - 1) // QUESTIONS_PER_PAGE

    start_index = (page - 1) * QUESTIONS_PER_PAGE
    end_index = min(start_index + QUESTIONS_PER_PAGE, total_questions)
    questions_to_show = mcq_questions[start_index:end_index]

    answers = {}
    if request.method == 'POST':
        answers = request.form.to_dict()

    return render_template('mcq_test.html',
                           questions=questions_to_show,
                           page=page,
                           total_pages=total_pages,
                           answers=answers,
                           start_index=start_index)

@app.route('/questions')
@login_required
def questions():
    categories = sorted(desc_data['Category'].unique())
    difficulties = sorted(desc_data['Difficulty'].unique())
    selected_category = request.args.get('category', categories[0] if categories else '')
    selected_difficulty = request.args.get('difficulty', difficulties[0] if difficulties else '')
    filtered_questions = desc_data[
        (desc_data['Category'] == selected_category) &
        (desc_data['Difficulty'] == selected_difficulty)
    ].to_dict(orient='records')
    return render_template('question.html',
                           categories=categories,
                           difficulties=difficulties,
                           selected_category=selected_category,
                           selected_difficulty=selected_difficulty,
                           filtered_questions=filtered_questions)

@app.route('/admin', methods=['GET'])
@login_required
@admin_required
def admin():
    conn = sqlite3.connect('database.db')
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM users")
    user_count = cur.fetchone()[0]
    conn.close()
    descriptive_count = len(desc_data)
    mcq_count = len(mcq_questions)
    return render_template('admin.html', user_count=user_count, descriptive_count=descriptive_count, mcq_count=mcq_count)

if __name__ == '__main__':
    app.run(debug=True)
