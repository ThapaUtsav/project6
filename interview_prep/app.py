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
import nltk
from nltk.corpus import stopwords
try:
    STOPWORDS = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    STOPWORDS = set(stopwords.words('english'))


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

# Load CSV files from local files (in same directory as app.py)
base_dir = os.path.dirname(os.path.abspath(__file__))

desc_path = os.path.join(base_dir, 'software_questions.csv')
desc_data = pd.read_csv(desc_path, encoding='ISO-8859-1', engine='python', on_bad_lines='skip')
desc_data.columns = desc_data.columns.str.strip()

mcq_path = os.path.join(base_dir, 'MCQ.csv')
mcq_data = pd.read_csv(mcq_path, on_bad_lines='skip')
mcq_data.columns = mcq_data.columns.str.strip()
mcq_data.rename(columns={'Correct Answer': 'CorrectAnswer'}, inplace=True)
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


#accuracy 
def process_answer(ref_answer, answer, user_id, question, threshold=0.95):
    embeddings = model.encode([ref_answer, answer], convert_to_tensor=True)
    similarity_score = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
    similarity_percentage = int(similarity_score * 100)

    ref_tokens = set(word.lower() for word in re.findall(r'\w+', ref_answer) if word.lower() not in STOPWORDS)
    ans_tokens = set(word.lower() for word in re.findall(r'\w+', answer) if word.lower() not in STOPWORDS)

    if similarity_score >= threshold:
        return (
            "⚠️ Your answer is too similar to the reference. Please write in your own words.",
            similarity_percentage,
            True,
            False
        )

    ai_warning = False
    if similarity_score >= 0.98 and len(answer.split()) > 30:
        ai_warning = True

    ref_word_count = len(ref_answer.split())
    ans_word_count = len(answer.split())
    too_short = ans_word_count < ref_word_count * 0.6

    keyword_overlap = len(ans_tokens & ref_tokens)
    overlap_ratio = keyword_overlap / (len(ref_tokens) + 1e-5)

    stuffing_penalty = keyword_overlap > 0 and similarity_score < 0.5 and overlap_ratio > 0.5

    feedback = []

    
    if len(ans_tokens) < 3 or answer.lower().strip() in ["yes", "no", "okay", "idk", "i don’t know", "phishing is great", "phishing is god"]:
        feedback.append("❌ Your answer is too short or lacks relevance. Try explaining in your own words.")
        star_rating = 1
        return "\n".join(feedback), similarity_percentage, False, ai_warning

    if stuffing_penalty:
        feedback.append("❌ Your answer includes keywords but lacks proper context or explanation.")
        feedback.append("📌 Try writing in full sentences that show understanding.")
    elif similarity_percentage > 85:
        feedback.append("✅ Excellent answer! Covers most or all key ideas.")
    elif similarity_percentage > 70:
        feedback.append("👍 Good effort. Expand on specific details.")
    elif similarity_percentage > 50:
        feedback.append("⚠️ Decent start, but missing several key points.")
    else:
        feedback.append("❌ Weak answer. Needs major improvement in clarity and content.")

    if too_short:
        feedback.append("📉 Your answer is too short compared to the reference. Add more detail.")

    
    missing_words = list(ref_tokens - ans_tokens)
    if len(missing_words) >= 3:
        feedback.append("🧠 Consider including key ideas like: " + ', '.join(missing_words[:5]))

    
    if stuffing_penalty or too_short or len(ans_tokens) < 5:
        star_rating = 1
    elif similarity_percentage >= 90:
        star_rating = 5
    elif similarity_percentage >= 75:
        star_rating = 4
    elif similarity_percentage >= 60:
        star_rating = 3
    elif similarity_percentage >= 45:
        star_rating = 2
    else:
        star_rating = 1

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
    categories = sorted(mcq_data['Category'].unique())
    difficulties = sorted(mcq_data['Difficulty'].unique())

    selected_category = request.args.get('category', categories[0] if categories else '')
    selected_difficulty = request.args.get('difficulty', difficulties[0] if difficulties else '')

    filtered_mcq_questions = [q for q in mcq_questions if
                              (selected_category == '' or q['Category'] == selected_category) and
                              (selected_difficulty == '' or q['Difficulty'] == selected_difficulty)]

    page = int(request.args.get('page', 1))
    total_questions = len(filtered_mcq_questions)
    total_pages = (total_questions + QUESTIONS_PER_PAGE - 1) // QUESTIONS_PER_PAGE
    page = max(1, min(page, total_pages))

    start_index = (page - 1) * QUESTIONS_PER_PAGE
    end_index = min(start_index + QUESTIONS_PER_PAGE, total_questions)
    questions_to_show = filtered_mcq_questions[start_index:end_index]

    if 'mcq_answers' not in session:
        session['mcq_answers'] = {}

    if request.method == 'POST':
        form_answers = {k: v for k, v in request.form.items() if k.startswith('answer-')}
        session['mcq_answers'].update(form_answers)
        session.modified = True

        form_category = request.form.get('category', selected_category)
        form_difficulty = request.form.get('difficulty', selected_difficulty)

        if 'next' in request.form:
            next_page = min(page + 1, total_pages)
            return redirect(url_for('mcq_test', page=next_page, category=form_category, difficulty=form_difficulty))
        elif 'prev' in request.form:
            prev_page = max(page - 1, 1)
            return redirect(url_for('mcq_test', page=prev_page, category=form_category, difficulty=form_difficulty))
        elif 'submit' in request.form:
            answers = session.pop('mcq_answers', {})
            correct_count = 0
            total_answered = 0
            for i, question in enumerate(filtered_mcq_questions):
                key = f"answer-{i}"
                if key in answers and answers[key].strip():
                    total_answered += 1
                    
                    correct_answer = question.get('CorrectAnswer') or question.get('Correct Answer')
                    if correct_answer and answers[key].strip().upper() == correct_answer.strip().upper():
                        correct_count += 1
            score = round((correct_count / total_answered) * 100, 2) if total_answered > 0 else 0
            return render_template("mcq_result.html",
                       total_questions=total_questions,
                       total_answered=total_answered,
                       correct_count=correct_count,
                       score=score,
                       selected_category=form_category,
                       selected_difficulty=form_difficulty)

    answers = session.get('mcq_answers', {})

    return render_template('mcq_test.html',
                           categories=categories,
                           difficulties=difficulties,
                           selected_category=selected_category,
                           selected_difficulty=selected_difficulty,
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

@app.route('/admin/add_descriptive_question', methods=['POST'])
@login_required
@admin_required
def add_descriptive_question():
    global desc_data
    question = request.form.get('question', '').strip()
    answer = request.form.get('answer', '').strip()
    category = request.form.get('category', '').strip()
    difficulty = request.form.get('difficulty', '').strip()

    if question and answer and category and difficulty:
        new_row = {'Question': question, 'Answer': answer, 'Category': category, 'Difficulty': difficulty}
        desc_data = pd.concat([desc_data, pd.DataFrame([new_row])], ignore_index=True)
        desc_data.to_csv(desc_path, index=False)
        flash('Descriptive question added successfully.', 'success')
    else:
        flash('Please fill in all fields for descriptive question.', 'danger')

    return redirect(url_for('admin'))

@app.route('/admin/add_mcq_question', methods=['POST'])
@login_required
@admin_required
def add_mcq_question():
    global mcq_data, mcq_questions
    question = request.form.get('question', '').strip()
    option_a = request.form.get('option_a', '').strip()
    option_b = request.form.get('option_b', '').strip()
    option_c = request.form.get('option_c', '').strip()
    option_d = request.form.get('option_d', '').strip()
    correct = request.form.get('correct_answer', '').strip().upper()

    if question and option_a and option_b and option_c and option_d and correct in ['A', 'B', 'C', 'D']:
        new_row = {
            'Question': question,
            'Option A': option_a,
            'Option B': option_b,
            'Option C': option_c,
            'Option D': option_d,
            'CorrectAnswer': correct
        }
        mcq_data = pd.concat([mcq_data, pd.DataFrame([new_row])], ignore_index=True)
        mcq_data.to_csv(mcq_path, index=False)
        mcq_questions = mcq_data.to_dict(orient='records')
        flash('MCQ question added successfully.', 'success')
    else:
        flash('Please fill in all fields for MCQ question and ensure correct answer is A, B, C, or D.', 'danger')

    return redirect(url_for('admin'))
@app.route('/admin')
@login_required
@admin_required
def admin():
    conn = sqlite3.connect('database.db')
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM users")
    user_count = cur.fetchone()[0]
    conn.close()

    descriptive_count = len(desc_data)
    mcq_count = len(mcq_data)

    return render_template('admin.html',
                           user_count=user_count,
                           descriptive_count=descriptive_count,
                           mcq_count=mcq_count)


if __name__ == '__main__':
    app.run(debug=True)
