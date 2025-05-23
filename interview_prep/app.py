import warnings
import os
import random
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
from sentence_transformers import SentenceTransformer, InputExample, losses, util
from torch.utils.data import DataLoader

warnings.simplefilter(action='ignore', category=FutureWarning)
os.environ["WANDB_DISABLED"] = "true"

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Flask-Login setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin):
    def __init__(self, id_, username, password):
        self.id = id_
        self.username = username
        self.password = password

@login_manager.user_loader
def load_user(user_id):
    conn = sqlite3.connect('database.db')
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    row = cur.fetchone()
    conn.close()
    if row:
        return User(id_=row[0], username=row[1], password=row[2])
    return None

def init_db():
    conn = sqlite3.connect('database.db')
    cur = conn.cursor()
    cur.execute('''CREATE TABLE IF NOT EXISTS users 
                   (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT NOT NULL UNIQUE, password TEXT NOT NULL)''')
    conn.commit()
    conn.close()

init_db()

# Load descriptive questions dataset
desc_url = "https://raw.githubusercontent.com/ThapaUtsav/project6/main/Software%20Questions.csv"
desc_data = pd.read_csv(desc_url, encoding='ISO-8859-1', engine='python', on_bad_lines='skip')
desc_data.columns = desc_data.columns.str.strip()

# Load MCQ dataset and fix columns
mcq_url = "https://raw.githubusercontent.com/ThapaUtsav/project6/main/MCQ.csv"
mcq_data = pd.read_csv(mcq_url, on_bad_lines='skip')  
mcq_data.columns = mcq_data.columns.str.strip()
mcq_questions = mcq_data.to_dict(orient='records')

mcq_questions = mcq_data.to_dict(orient='records')

QUESTIONS_PER_PAGE = 5

# Load and train SBERT model for descriptive answers (kept from your original code)
model = SentenceTransformer('all-MiniLM-L6-v2')
train_data = [InputExample(texts=[row['Question'], row['Answer']], label=1.0) for _, row in desc_data.iterrows()]
train_dataloader = DataLoader(train_data, batch_size=8)
train_loss = losses.CosineSimilarityLoss(model)
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=10)

def give_feedback(score, answer, reference_answer):
    similarity_percentage = int(score * 100)
    if similarity_percentage > 85:
        feedback = "✅ Excellent answer! You covered all key points, and the structure is perfect. Keep it up!"
    elif similarity_percentage > 70:
        feedback = "👍 Good job. Consider adding a bit more detail or elaborating on some key points."
    elif similarity_percentage > 50:
        feedback = "⚠️ Decent start, but you missed some key parts. Review the question and make sure to cover all aspects."
    elif similarity_percentage > 30:
        feedback = "❌ The answer lacks core concepts. Review the topic and try again, focusing on important points."
    else:
        feedback = "❌ The answer doesn't seem to address the question well. Please review the material and try again."
    missing_points = compare_answers(answer, reference_answer)
    if missing_points:
        feedback += "\n\nAdditional Notes:\n" + "\n".join(missing_points)
    return feedback, similarity_percentage

def compare_answers(answer, reference_answer):
    answer_tokens = set(answer.lower().split())
    ref_answer_tokens = set(reference_answer.lower().split())
    missing_points = []
    if not answer_tokens & ref_answer_tokens:
        missing_points.append("Key concepts or keywords are missing.")
    else:
        missing_points = list(ref_answer_tokens - answer_tokens)
        if missing_points:
            missing_points = [f"Consider adding more details on: {', '.join(missing_points)}"]
    return missing_points

def process_answer(ref_answer, answer):
    embeddings = model.encode([ref_answer, answer], convert_to_tensor=True)
    similarity_score = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
    
    # Tokenize and find missing keywords
    answer_tokens = set(answer.lower().split())
    ref_answer_tokens = set(ref_answer.lower().split())
    missing_keywords = ref_answer_tokens - answer_tokens
    penalty = len(missing_keywords) / max(len(ref_answer_tokens), 1)
    
    # Reduce similarity score proportional to missing keywords (tune factor as needed)
    adjusted_score = similarity_score * (1 - penalty * 0.6)  # 0.6 is a penalty weight you can tune
    
    feedback, similarity_percentage = give_feedback(adjusted_score, answer, ref_answer)
    return feedback, similarity_percentage

def similarity_to_stars(score):
    percentage = score * 100
    if percentage > 80:
        stars = 5
    elif percentage > 60:
        stars = 4
    elif percentage > 40:
        stars = 3
    elif percentage > 20:
        stars = 2
    else:
        stars = 1
    return stars

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
            user = User(id_=row[0], username=row[1], password=row[2])
            login_user(user)
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

            feedback, similarity_percentage = process_answer(reference_answer, answer)
            star_rating = similarity_to_stars(similarity_percentage / 100)

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
    
    start_idx = (page - 1) * QUESTIONS_PER_PAGE
    end_idx = min(start_idx + QUESTIONS_PER_PAGE, total_questions)
    
    questions_to_show = mcq_questions[start_idx:end_idx]

    # Initialize or get answers dict from session
    if 'answers' not in session:
        session['answers'] = {}

    if request.method == 'POST':
        # Save answers from this page
        for i, q in enumerate(questions_to_show):
            ans = request.form.get(f'answer-{start_idx + i}')
            if ans:
                session['answers'][str(start_idx + i)] = ans
        session.modified = True

        if 'next' in request.form:
            if page < total_pages:
                return redirect(url_for('mcq_test', page=page + 1))
        elif 'prev' in request.form:
            if page > 1:
                return redirect(url_for('mcq_test', page=page - 1))
        elif 'submit' in request.form:
            correct_count = 0
            total_answered = 0
            for idx_str, user_ans in session['answers'].items():
                idx = int(idx_str)
                correct_option = mcq_questions[idx]['CorrectOption'].strip().upper()
                if user_ans.strip().upper() == correct_option:
                    correct_count += 1
                total_answered += 1
            
            score = 0
            if total_answered > 0:
                score = (correct_count / total_answered) * 100

            session.pop('answers', None)  # Clear saved answers on submission
            return render_template('mcq_result.html', score=round(score, 2), total_answered=total_answered, correct_count=correct_count)

    return render_template(
        'mcq_test.html',
        questions=questions_to_show,
        page=page,
        total_pages=total_pages,
        start_index=start_idx,
        answers=session['answers']
    )

if __name__ == '__main__':
    app.run(debug=True)
