from flask import Flask, render_template, request
import joblib
import sqlite3
from datetime import datetime

app = Flask(__name__)

# =========================
# LOAD ML MODEL
# =========================

model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# =========================
# DATABASE CONNECTION
# =========================

connection = sqlite3.connect(
    'database.db',
    check_same_thread=False
)

cursor = connection.cursor()

# =========================
# CREATE TABLE
# =========================
# DELETE OLD TABLE

cursor.execute("DROP TABLE IF EXISTS history")

# CREATE NEW TABLE

cursor.execute('''
CREATE TABLE history (

    id INTEGER PRIMARY KEY AUTOINCREMENT,

    username TEXT,

    review TEXT,

    prediction TEXT,

    confidence REAL,

    created_at TEXT
)
''')

connection.commit()

# =========================
# HOME PAGE
# =========================

@app.route('/', methods=['GET', 'POST'])
def home():

    prediction = ""
    confidence = 0
    emoji = "🤖"

    if request.method == 'POST':

        username = request.form.get('username')

        user_text = request.form.get('text')

        if user_text:

            # VECTORIZE INPUT

            text_vector = vectorizer.transform([user_text])

            # PREDICT

            result = model.predict(text_vector)[0]

            # CONFIDENCE SCORE

            confidence = round(
                max(model.predict_proba(text_vector)[0]) * 100,
                2
            )

            # POSITIVE / NEGATIVE

            # CONFIDENCE LOGIC
            if result == 1:
                prediction = "Positive"
                emoji = "😊"
            else:
                prediction = "Negative"
                emoji = "😠"

            current_time: str = datetime.now().strftime(
                "%d-%m-%Y %H:%M:%S"
            )

            # SAVE INTO DATABASE

            cursor.execute('''
            INSERT INTO history
            (username, review, prediction, confidence, created_at)

            VALUES (?, ?, ?, ?, ?)
            ''', (

                username,
                user_text,
                prediction,
                confidence,
                current_time
            ))

            connection.commit()

    return render_template(

        'index.html',

        prediction=prediction,

        confidence=confidence,

        emoji=emoji
    )

# =========================
# DASHBOARD PAGE
# =========================

@app.route('/dashboard')
def dashboard():

    # TOTAL REVIEWS

    cursor.execute(
        "SELECT COUNT(*) FROM history"
    )

    total = cursor.fetchone()[0]

    # POSITIVE REVIEWS

    cursor.execute(
        "SELECT COUNT(*) FROM history WHERE prediction='Positive'"
    )

    positive = cursor.fetchone()[0]

    # NEGATIVE REVIEWS

    cursor.execute(
        "SELECT COUNT(*) FROM history WHERE prediction='Negative'"
    )

    negative = cursor.fetchone()[0]

    # RECENT HISTORY

    cursor.execute('''
    SELECT username, review, prediction, confidence, created_at

    FROM history

    ORDER BY id DESC

    LIMIT 10
    ''')

    history = cursor.fetchall()

    return render_template(

        'dashboard.html',

        total=total,

        positive=positive,

        negative=negative,

        history=history
    )

# =========================
# RUN APP
# =========================

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)