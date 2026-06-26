"""
CBT Exam Simulator — Advanced Backend
Routes, question parser (TXT + PDF), history tracking, error log vault, and grading engine.
"""

import os
import re
import json
import uuid
import random
import datetime
from flask import Flask, render_template, request, redirect, url_for, session, jsonify

app = Flask(__name__)
app.secret_key = os.urandom(32)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB for PDFs

# In-memory store for exam data (keyed by session id)
exam_store = {}


# ══════════════════════════════════════════════════
#  TEXT PARSER — Dynamic Delimited format
# ══════════════════════════════════════════════════

def parse_questions(file_content: str) -> list[dict]:
    """
    Parse pipe-delimited question file into structured data.
    Supports dynamic number of options (3 or 4 choices).
    Formats supported:
      Standalone:  Q | A | B | C | [D] | CorrectAns
      Vignette:    VIGNETTE_START | Case text... | VIGNETTE_END
                   followed by Q | A | B | C | [D] | CorrectAns blocks
    """
    questions = []
    current_topic = "General"
    current_vignette = None
    in_vignette = False
    vignette_parts = []

    lines = file_content.strip().split('\n')

    for line in lines:
        line = line.strip()

        # Skip empty lines and comments
        if not line or line.startswith('#'):
            continue

        # Topic marker
        if line.startswith('TOPIC:'):
            current_topic = line[6:].strip()
            continue

        # Parse pipe-delimited parts
        parts = [p.strip() for p in line.split('|')]

        # Vignette start
        if parts[0].upper() == 'VIGNETTE_START':
            in_vignette = True
            vignette_parts = []
            for part in parts[1:]:
                if part.upper() == 'VIGNETTE_END':
                    in_vignette = False
                    current_vignette = ' '.join(vignette_parts).strip()
                    break
                else:
                    vignette_parts.append(part)
            continue

        # Vignette end on its own line
        if in_vignette:
            if any(p.upper() == 'VIGNETTE_END' for p in parts):
                for part in parts:
                    if part.upper() == 'VIGNETTE_END':
                        break
                    vignette_parts.append(part)
                in_vignette = False
                current_vignette = '\n'.join(vignette_parts).strip()
            else:
                vignette_parts.append(line)
            continue

        # Dynamic question line: Q | Option 1 | Option 2 | Option 3 | [Option 4] | CorrectAns
        if len(parts) >= 5:
            q_text = parts[0]
            correct = parts[-1].strip().upper()
            
            choices = {}
            letters = ['A', 'B', 'C', 'D', 'E', 'F']
            num_options = len(parts) - 2  # subtract question and answer key
            for idx in range(num_options):
                if idx < len(letters):
                    choices[letters[idx]] = parts[idx + 1]

            question = {
                'id': len(questions) + 1,
                'text': q_text,
                'choices': choices,
                'correct': correct,
                'topic': current_topic,
                'vignette': current_vignette,
                'explanation': None,
                'answer_found': True,
            }
            questions.append(question)
        elif current_vignette is None and not in_vignette:
            continue

    return questions


# ══════════════════════════════════════════════════
#  PDF PARSER — Multi-Exam structure (A-D options)
# ══════════════════════════════════════════════════

def parse_pdf_text(pdf_bytes: bytes) -> list[dict]:
    """Parse a Schweser/Prep PDF into structured questions page by page."""
    from pypdf import PdfReader
    import io

    reader = PdfReader(io.BytesIO(pdf_bytes))
    full_text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            full_text += page_text + "\n"

    return _parse_schweser_text(full_text)


def _parse_schweser_text(full_text: str) -> list[dict]:
    """Parse question blocks with dynamic choices logic."""
    questions = []

    q_pattern = re.compile(
        r'(?:Question\s*#?\s*(\d+))',
        re.IGNORECASE
    )

    q_matches = list(q_pattern.finditer(full_text))
    if not q_matches:
        return questions

    for idx, match in enumerate(q_matches):
        q_num = match.group(1)
        start = match.end()
        end = q_matches[idx + 1].start() if idx + 1 < len(q_matches) else len(full_text)
        block = full_text[start:end].strip()

        parsed = _parse_single_question_block(block, len(questions) + 1)
        if parsed:
            questions.append(parsed)

    # Trigger Split-PDF Document Scan if inline answers are missing
    unresolved_count = sum(1 for q in questions if not q.get('answer_found', False))
    if len(questions) > 0 and (unresolved_count / len(questions)) > 0.7:
        parts = re.split(r'(?:Answer\s*Key|Solutions|Appendix)', full_text, flags=re.IGNORECASE)
        if len(parts) > 1:
            answer_block = "\n".join(parts[1:])
            pairs = re.findall(r'\b(?:Question|Q)?\s*(\d+)\s*[\.\:\-\s\)]+\s*([A-D])\b', answer_block, re.IGNORECASE)
            fallback_map = {int(num): letter.upper() for num, letter in pairs}
            for q in questions:
                q_id = q['id']
                if q_id in fallback_map:
                    q['correct'] = fallback_map[q_id]
                    q['answer_found'] = True

    return questions


def _parse_single_question_block(block: str, q_id: int) -> dict | None:
    """Parse a single block matching A-D choices dynamically."""
    
    # Extract options A-D
    opt_pattern = re.compile(
        r'^([A-D])\)\s*(.+?)(?=^[A-D]\)|^(?:Answer|Ans|Explanation|Correct)|$)',
        re.MULTILINE | re.DOTALL | re.IGNORECASE
    )
    options_raw = opt_pattern.findall(block)

    choices = {}
    for letter, text in options_raw:
        choices[letter.upper()] = text.strip().rstrip('.')

    if len(choices) < 2:
        # Fallback to A. B. C. D. style
        opt_pattern2 = re.compile(
            r'^([A-D])\.\s*(.+?)(?=^[A-D]\.|^(?:Answer|Ans|Explanation|Correct)|$)',
            re.MULTILINE | re.DOTALL | re.IGNORECASE
        )
        options_raw2 = opt_pattern2.findall(block)
        for letter, text in options_raw2:
            choices[letter.upper()] = text.strip().rstrip('.')

    if len(choices) < 2:
        return None

    # Extract question text before first option match
    first_opt_match = re.search(r'^[A-D][)\.]', block, re.MULTILINE)
    q_text = block[:first_opt_match.start()].strip() if first_opt_match else block[:100].strip()

    # Strip pagination header artifacts (like "of X" or "Question # of #")
    q_text = re.sub(r'^\s*(?:Question\s*\d+\s*of\s*\d+|Question\s*#?\s*of\s*#?|of\s*\d+|of\s*[A-Z_#]+)\b\s*', '', q_text, flags=re.IGNORECASE)

    # Strip "Question ID: [0-9]+" text patterns (including brackets/parentheses)
    q_text = re.sub(r'[\(\[\s]*Question\s*ID\s*:\s*\d+[\)\]\s]*', ' ', q_text, flags=re.IGNORECASE)

    q_text = re.sub(r'\s+', ' ', q_text).strip()

    # Extract explanation
    explanation_text = None
    expl_match = re.search(
        r'(?:Explanation|Rationale)[:\s]*(.+)',
        block, re.IGNORECASE | re.DOTALL
    )
    if expl_match:
        explanation_text = expl_match.group(1).strip()
        next_q = re.search(r'Question\s*#?\s*\d+', explanation_text, re.IGNORECASE)
        if next_q:
            explanation_text = explanation_text[:next_q.start()].strip()

    # Detect answer key
    correct_answer, answer_found = _find_correct_answer(block, choices, explanation_text)

    return {
        'id': q_id,
        'text': q_text,
        'choices': choices,
        'correct': correct_answer,
        'topic': 'General',
        'vignette': None,
        'explanation': explanation_text,
        'answer_found': answer_found,
    }


def _find_correct_answer(block: str, choices: dict, explanation_text: str | None) -> tuple[str, bool]:
    """Detect answer key using explicit strings or explanation overlap analysis."""
    ans_patterns = [
        r'(?:Answer|Ans|Correct\s*Answer)[:\s]*([A-D])',
        r'(?:The\s+(?:correct|right)\s+answer\s+is)\s*[:\s]*([A-D])',
        r'([A-D])\s+is\s+(?:correct|the\s+correct\s+answer)',
    ]
    for pat in ans_patterns:
        m = re.search(pat, block, re.IGNORECASE)
        if m:
            return m.group(1).upper(), True

    if explanation_text and choices:
        return _best_overlap_answer(choices, explanation_text), True

    return 'A', False


def _best_overlap_answer(choices: dict, explanation_text: str) -> str:
    """Compare choice word occurrences inside the explanation string."""
    def _clean_words(text: str) -> set:
        return set(w.lower() for w in re.findall(r'[a-zA-Z]{3,}', text))

    expl_words = _clean_words(explanation_text)
    if not expl_words:
        return 'A'

    best_letter = 'A'
    best_score = -1

    for letter, text in choices.items():
        option_words = _clean_words(text)
        if not option_words:
            continue
        overlap = len(option_words & expl_words)
        score = overlap / len(option_words) if option_words else 0
        if score > best_score:
            best_score = score
            best_letter = letter

    return best_letter


# ══════════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════════

@app.route('/')
def setup():
    """Serve the configuration setup screen."""
    # Capture mistake logs flag if redirecting
    error_flag = request.args.get('error')
    return render_template('setup.html', error=error_flag)


@app.route('/demo')
def demo():
    """Auto-load sample questions for testing."""
    sample_path = os.path.join(os.path.dirname(__file__), 'sample_questions.txt')
    if not os.path.exists(sample_path):
        return redirect(url_for('setup'))

    with open(sample_path, 'r', encoding='utf-8') as f:
        content = f.read()

    questions = parse_questions(content)
    if not questions:
        return redirect(url_for('setup'))

    exam_id = str(uuid.uuid4())
    exam_store[exam_id] = {
        'questions': questions,
        'duration': 15,
        'session': 'AM',
        'mark_correct': 3,
        'mark_incorrect': 0,
        'candidate_name': 'Demo Candidate',
        'exam_topic': 'General Knowledge Sample Practice'
    }
    session['exam_id'] = exam_id
    return redirect(url_for('exam'))


def _extract_text_from_file(file_obj) -> str:
    if not file_obj or file_obj.filename == '':
        return ""
    filename = file_obj.filename.lower()
    raw_bytes = file_obj.read()
    if filename.endswith('.pdf'):
        from pypdf import PdfReader
        import io
        reader = PdfReader(io.BytesIO(raw_bytes))
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    else:
        return raw_bytes.decode('utf-8', errors='ignore')


@app.route('/upload', methods=['POST'])
def upload():
    """Process file upload and configuration settings."""
    duration = int(request.form.get('duration', 135))
    exam_session = request.form.get('session', 'AM')
    mark_correct = int(request.form.get('mark_correct', 3))
    mark_incorrect = int(request.form.get('mark_incorrect', 0))
    candidate_name = request.form.get('candidate_name', 'Anonymous Candidate').strip()
    exam_topic = request.form.get('exam_topic', 'General Reading').strip()
    target_exam = request.form.get('target_exam', 'CFA').strip()

    file = request.files.get('question_file')
    answer_file = request.files.get('answer_file')

    if not file or file.filename == '':
        return redirect(url_for('setup'))

    filename = file.filename.lower()
    raw_bytes = file.read()

    if filename.endswith('.pdf'):
        questions = parse_pdf_text(raw_bytes)
    else:
        content = raw_bytes.decode('utf-8', errors='ignore')
        questions = parse_questions(content)

    # Process separate answer key file if provided
    if answer_file and answer_file.filename != '':
        answer_text = _extract_text_from_file(answer_file)
        if answer_text:
            pairs = re.findall(r'\b(?:Question|Q)?\s*(\d+)\s*[\.\:\-\s\)]+\s*([A-D])\b', answer_text, re.IGNORECASE)
            ans_map = {int(num): letter.upper() for num, letter in pairs}
            for q in questions:
                q_id = q['id']
                if q_id in ans_map:
                    q['correct'] = ans_map[q_id]
                    q['answer_found'] = True

    if not questions:
        return redirect(url_for('setup'))

    exam_id = str(uuid.uuid4())
    exam_store[exam_id] = {
        'questions': questions,
        'duration': duration,
        'session': exam_session,
        'mark_correct': mark_correct,
        'mark_incorrect': mark_incorrect,
        'candidate_name': candidate_name or 'Anonymous Candidate',
        'exam_topic': exam_topic or 'General Reading',
        'target_exam': target_exam
    }

    session['exam_id'] = exam_id
    return redirect(url_for('exam'))


@app.route('/exam')
def exam():
    """Serve live CBT workspace panel."""
    exam_id = session.get('exam_id')
    if not exam_id or exam_id not in exam_store:
        return redirect(url_for('setup'))

    data = exam_store[exam_id]
    
    # Render config settings, determining dynamic rules (A-C vs A-D choices)
    target_exam = data.get('target_exam', 'CFA')
    num_choices_allowed = 3 if target_exam == 'CFA' else 4
    
    return render_template('exam.html',
                           questions_json=json.dumps(data['questions']),
                           duration=data['duration'],
                           exam_session=data['session'],
                           mark_correct=data['mark_correct'],
                           mark_incorrect=data['mark_incorrect'],
                           total_questions=len(data['questions']),
                           candidate_name=data['candidate_name'],
                           exam_topic=data['exam_topic'],
                           num_choices_allowed=num_choices_allowed)


@app.route('/submit', methods=['POST'])
def submit():
    """Grade examination, update persistent stats database and error vaults."""
    exam_id = session.get('exam_id')
    if not exam_id or exam_id not in exam_store:
        return redirect(url_for('setup'))

    data = exam_store[exam_id]
    questions = data['questions']
    mark_correct = data['mark_correct']
    mark_incorrect = data['mark_incorrect']

    answers = json.loads(request.form.get('answers', '{}'))
    flags = json.loads(request.form.get('flags', '{}'))
    confidences = json.loads(request.form.get('confidences', '{}'))
    time_spent_raw = request.form.get('time_spent', '[]')
    
    try:
        time_spent = json.loads(time_spent_raw)
    except Exception:
        time_spent = []

    total = len(questions)
    if len(time_spent) < total:
        time_spent.extend([0] * (total - len(time_spent)))
    else:
        time_spent = time_spent[:total]

    attempted = 0
    correct_count = 0
    points = 0
    max_points = total * mark_correct

    topic_stats = {}
    correct_times = []
    incorrect_times = []
    
    # Surprise Metric counts
    overconfidence_count = 0
    lucky_guesses_count = 0
    time_traps_count = 0
    
    results = []
    failed_questions_list = []

    for idx, q in enumerate(questions):
        qid = str(q['id'])
        q_time = time_spent[idx]
        user_answer = answers.get(qid, None)
        is_attempted = user_answer is not None and user_answer != ''
        is_correct = user_answer == q['correct'] if is_attempted else False
        is_flagged = flags.get(qid, False)
        
        # Pull candidate confidence pill choice
        confidence = confidences.get(qid, 'High')  # High, Guess, Blind

        is_time_trap = (not is_correct) and (q_time > 120)

        if is_attempted:
            attempted += 1
            if is_correct:
                correct_count += 1
                points += mark_correct
                correct_times.append(q_time)
                if confidence == 'Blind':
                    lucky_guesses_count += 1
            else:
                points += mark_incorrect
                incorrect_times.append(q_time)
                failed_questions_list.append(q)
                if confidence == 'High':
                    overconfidence_count += 1
        else:
            points += mark_incorrect
            failed_questions_list.append(q)

        if is_time_trap:
            time_traps_count += 1

        topic = q.get('topic', 'General')
        if topic not in topic_stats:
            topic_stats[topic] = {'total': 0, 'attempted': 0, 'correct': 0}
        topic_stats[topic]['total'] += 1
        if is_attempted:
            topic_stats[topic]['attempted'] += 1
        if is_correct:
            topic_stats[topic]['correct'] += 1

        results.append({
            'id': q['id'],
            'text': q['text'],
            'choices': q['choices'],
            'correct': q['correct'],
            'user_answer': user_answer,
            'is_correct': is_correct,
            'is_attempted': is_attempted,
            'is_flagged': is_flagged,
            'topic': topic,
            'vignette': q.get('vignette'),
            'explanation': q.get('explanation'),
            'time_spent': q_time,
            'confidence': confidence,
            'is_time_trap': is_time_trap,
        })

    accuracy = (correct_count / attempted * 100) if attempted > 0 else 0
    score_pct = (points / max_points * 100) if max_points > 0 else 0

    avg_time_correct = round(sum(correct_times) / len(correct_times), 1) if correct_times else 0.0
    avg_time_incorrect = round(sum(incorrect_times) / len(incorrect_times), 1) if incorrect_times else 0.0

    # Mastery performance classification
    if score_pct >= 80:
        mastery_class = "Mastery Level"
        mastery_color = "var(--color-correct)"
    elif score_pct >= 70:
        mastery_class = "Borderline Pass"
        mastery_color = "var(--color-flagged)"
    else:
        mastery_class = "Requires Immediate Review"
        mastery_color = "var(--color-incorrect)"

    result_data = {
        'total': total,
        'attempted': attempted,
        'correct': correct_count,
        'points': points,
        'max_points': max_points,
        'accuracy': round(accuracy, 1),
        'score_pct': round(score_pct, 1),
        'exam_session': data['session'],
        'candidate_name': data.get('candidate_name', 'Anonymous Candidate'),
        'exam_topic': data.get('exam_topic', 'General Reading'),
        'topic_stats': topic_stats,
        'results': results,
        'time_spent_all': time_spent,
        'avg_time_correct': avg_time_correct,
        'avg_time_incorrect': avg_time_incorrect,
        'mastery_class': mastery_class,
        'mastery_color': mastery_color,
        'overconfidence_count': overconfidence_count,
        'lucky_guesses_count': lucky_guesses_count,
        'time_traps_count': time_traps_count
    }

    result_id = str(uuid.uuid4())
    exam_store[result_id] = result_data
    session['result_id'] = result_id

    # Append to local file database test_history.json
    history_path = os.path.join(os.path.dirname(__file__), 'test_history.json')
    history_db = {"history": [], "error_vault": []}
    
    if os.path.exists(history_path):
        try:
            with open(history_path, 'r', encoding='utf-8') as hf:
                loaded = json.load(hf)
                if isinstance(loaded, dict):
                    history_db["history"] = loaded.get("history", [])
                    history_db["error_vault"] = loaded.get("error_vault", [])
                elif isinstance(loaded, list):
                    # Handle legacy format where history was a flat list
                    history_db["history"] = loaded
        except Exception:
            pass

    # Save attempt
    new_record = {
        'id': result_id,
        'candidate_name': data.get('candidate_name', 'Anonymous Candidate'),
        'exam_topic': data.get('exam_topic', 'General Reading'),
        'timestamp': datetime.datetime.now().isoformat(),
        'total': total,
        'attempted': attempted,
        'correct': correct_count,
        'points': points,
        'max_points': max_points,
        'score_pct': round(score_pct, 1),
        'time_spent': time_spent,
        'avg_time_correct': avg_time_correct,
        'avg_time_incorrect': avg_time_incorrect,
        'mastery_class': mastery_class
    }
    history_db["history"].append(new_record)

    # Extract failed questions and save to mistake vault (preventing duplicate entries)
    existing_failed_texts = {fq['text'] for fq in history_db["error_vault"]}
    for fq in failed_questions_list:
        if fq['text'] not in existing_failed_texts:
            history_db["error_vault"].append(fq)

    try:
        with open(history_path, 'w', encoding='utf-8') as hf:
            json.dump(history_db, hf, indent=2)
    except Exception as e:
        print("Error writing database history:", e)

    return redirect(url_for('result'))


@app.route('/result')
def result():
    """Serve live grading scorecard panel."""
    result_id = session.get('result_id')
    if not result_id or result_id not in exam_store:
        return redirect(url_for('setup'))

    data = exam_store[result_id]
    return render_template('result.html', data_json=json.dumps(data))


@app.route('/dashboard')
def dashboard():
    """Serve performance analytics history."""
    history_path = os.path.join(os.path.dirname(__file__), 'test_history.json')
    history_db = {"history": [], "error_vault": []}
    
    if os.path.exists(history_path):
        try:
            with open(history_path, 'r', encoding='utf-8') as hf:
                loaded = json.load(hf)
                if isinstance(loaded, dict):
                    history_db = loaded
                elif isinstance(loaded, list):
                    history_db["history"] = loaded
        except Exception:
            pass

    history_list = history_db.get("history", [])
    total_runs = len(history_list)
    
    if total_runs > 0:
        avg_score = round(sum(r['score_pct'] for r in history_list) / total_runs, 1)
        total_attempted = sum(r.get('attempted', 0) for r in history_list)
        total_correct = sum(r.get('correct', 0) for r in history_list)
        avg_accuracy = round((total_correct / total_attempted * 100) if total_attempted > 0 else 0.0, 1)
    else:
        avg_score = 0.0
        avg_accuracy = 0.0

    return render_template('dashboard.html',
                           history_json=json.dumps(history_list),
                           total_runs=total_runs,
                           avg_score=avg_score,
                           avg_accuracy=avg_accuracy,
                           vault_count=len(history_db.get("error_vault", [])))


@app.route('/generate-error-quiz')
def generate_error_quiz():
    """Assemble a fresh review session from mistake logs."""
    history_path = os.path.join(os.path.dirname(__file__), 'test_history.json')
    error_vault = []
    
    if os.path.exists(history_path):
        try:
            with open(history_path, 'r', encoding='utf-8') as hf:
                loaded = json.load(hf)
                if isinstance(loaded, dict):
                    error_vault = loaded.get("error_vault", [])
        except Exception:
            pass

    if not error_vault:
        # Redirect back to setup displaying warning flag
        return redirect(url_for('setup', error='no_mistakes'))

    # Select up to 10 failed items randomly
    quiz_size = min(len(error_vault), 10)
    selected_questions = random.sample(error_vault, quiz_size)

    # Format question dictionary indexing keys sequentially
    questions = []
    for idx, sq in enumerate(selected_questions):
        questions.append({
            'id': idx + 1,
            'text': sq['text'],
            'choices': sq['choices'],
            'correct': sq['correct'],
            'topic': sq.get('topic', 'Error Review'),
            'vignette': sq.get('vignette'),
            'explanation': sq.get('explanation')
        })

    # Prepare temporary exam session
    exam_id = str(uuid.uuid4())
    exam_store[exam_id] = {
        'questions': questions,
        'duration': quiz_size * 2,  # 2 minutes per question
        'session': 'Review',
        'mark_correct': 3,
        'mark_incorrect': 0,
        'candidate_name': 'Vault Review Candidate',
        'exam_topic': 'Error Log Vault Quiz',
        'target_exam': 'Other (Custom Mock Test)'  # Allow dynamic options choice selection
    }

    session['exam_id'] = exam_id
    return redirect(url_for('exam'))


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
