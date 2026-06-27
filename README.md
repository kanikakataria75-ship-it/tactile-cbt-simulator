# 📊 Multi-Exam CBT Simulator & Ingestor

A premium, widescreen Spatial glassmorphic Computer-Based Test (CBT) Simulator engine designed for high-stakes examinations (CFA, FRM, ACCA, JEE, NEET). This platform integrates an advanced vision-first PDF ingestion pipeline with standard productivity widgets, custom rules engines, and interactive visual indicators.

---

## ✨ Key Features

### 1. 👁️ Vision-First Multimodal Ingestion
- **Gemini 2.0 Flash Vision:** Parses high-resolution page layout structures directly to extract questions, vignetted cases, and custom formatting.
- **Direct Bounding Box Crops:** Extracts diagram images, mathematical graphs, or complex drawings from coordinates returned by the vision model, saving and injecting them natively into test items.
- **Structural Option Splitting:** Automatically maps vertical/horizontal answer options (e.g., `A`, `B`, `C`) to structured key-value blocks.
- **Unicode & Math Preservation:** Natively maintains algebraic and thermodynamic math symbols (`π`, `Δ`, `Ω`, `α`, `β`, `γ`, `θ`, `∞`) with appropriate subscript/superscript markup.

### 2. 🌌 Spatial Glassmorphism & Visual Polish
- **Aurora Background Orbs:** Includes layered background mesh orbs moving under a gentle float animation.
- **Cursor Spotlights:** Follows cursor movements with hover radial-gradient spotlight reflections inside glass card cells.
- **Balanced Motion Stability:** Card panels, question lists, and option items utilize static soft border outlines and glows on hover, avoiding distracting position shifts or screen jitter.

### 3. 🛠️ Integrated Study Cockpit (Setup Page)
- **Interactive Tasks Workspace:** A native to-do list featuring task addition forms, inline title edits, checkbox toggles (with strike-through styling), and data persistence via browser `localStorage`.
- **Focus Timer Widget:** A study timer rendering an SVG countdown progress ring that supports start, pause, resume, and reset.
- **Heartbeat Focus Pulse:** Starts a slow breathing cycle (slowing aurora animations from 8s to 16s) when the timer is active to promote visual calm.
- **Header Ticker & Floating Dock:** A top status bar tracks live connection states and active elapsed time, while a bottom dock centers quick links.

### 4. 📝 CBT Exam Simulation & Scoring
- **Dynamic Rules Engine:** Adjusts marking structures and option choices based on exam boards (e.g., 3 options for CFA, 4 choices for FRM/ACCA).
- **Exam HUD & Review Panel:** Features a sticky header timer with warning indicators, question flag tags, status badges, and index lists.
- **Mistakes Vault & Analytics:** Automatically logs incorrect questions in a local database to review score trajectory curves and compile custom recovery quizzes.

---

## 🛠️ Tech Stack
- **Backend:** Python 3.11 / Flask / PyMuPDF (fitz) / Pillow / python-dotenv
- **Frontend:** Vanilla HTML5 / ES6 JavaScript / Font Awesome / Custom CSS (Liquid Glassmorphism)
- **Data Persistence:** Local session-based state and `localStorage`

---

## 🚀 Local Setup & Execution Guide

Follow these steps to set up and run the simulator locally on your PC:

### Prerequisites
Make sure you have **Python 3.11+** installed on your system. You will also need a **Gemini API Key** from Google AI Studio.

### 1. Clone the Repository
Open a terminal and navigate to your workspace directory:
```bash
git clone <repository-url>
cd cfa-cbt-simulator
```

### 2. Set Up a Virtual Environment (Recommended)
Creating a virtual environment ensures that the project dependencies do not interfere with other Python libraries:

**On Windows:**
```powershell
python -m venv .venv
.venv\Scripts\activate
```

**On macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
Install all required libraries using the pip package manager:
```bash
pip install -r requirements.txt
```

### 4. Configure API Credentials
Create a `.env` file in the project root directory:
```env
FLASK_SECRET_KEY=your-custom-secure-key-here
GEMINI_API_KEY=AIzaSy...your-actual-api-key-here
```
> **Note:** Ensure that there are no quotes or whitespaces around the API key values.

### 5. Launch the Server
Execute the Flask server:
```bash
python app.py
```

After starting, navigate to the local address in your web browser:
```text
http://127.0.0.1:5000
```
- **Upload Page:** Drop your practice PDF document into the Ingestion Desk, fill out your profile details, and click **START YOUR EXAM NOW**.
- **Analytics:** Access historical scoring charts via the bottom floating dock links.