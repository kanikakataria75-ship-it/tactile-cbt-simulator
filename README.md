# 📊 Multi-Exam CBT Simulator & Ingestor

A premium, widescreen glassmorphic Computer-Based Test (CBT) Simulator engine designed for high-stakes exams (CFA, FRM, JEE, NEET). It contains an advanced multimodal PDF ingestion pipeline that extracts, structure-tokenizes, and recovers diagrams/tables natively to reconstruct full exams.

---

## ✨ Key Features

### 1. 👁️ Multimodal Ingestion Pipeline
- **Dual-Engine Architecture:** Uses Gemini 2.0 Flash Vision to parse high-resolution page layout structures with a fast layout-aware text fallback parser.
- **Structural Option Splitting:** Instantly tokenizes horizontal/vertical choices (e.g. `A. 12V B. 24V C. 36V`) into clean option key-value mappings.
- **Scientific & Math Preservation:** Natively preserves thermodynamic, algebraic, and structural math characters (e.g. `π, Δ, Ω, α, β, γ, θ, ∞`) as native Unicode strings with strict HTML sub/sup rendering.

### 2. 📐 Spatial Layout & Recovery Engine
- **Header/Footer Bypass:** Standard margin filters are bypassed for large visual components (tables and diagrams), ensuring top-of-page and bottom-of-page visual vignettes are never skipped.
- **Cascading Multi-Resolution Search:** Uses PyMuPDF literal page lookup at multiple resolutions (first 35 chars, then 20 chars, then 12 chars, and first 3 words) to accurately assign vertical coordinates to questions.
- **Unreferenced Media Ingestion:** Maps extracted drawings and table crops to the vertically closest question on the same page based on spatial vertical flow.

### 3. 💻 Widescreen CBT Exam Interface
- **Premium Glassmorphic UI:** Smooth gradients, modern Outfit/Inter typography, and subtle micro-animations for an interactive testing experience.
- **Dynamic Question Pane:** Double-column layout separating vignettes/stems from options to mirror real professional testing environments.
- **Real-Time Analytics:** Time-on-question metrics, flags, instant grading scorecard, and review metrics.

---

## 🛠️ Tech Stack
- **Backend:** Python 3.11 / Flask / PyMuPDF (fitz) / pdfplumber
- **Frontend:** Vanilla HTML5 / ES6 JavaScript / Tailwind CSS (Glassmorphism & Interactive styles)
- **Data Persistence:** Local session-based state machines

---

## 🚀 Setup & Execution

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure API Key:**
   Create a `.env` file in the root directory:
   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

3. **Run the Server:**
   ```bash
   python app.py
   ```
   Open `http://127.0.0.1:5000` in your web browser.