"""
CBT Exam Simulator — Backend
Vision-First Multimodal Ingestion Pipeline (v5)
Powered by Gemini 2.0 Flash Vision + Pydantic Auto-Correction
"""

import os
import re
import io
import json
import uuid
import base64
import datetime
import traceback
from flask import Flask, render_template, request, redirect, url_for, session
from google import genai
from google.genai import types
from pydantic import BaseModel, Field, ConfigDict, TypeAdapter

# ── Load .env for GEMINI_API_KEY ──────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed; env vars must be set externally

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'cbt-simulator-secure-key-1892')
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024   # 50 MB

STATIC_MEDIA_DIR = os.path.join(app.root_path, 'static', 'extracted_media')
os.makedirs(STATIC_MEDIA_DIR, exist_ok=True)

try:
    import fitz          # PyMuPDF
    import pdfplumber
    ADVANCED_PARSING = True
except ImportError:
    ADVANCED_PARSING = False

class FileBasedStore:
    def __init__(self, directory):
        self.directory = directory
        os.makedirs(directory, exist_ok=True)
        self._in_memory = {}

    def __contains__(self, key):
        if not key:
            return False
        if key in self._in_memory:
            return True
        path = os.path.join(self.directory, f"store_{key}.json")
        return os.path.exists(path)

    def __getitem__(self, key):
        if not key:
            raise KeyError(key)
        if key in self._in_memory:
            return self._in_memory[key]
        path = os.path.join(self.directory, f"store_{key}.json")
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self._in_memory[key] = data
                    return data
            except Exception:
                pass
        raise KeyError(key)

    def __setitem__(self, key, value):
        if not key:
            return
        self._in_memory[key] = value
        path = os.path.join(self.directory, f"store_{key}.json")
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(value, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[STORE ERROR] Failed to write key {key}: {e}", flush=True)

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

exam_store = FileBasedStore(STATIC_MEDIA_DIR)

# Gemini client — singleton, reads GEMINI_API_KEY from environment
_gemini_client = None

def _get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        # Force load dotenv explicitly
        from dotenv import load_dotenv
        load_dotenv()
        
        api_key = os.environ.get('GEMINI_API_KEY')
        
        # Debugging: Terminal mein check kar ki key dikh rahi hai ya nahi
        print(f"DEBUG: Key found? {'YES' if api_key else 'NO'}")
        
        if not api_key:
            raise ValueError("GEMINI_API_KEY is missing! Check your .env file.")
            
        # Clean the key (remove extra quotes if user put them in .env)
        api_key = api_key.strip().strip('"').strip("'")
        
        _gemini_client = genai.Client(api_key=api_key)
    return _gemini_client

# =============================================================================
# PYDANTIC STRUCTURED OUTPUT SCHEMA
# =============================================================================
# Gemini returns diagram bounding boxes directly inside the schema.
# No mechanical coordinate proximity matching is used.

class ChoiceModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label: str
    text: str


class QuestionModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: int
    page_number: int = Field(description="0-indexed page number of the PDF where this question is located")
    text: str = Field(description="Full question stem. Include HTML tags for tables. Preserve math symbols as Unicode. Write [DIAGRAM_INJECT] in the text field where the image belongs.")
    choices: list[ChoiceModel] = Field(description="List of choices for multiple choice questions, e.g. [{'label': 'A', 'text': 'val'}, ...]. Empty list for integer-type.")
    correct: str = Field(description="Correct answer letter (A-E) or numeric string, or empty string")
    is_mcq: bool = Field(description="True when question has 2+ choices")
    explanation: str | None = None
    vignette: str | None = Field(default=None, description="Shared vignette passage, if any")
    diagram_bbox: list[float] = Field(default_factory=list, description="Exactly 4 floats [ymin, xmin, ymax, xmax] scaled proportionally from 0.0 to 1000.0 based on page dimensions, or [] if no visual")


# =============================================================================
# TXT / PIPE-DELIMITED PARSER  (unchanged - handles .txt uploads)
# =============================================================================

def parse_questions(file_content: str) -> list[dict]:
    """Parse pipe-delimited question file into structured data."""
    questions, current_topic, current_vignette = [], "General", None
    in_vignette, vignette_parts = False, []

    for line in file_content.strip().split('\n'):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if line.startswith('TOPIC:'):
            current_topic = line[6:].strip(); continue

        parts = [p.strip() for p in line.split('|')]

        if parts[0].upper() == 'VIGNETTE_START':
            in_vignette, vignette_parts = True, []
            for part in parts[1:]:
                if part.upper() == 'VIGNETTE_END':
                    in_vignette = False
                    current_vignette = ' '.join(vignette_parts).strip(); break
                vignette_parts.append(part)
            continue

        if in_vignette:
            if any(p.upper() == 'VIGNETTE_END' for p in parts):
                for part in parts:
                    if part.upper() == 'VIGNETTE_END': break
                    vignette_parts.append(part)
                in_vignette = False
                current_vignette = '\n'.join(vignette_parts).strip()
            else:
                vignette_parts.append(line)
            continue

        if len(parts) >= 5:
            letters = ['A','B','C','D','E','F']
            num_opts = len(parts) - 2
            choices  = {letters[i]: parts[i+1] for i in range(min(num_opts, len(letters)))}
            questions.append({
                'id': len(questions)+1, 'text': parts[0],
                'choices': choices, 'correct': parts[-1].strip().upper(),
                'topic': current_topic, 'vignette': current_vignette,
                'explanation': None, 'answer_found': True, 'q_type': 'mcq',
            })
    return questions


# =============================================================================
# VISION-FIRST MULTIMODAL INGESTION PIPELINE
# =============================================================================

# -- Stage 1: High-DPI Page Rendering -----------------------------------------

def _render_pages_to_images(doc) -> list[dict]:
    """
    Render each PDF page to a 300 DPI PNG image.
    Returns list of {page_num, image_bytes, width_pts, height_pts}.
    """
    DPI = 300
    matrix = fitz.Matrix(DPI / 72, DPI / 72)
    page_images = []

    for page_num in range(len(doc)):
        page = doc[page_num]

        pix = page.get_pixmap(matrix=matrix)
        img_bytes = pix.tobytes("png")

        page_images.append({
            'page_num': page_num,
            'image_bytes': img_bytes,
            'width_pts': page.rect.width,    # PDF point dimensions for cropping
            'height_pts': page.rect.height,
        })

    return page_images


# -- Stage 2: Gemini Vision Extraction ----------------------------------------

_VISION_SYSTEM_INSTRUCTION = """
You are an elite, universal document parsing engine. Your objective is to process high-resolution images of exam papers and extract the content into a strictly formatted JSON array.

CRITICAL RULES:
1. UNIVERSAL OPTION SPLITTING: Split inline/side-by-side options into separate objects in the `choices` array. Strip prefixes (A, B, 1, 2) from the text and put them in the `label`.
2. INDEPENDENT QUESTION ISOLATION: Treat every numbered item as an independent object. Do not merge distinct questions.
3. SPATIAL BOUNDING BOXES FOR VISUALS & TABLES (CRITICAL): If a question contains a diagram, graph, circuit, or any table (including financial statements, schedules, lists, or tables in image/drawn form), DO NOT transcribe it as plain text and DO NOT ignore its structure. Instead, either:
   a. Convert it into a clean, responsive HTML <table> with inline borders embedded in the question `text` or `vignette` field if it is a simple, easily readable table.
   b. If the table is complex, contains math/visuals, is scanned, or is in image form, treat it as a visual component: populate the `diagram_bbox` array with its exact relative coordinates [ymin, xmin, ymax, xmax] scaled proportionally from 0.0 to 1000.0 based on the page dimensions, and write [DIAGRAM_INJECT] in the question `text` or `vignette` field where the visual belongs.
4. MATH PRESERVATION: Preserve all math symbols as Unicode (π, Δ, Ω). Use HTML <sup> and <sub>. Never output □.
5. NOISE FILTERING: Ignore headers, footers, page numbers, and section titles like "SECTION-B".
"""


def _call_gemini_vision(page_images: list[dict], batch_start_id: int = 1) -> list[dict]:
    """
    Send page images to Gemini 2.0 Flash for structured extraction.
    Batches pages in groups of 10 to avoid token limits.
    """
    client = _get_gemini_client()
    all_questions = []

    # Generate and clean JSON schema for Gemini Developer API compatibility
    # (Strips 'additionalProperties' to prevent API 400 validation error)
    adapter = TypeAdapter(list[QuestionModel])
    schema_dict = adapter.json_schema()

    def _clean_schema_for_gemini(d):
        if not isinstance(d, dict):
            return
        d.pop('additionalProperties', None)
        d.pop('additional_properties', None)
        for k, v in list(d.items()):
            if isinstance(v, dict):
                _clean_schema_for_gemini(v)
            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, dict):
                        _clean_schema_for_gemini(item)

    _clean_schema_for_gemini(schema_dict)

    BATCH_SIZE = 10
    for batch_start in range(0, len(page_images), BATCH_SIZE):
        batch = page_images[batch_start:batch_start + BATCH_SIZE]

        # Build multimodal content parts
        content_parts = []
        page_desc_parts = []
        for pi in batch:
            content_parts.append(
                types.Part.from_bytes(
                    data=pi['image_bytes'],
                    mime_type='image/png',
                )
            )
            page_desc_parts.append(f"Page {pi['page_num']}")

        # Text instruction
        instruction_text = (
            f"Extract ALL exam questions from these {len(batch)} page images "
            f"(pages: {', '.join(page_desc_parts)}). "
            f"Start question numbering from {batch_start_id + len(all_questions)}. "
            f"For every diagram/figure/chart, return its precise bounding box coordinates "
            f"as [ymin, xmin, ymax, xmax] scaled 0-1000 in the diagram_bbox field. "
            f"Return structured JSON array."
        )
        content_parts.append(types.Part.from_text(text=instruction_text))

        try:
            response = client.models.generate_content(
                model='gemini-2.0-flash',
                contents=content_parts,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=schema_dict,
                    system_instruction=_VISION_SYSTEM_INSTRUCTION,
                    temperature=0.1,
                ),
            )

            batch_questions = json.loads(response.text)
            if isinstance(batch_questions, list):
                all_questions.extend(batch_questions)

        except Exception as e:
            print(f"[ERROR] Gemini vision batch failed (pages {batch_start}-{batch_start + len(batch)}): {e}")
            traceback.print_exc()

    return all_questions


def _inject_diagrams_from_bbox(questions: list[dict], page_images: list[dict]) -> None:
    """
    Loop through questions. If diagram_bbox contains exactly 4 floats,
    open the corresponding high-res PNG from page_images.
    Calculate the exact pixel crop based on 0-1000 scale coordinate values.
    Crop using PIL, save the cropped image to STATIC_MEDIA_DIR,
    and replace/append [DIAGRAM_INJECT] in the question stem with the correct HTML image tag.
    """
    from PIL import Image
    import io
    import uuid

    # Create mapping of page_num -> image_bytes for fast lookup
    page_img_map = {pi['page_num']: pi['image_bytes'] for pi in page_images}

    for q in questions:
        bbox = q.get('diagram_bbox')
        if isinstance(bbox, list) and len(bbox) == 4:
            ymin, xmin, ymax, xmax = bbox
            # Ensure coordinates are within [0, 1000]
            ymin = max(0.0, min(float(ymin), 1000.0))
            xmin = max(0.0, min(float(xmin), 1000.0))
            ymax = max(0.0, min(float(ymax), 1000.0))
            xmax = max(0.0, min(float(xmax), 1000.0))

            # Check if this is a non-empty bounding box
            if ymax <= ymin or xmax <= xmin:
                continue

            page_num = q.get('page_number', 0)
            if page_num not in page_img_map:
                print(f"[CROP WARNING] Page {page_num} not found in page_images map (skipped).")
                continue

            try:
                img_bytes = page_img_map[page_num]
                img = Image.open(io.BytesIO(img_bytes))
                width, height = img.size

                # Calculate coordinates scaled from 0-1000 to actual pixel size
                ymin_px = (ymin / 1000.0) * height
                xmin_px = (xmin / 1000.0) * width
                ymax_px = (ymax / 1000.0) * height
                xmax_px = (xmax / 1000.0) * width

                # Crop bounding box in PIL is (left, upper, right, lower) -> (xmin, ymin, xmax, ymax)
                crop_box = (xmin_px, ymin_px, xmax_px, ymax_px)
                cropped_img = img.crop(crop_box)

                # Save cropped image
                img_name = f"crop_{uuid.uuid4().hex}.png"
                img_path = os.path.join(STATIC_MEDIA_DIR, img_name)
                cropped_img.save(img_path)

                # HTML tag
                img_tag = (
                    f'<div class="diagram-container" style="margin:15px 0;text-align:left;">'
                    f'<img src="/static/extracted_media/{img_name}" '
                    f'style="max-width:100%;border-radius:8px;'
                    f'box-shadow:0 4px 12px rgba(0,0,0,0.18);" '
                    f'alt="Diagram" loading="lazy">'
                    f'</div>'
                )

                # Replace token if present, otherwise append
                text = q.get('text', '')
                vignette = q.get('vignette') or ''
                if '[DIAGRAM_INJECT]' in text:
                    q['text'] = text.replace('[DIAGRAM_INJECT]', img_tag)
                elif '[DIAGRAM_INJECT]' in vignette:
                    q['vignette'] = vignette.replace('[DIAGRAM_INJECT]', img_tag)
                else:
                    q['text'] = text + "\n" + img_tag

                print(f"[CROP] Cropped page {page_num} bbox {bbox} and saved as {img_name} for Q{q['id']}")

            except Exception as e:
                print(f"[CROP ERROR] Failed to crop diagram for Q{q['id']} on page {page_num}: {e}")
                traceback.print_exc()


# -- Stage 4: Pydantic Auto-Correction ----------------------------------------

# Symbol fix table — using escaped unicode to avoid syntax errors
_SYMBOL_FIXES = {
    '\u00ce\u00a9': '\u03a9',     # Omega
    '\u00cf\u0080': '\u03c0',     # pi
    '\u00ce\u00b1': '\u03b1',     # alpha
    '\u00ce\u00b2': '\u03b2',     # beta
    '\u00ce\u00b3': '\u03b3',     # gamma
    '\u00ce\u00b8': '\u03b8',     # theta
    '\u00ce\u00bb': '\u03bb',     # lambda
    '\u00ce\u00bc': '\u03bc',     # mu
    '\u00ce\u00b4': '\u03b4',     # delta
    '\u00cf\u0083': '\u03c3',     # sigma
    '\u00e2\u0089\u00a4': '\u2264',  # less-than-or-equal
    '\u00e2\u0089\u00a5': '\u2265',  # greater-than-or-equal
    '\u00e2\u0089\u0088': '\u2248',  # approximately-equal
    '\u00c2\u00b1': '\u00b1',     # plus-minus
    '\u00e2\u0088\u009a': '\u221a',  # square-root
    '\u00e2\u0088\u009e': '\u221e',  # infinity
    '\u00e2\u0088\u0091': '\u2211',  # summation
    '\u00e2\u0088\u00ab': '\u222b',  # integral
}


def _fix_symbols(text: str) -> str:
    """Fix common UTF-8 mojibake in scientific/math content."""
    if not text:
        return text
    for bad, good in _SYMBOL_FIXES.items():
        text = text.replace(bad, good)
    # Strip box placeholder characters
    text = text.replace('\u25a1', '')  # Box character
    return text.strip()


def _validate_and_correct(raw_questions: list[dict], global_answers: dict) -> list[dict]:
    """
    Run Pydantic validation and auto-correction on every question.
    Fixes: symbol corruption, duplicate IDs, missing answers, option fragmentation.
    """
    corrected = []
    seen_ids = set()
    next_id = 1

    for item in raw_questions:
        # Ensure unique sequential IDs
        q_id = item.get('id', next_id)
        if q_id in seen_ids:
            q_id = next_id
        while q_id in seen_ids:
            q_id += 1
        seen_ids.add(q_id)
        next_id = q_id + 1

        # Fix text
        q_text = _fix_symbols(item.get('text', '').strip())
        if not q_text:
            continue  # Skip empty questions

        # Fix choices (can be dict or list of ChoiceModels / dicts from Gemini)
        choices_input = item.get('choices', {})
        choices = {}
        if isinstance(choices_input, list):
            for opt in choices_input:
                if isinstance(opt, dict):
                    lbl = opt.get('label')
                    val = opt.get('text')
                    if lbl and val:
                        choices[str(lbl).strip().upper()] = _fix_symbols(str(val))
                elif hasattr(opt, 'label') and hasattr(opt, 'text'):
                    choices[str(opt.label).strip().upper()] = _fix_symbols(str(opt.text))
        elif isinstance(choices_input, dict):
            choices = {k.upper(): _fix_symbols(str(v)) for k, v in choices_input.items() if v}

        # Determine type
        is_mcq = len(choices) >= 2
        q_type = 'mcq' if is_mcq else 'integer'

        # Fix correct answer
        correct_val = str(item.get('correct', '')).strip().upper()
        if is_mcq and correct_val and correct_val not in choices:
            correct_val = ''  # Invalid answer letter

        # Fallback to global answer key
        if not correct_val and q_id in global_answers:
            correct_val = global_answers[q_id]

        # Default fallback
        if not correct_val and is_mcq:
            correct_val = 'A'

        # Fix explanation & vignette
        explanation = _fix_symbols(item.get('explanation') or '')
        vignette = _fix_symbols(item.get('vignette') or '')

        diagram_bbox = item.get('diagram_bbox')
        if not isinstance(diagram_bbox, list):
            diagram_bbox = []
        else:
            try:
                diagram_bbox = [float(x) for x in diagram_bbox]
            except (ValueError, TypeError):
                diagram_bbox = []

        corrected.append({
            'id': q_id,
            'page_number': int(item.get('page_number', 0)),
            'text': q_text,
            'original_stem': q_text,
            'choices': choices,
            'correct': correct_val,
            'topic': 'General',
            'vignette': vignette if vignette else None,
            'explanation': explanation if explanation else None,
            'answer_found': bool(correct_val),
            'q_type': q_type,
            'plain_text': re.sub(r'<[^>]+>', '', q_text),
            'media': {},
            'diagram_bbox': diagram_bbox,
        })

    return corrected


# -- Fallback: Text-Based Heuristic Parser ------------------------------------

def _format_text_table_as_html(text: str) -> str:
    """
    Detect text-based tables in raw text (aligned columns separated by multiple spaces or tabs)
    and format them as HTML tables.
    """
    if not text:
        return ""
    lines = text.split('\n')
    output_parts = []
    in_table = False
    table_rows = []
    
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if in_table:
                output_parts.append(_build_html_table(table_rows))
                in_table = False
            output_parts.append(line)
            continue
            
        # Split by 2 or more spaces, or tab
        cols = [c.strip() for c in re.split(r'\s{2,}|\t', stripped) if c.strip()]
        
        # If line has at least 2 columns and they are not just long sentences
        if len(cols) >= 2 and all(len(c) < 50 for c in cols):
            if not in_table:
                in_table = True
                table_rows = []
            table_rows.append(cols)
        else:
            if in_table:
                output_parts.append(_build_html_table(table_rows))
                in_table = False
            output_parts.append(line)
            
    if in_table:
        output_parts.append(_build_html_table(table_rows))
        
    return '\n'.join(output_parts)


def _build_html_table(rows: list[list[str]]) -> str:
    html = (
        '<div class="table-responsive" style="overflow-x:auto;margin:15px 0;">'
        '<table border="1" style="border-collapse:collapse;width:100%;max-width:600px;font-size:14px;'
        'background:rgba(255,255,255,0.05);">'
    )
    for r_idx, row in enumerate(rows):
        html += '<tr>'
        for cell in row:
            tag = 'th' if r_idx == 0 else 'td'
            bg = ' background:rgba(128,128,128,0.12);' if r_idx == 0 else ''
            html += f'<{tag} style="padding:8px 12px;border:1px solid rgba(128,128,128,0.3);{bg}">{cell}</{tag}>'
        html += '</tr>'
    html += '</table></div>'
    return html


def _fallback_text_parser(doc) -> list[dict]:
    """
    Legacy text-based extraction fallback.
    Used only when Gemini vision call fails completely.
    """
    page_offsets = []
    raw_text = ""
    for page_num in range(len(doc)):
        page = doc[page_num]
        page_text = page.get_text("text") or ""
        page_offsets.append((len(raw_text), len(raw_text) + len(page_text), page_num))
        raw_text += page_text + "\n"

    if not raw_text.strip():
        return []

    global_answers = _parse_global_answer_key(raw_text)

    q_starts = list(re.finditer(
        r'(?:(?:^|\n)\s*(?:Question|Q\.?|Qno\.?)\s*#?\s*[-:]?\s*(\d+)[.\-\):\s])'
        r'|(?:(?:^|\n)\s*(\d{1,3})\.\s+(?=[A-Z\w]))',
        raw_text, re.I
    ))

    if not q_starts:
        return []

    # Get preamble before Q1
    first_preamble = raw_text[:q_starts[0].start()].strip()
    first_preamble = _format_text_table_as_html(first_preamble)

    questions = []
    preambles = {} # q_idx -> preamble_text

    for idx, match in enumerate(q_starts):
        q_num = int(match.group(1) or match.group(2))
        start_pos = match.end()
        end_pos = q_starts[idx+1].start() if idx+1 < len(q_starts) else len(raw_text)

        # Track which page this question starts on
        start_char = match.end()
        q_page = 0
        for start_off, end_off, page_num in page_offsets:
            if start_off <= start_char <= end_off:
                q_page = page_num
                break

        block = raw_text[start_pos:end_pos]

        trunc = re.search(r'\b(?:Answer\s*Key|Solutions|Explanations)\b', block, re.I)
        if trunc:
            block = block[:trunc.start()]

        parsed = _parse_linear_block(block)
        q_id = idx + 1
        choices = parsed['choices']
        correct_val = parsed['correct']

        if not correct_val and q_num in global_answers:
            correct_val = global_answers[q_num]

        # Carry forward previous preamble if any
        q_text = parsed['stem']
        if q_id == 1 and first_preamble:
            q_text = first_preamble + "\n\n" + q_text
        elif q_id in preambles:
            q_text = preambles[q_id] + "\n\n" + q_text

        # Record this question's preamble for the next question
        if parsed['preamble'] and parsed['preamble'].strip():
            preambles[q_id + 1] = parsed['preamble']

        questions.append({
            'id': q_id,
            'page_number': q_page,
            'text': q_text,
            'original_stem': parsed['stem'],
            'choices': choices,
            'correct': correct_val or 'A',
            'topic': 'General',
            'vignette': None,
            'explanation': parsed['explanation'],
            'answer_found': bool(correct_val),
            'q_type': 'mcq' if len(choices) >= 2 else 'integer',
            'plain_text': re.sub(r'<[^>]+>', '', q_text),
            'media': {},
        })

        if trunc:
            break

    return questions


def _parse_linear_block(block_text: str) -> dict:
    """Parse a single question block into stem, choices, correct answer, and preamble."""
    opt_pattern = re.compile(
        r'(?:^|\n|>|<br>)\s*[\(\[]?([A-E])[\)\.\]]\s+(.+?)'
        r'(?=(?:^|\n|>|<br>)\s*[\(\[]?[A-E][\)\.\]]\s|^(?:Answer|Ans|Explanation|Correct|Rationale)|$)',
        re.MULTILINE | re.DOTALL | re.IGNORECASE
    )
    opt_matches = list(opt_pattern.finditer(block_text))
    choices = {m.group(1).upper(): m.group(2).strip().rstrip('.') for m in opt_matches}

    first_opt = opt_matches[0] if opt_matches else None
    stem = block_text[:first_opt.start()].strip() if first_opt else block_text.strip()

    # trailing text after options
    trailing = ""
    if opt_matches:
        last_opt_end = opt_matches[-1].end()
        trailing = block_text[last_opt_end:].strip()

    explanation = None
    em = re.search(r'(?:Explanation|Rationale)[:\s]*(.+)', block_text, re.I | re.DOTALL)
    if em:
        explanation = em.group(1).strip()
        # strip explanation from trailing text
        trailing = trailing.replace(em.group(0), "").strip()

    correct_answer = None
    for pat in [
        r'(?:Answer|Ans|Correct\s*Answer)[:\s]*([A-E])\b',
        r'(?:The\s+(?:correct|right)\s+answer\s+is)\s*[:\s]*([A-E])\b',
        r'\b([A-E])\b\s+is\s+(?:correct|the\s+correct\s+answer)\b',
    ]:
        m = re.search(pat, block_text, re.I)
        if m:
            correct_answer = m.group(1).upper()
            # strip answer pattern from trailing text
            trailing = re.sub(pat, "", trailing, flags=re.I).strip()
            break

    # Format table text in preamble and stem
    stem = _format_text_table_as_html(stem)
    preamble = _format_text_table_as_html(trailing)

    return {
        'stem': stem,
        'choices': choices,
        'correct': correct_answer,
        'explanation': explanation,
        'preamble': preamble
    }


def _parse_global_answer_key(full_text: str) -> dict:
    """Parse global answer key from full text content."""
    parts = re.split(
        r'(?:Answer\s*Key|Solutions|Appendix|Answers|Solutions\s*and\s*Explanations)',
        full_text, flags=re.I
    )
    if len(parts) < 2:
        return {}

    ans_block = "\n".join(parts[1:])
    ans_map = {}

    patterns = [
        re.compile(r'\b(?:Question|Q|Qno)?\s*(\d+)\s*[.:\-\s\)]+\s*[\(\[]?([A-E])[\)\]]?\b', re.I),
        re.compile(r'\b(?:Question|Q|Qno)?\s*(\d+)\s*[.:\-\s\)]+\s*([+-]?\d+(?:\.\d+)?)\b', re.I),
    ]

    for pat in patterns:
        for m in pat.finditer(ans_block):
            q_num = int(m.group(1))
            val = m.group(2).upper()
            if q_num not in ans_map:
                ans_map[q_num] = val

    return ans_map


# =============================================================================
# MAIN EXTRACTION ORCHESTRATOR
# =============================================================================

def extract_questions_vision(pdf_path: str) -> list[dict]:
    """
    Vision-First Multimodal Ingestion Pipeline.

    1. Render pages to 300 DPI PNG images
    2. Send images to Gemini 2.0 Flash for structured extraction & bounding boxes
    3. Validate and auto-correct with Pydantic
    4. Crop bounding boxes natively using Pillow and inject HTML tags
    """
    doc = fitz.open(pdf_path)

    # Parse global answer key from text layer (fallback source)
    raw_text = ""
    for page in doc:
        t = page.get_text("text")
        if t:
            raw_text += t + "\n"
    global_answers = _parse_global_answer_key(raw_text)

    # Stage 1: Render pages to high-DPI images for Gemini
    print(f"[PIPELINE] Rendering {len(doc)} pages at 300 DPI...")
    page_images = _render_pages_to_images(doc)
    print(f"[PIPELINE] {len(page_images)} pages rendered (blank pages skipped).")

    if not page_images:
        doc.close()
        return []

    # Stage 2: Gemini Vision structured extraction (for question text/options)
    print("[PIPELINE] Sending page images to Gemini 2.0 Flash...")
    raw_questions = _call_gemini_vision(page_images)
    print(f"[PIPELINE] Gemini returned {len(raw_questions)} questions.")

    # If Gemini failed, use fallback text parser
    if not raw_questions:
        print("[PIPELINE] Gemini returned no results. Falling back to text parser...")
        questions = _fallback_text_parser(doc)
    else:
        # Stage 3: Validate and auto-correct
        print("[PIPELINE] Running Pydantic validation and auto-correction...")
        questions = _validate_and_correct(raw_questions, global_answers)
        print(f"[PIPELINE] {len(questions)} questions validated.")

        # Stage 4: Process Pillow crops for diagram_bbox
        print("[PIPELINE] Processing Pillow crops for diagram bboxes...")
        _inject_diagrams_from_bbox(questions, page_images)

    # Clean up internal fields not needed by frontend
    for q in questions:
        q.pop('diagram_bbox', None)

    doc.close()
    return questions


def parse_pdf_text(pdf_bytes: bytes) -> list[dict]:
    """Entry point for PDF parsing. Routes to vision pipeline or simple fallback."""
    import sys
    print(f"[PARSE] Received {len(pdf_bytes)} bytes of PDF data", flush=True)
    print(f"[PARSE] ADVANCED_PARSING = {ADVANCED_PARSING}", flush=True)

    if not ADVANCED_PARSING:
        print("[PARSE] Using simple fallback (no PyMuPDF)", flush=True)
        # Simple fallback without PyMuPDF
        full_text = ""
        try:
            from pypdf import PdfReader
            for page in PdfReader(io.BytesIO(pdf_bytes)).pages:
                t = page.extract_text()
                if t:
                    full_text += t + "\n"
        except Exception as e:
            print(f"[PARSE] pypdf failed: {e}", flush=True)

        q_pattern = re.compile(
            r'(?:(?:^|\n|>|<br>)\s*(?:Question|Q\.?|Qno\.?)\s*#?\s*[-:]?\s*(\d+)[.\-\):\s])'
            r'|(?:(?:^|\n)\s*(\d{1,3})\.\s+(?=[A-Z\w]))',
            re.MULTILINE | re.IGNORECASE
        )
        q_matches = list(q_pattern.finditer(full_text))
        print(f"[PARSE] Simple parser found {len(q_matches)} question anchors", flush=True)
        if not q_matches:
            return []

        questions = []
        for idx, match in enumerate(q_matches):
            start = match.end()
            end = q_matches[idx+1].start() if idx+1 < len(q_matches) else len(full_text)
            block = full_text[start:end].strip()
            parsed = _parse_linear_block(block)
            questions.append({
                'id': idx + 1,
                'text': parsed['stem'],
                'choices': parsed['choices'],
                'correct': parsed['correct'] or 'A',
                'topic': 'General',
                'vignette': None,
                'explanation': parsed['explanation'],
                'answer_found': bool(parsed['correct']),
                'q_type': 'mcq' if len(parsed['choices']) >= 2 else 'integer',
                'plain_text': parsed['stem'],
                'media': {},
            })
        return questions

    # Write to temp file for PyMuPDF
    tmp = os.path.join(STATIC_MEDIA_DIR, f"tmp_{uuid.uuid4().hex}.pdf")
    print(f"[PARSE] Writing temp PDF to: {tmp}", flush=True)
    with open(tmp, 'wb') as f:
        f.write(pdf_bytes)

    try:
        print("[PARSE] Calling extract_questions_vision()...", flush=True)
        questions = extract_questions_vision(tmp)
        print(f"[PARSE] extract_questions_vision returned {len(questions)} questions", flush=True)
    except Exception as e:
        print(f"[ERROR] Vision pipeline failed: {e}", flush=True)
        traceback.print_exc()
        sys.stderr.flush()
        questions = []

    try:
        os.remove(tmp)
    except Exception:
        pass

    return questions


# =============================================================================
# FLASK ROUTES
# =============================================================================

@app.route('/')
def setup():
    return render_template('setup.html', error=request.args.get('error'))


@app.route('/upload', methods=['POST'])
def upload():
    duration       = int(request.form.get('duration', 135))
    exam_session   = request.form.get('session', 'AM')
    mark_correct   = int(request.form.get('mark_correct', 3))
    mark_incorrect = int(request.form.get('mark_incorrect', 0))
    candidate_name = request.form.get('candidate_name', 'Anonymous').strip()
    exam_topic     = request.form.get('exam_topic', 'General').strip()
    target_exam    = request.form.get('target_exam', 'CFA').strip()

    file = request.files.get('question_file')
    if not file or not file.filename:
        return redirect(url_for('setup'))

    raw_bytes = file.read()
    try:
        if file.filename.lower().endswith('.pdf'):
            questions = parse_pdf_text(raw_bytes)
        else:
            questions = parse_questions(raw_bytes.decode('utf-8', errors='ignore'))
        if not questions:
            return render_template('setup.html',
                error="No questions detected. The PDF format may not be supported.")
    except Exception as e:
        print(f"Upload error: {e}")
        traceback.print_exc()
        return render_template('setup.html', error="Parsing failed. Try a different PDF.")

    exam_id = str(uuid.uuid4())
    exam_store[exam_id] = {
        'questions': questions, 'duration': duration,
        'session': exam_session, 'mark_correct': mark_correct,
        'mark_incorrect': mark_incorrect, 'candidate_name': candidate_name,
        'exam_topic': exam_topic, 'target_exam': target_exam,
    }
    session['exam_id'] = exam_id
    return redirect(url_for('exam'))


@app.route('/exam')
def exam():
    exam_id = session.get('exam_id')
    if not exam_id or exam_id not in exam_store:
        return redirect(url_for('setup'))
    data = exam_store[exam_id]
    num_choices = 3 if data.get('target_exam', 'CFA') == 'CFA' else 4
    return render_template('exam.html',
        questions_json    = json.dumps(data['questions'], ensure_ascii=False),
        duration          = data['duration'],
        exam_session      = data['session'],
        mark_correct      = data['mark_correct'],
        mark_incorrect    = data['mark_incorrect'],
        total_questions   = len(data['questions']),
        candidate_name    = data['candidate_name'],
        exam_topic        = data['exam_topic'],
        num_choices_allowed = num_choices,
    )


@app.route('/submit', methods=['POST'])
def submit():
    exam_id = session.get('exam_id')
    if not exam_id or exam_id not in exam_store:
        return redirect(url_for('setup'))

    data = exam_store[exam_id]
    questions      = data['questions']
    mark_correct   = data['mark_correct']
    mark_incorrect = data['mark_incorrect']

    answers     = json.loads(request.form.get('answers',     '{}'))
    flags       = json.loads(request.form.get('flags',       '{}'))
    confidences = json.loads(request.form.get('confidences', '{}'))
    try:    time_spent = json.loads(request.form.get('time_spent', '[]'))
    except: time_spent = []

    total = len(questions)
    time_spent.extend([0] * max(0, total - len(time_spent)))

    attempted = correct_count = points = 0
    topic_stats, results, failed_qs = {}, [], []
    overconf = lucky = traps = 0

    for idx, q in enumerate(questions):
        qid         = str(q['id'])
        q_time      = time_spent[idx]
        user_ans    = answers.get(qid)
        is_attempted = bool(user_ans)
        confidence  = confidences.get(qid, 'High')
        q_type      = q.get('q_type', 'mcq')

        if q_type == 'integer':
            if is_attempted and q.get('correct') is not None:
                try:    is_correct = abs(float(user_ans) - float(q['correct'])) < 0.01
                except: is_correct = False
            else:
                is_correct = False
        else:
            is_correct = (user_ans == q['correct']) if is_attempted else False

        if is_attempted:
            attempted += 1
            if is_correct:
                correct_count += 1; points += mark_correct
                if confidence == 'Blind': lucky += 1
            else:
                points += mark_incorrect; failed_qs.append(q)
                if confidence == 'High': overconf += 1
        else:
            points += mark_incorrect; failed_qs.append(q)

        if not is_correct and q_time > 120: traps += 1

        topic = q.get('topic', 'General')
        ts = topic_stats.setdefault(topic, {'total':0,'attempted':0,'correct':0})
        ts['total'] += 1
        if is_attempted: ts['attempted'] += 1
        if is_correct:   ts['correct']   += 1

        results.append({
            'id': q['id'],
            'text': q.get('text', ''),
            'plain_text': q.get('plain_text', q['text']),
            'choices': q['choices'],
            'correct': q['correct'], 'user_answer': user_ans,
            'is_correct': is_correct, 'is_attempted': is_attempted,
            'is_flagged': flags.get(qid, False), 'topic': topic,
            'time_spent': q_time, 'confidence': confidence, 'q_type': q_type,
            'vignette': q.get('vignette'),
            'explanation': q.get('explanation'),
        })

    max_pts   = total * mark_correct
    score_pct = (points / max_pts * 100) if max_pts else 0

    correct_times = [r['time_spent'] for r in results if r['is_correct']]
    incorrect_times = [r['time_spent'] for r in results if not r['is_correct']]
    avg_correct = round(sum(correct_times) / len(correct_times), 1) if correct_times else 0.0
    avg_incorrect = round(sum(incorrect_times) / len(incorrect_times), 1) if incorrect_times else 0.0
    accuracy = round(correct_count / attempted * 100, 1) if attempted > 0 else 0.0

    result_data = {
        'total': total, 'attempted': attempted, 'correct': correct_count,
        'points': points, 'max_points': max_pts, 'score_pct': round(score_pct, 1),
        'accuracy': accuracy, 'avg_time_correct': avg_correct, 'avg_time_incorrect': avg_incorrect,
        'exam_session': data['session'], 'candidate_name': data['candidate_name'],
        'exam_topic': data['exam_topic'], 'topic_stats': topic_stats,
        'results': results, 'overconfidence_count': overconf,
        'lucky_guesses_count': lucky, 'time_traps_count': traps,
    }

    rid = str(uuid.uuid4())
    exam_store[rid] = result_data
    session['result_id'] = rid

    hpath = os.path.join(os.path.dirname(__file__), 'test_history.json')
    hdb   = {"history": [], "error_vault": []}
    if os.path.exists(hpath):
        try:
            with open(hpath, 'r', encoding='utf-8') as hf:
                loaded = json.load(hf)
                if isinstance(loaded, dict):
                    hdb = {"history": loaded.get("history",[]),
                           "error_vault": loaded.get("error_vault",[])}
        except Exception: pass

    hdb["history"].append({
        'id': rid, 'score_pct': round(score_pct,1),
        'timestamp': datetime.datetime.now().isoformat(),
        'attempted': attempted, 'correct': correct_count,
    })
    existing = {fq['text'] for fq in hdb["error_vault"]}
    for fq in failed_qs:
        if fq['text'] not in existing:
            hdb["error_vault"].append(fq)

    with open(hpath, 'w', encoding='utf-8') as hf:
        json.dump(hdb, hf, indent=2)

    return redirect(url_for('result'))


@app.route('/result')
def result():
    if session.get('result_id') not in exam_store:
        return redirect(url_for('setup'))
    return render_template('result.html',
        data_json=json.dumps(exam_store[session['result_id']], ensure_ascii=False))


@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)