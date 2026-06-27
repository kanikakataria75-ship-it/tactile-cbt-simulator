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
    label: str = Field(description="Choice label, e.g., A, B, C, D")
    text: str = Field(description="Choice text content")


class QuestionModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: int
    page_number: int = Field(description="0-indexed page number of the PDF where this question is located")
    text: str = Field(description="Full question stem. Include HTML tags for tables. Preserve math symbols as Unicode. Insert [[DIAGRAM:page_X_index_Y]] at the logical position where a diagram/figure appears.")
    choices: list[ChoiceModel] = Field(description="List of choices for multiple choice questions, e.g. [{'label': 'A', 'text': 'val'}, ...]. Empty list for integer-type.")
    correct: str = Field(description="Correct answer letter (A-E) or numeric string, or empty string")
    is_mcq: bool = Field(description="True when question has 2+ choices")
    explanation: str | None = None
    vignette: str | None = Field(default=None, description="Shared vignette passage, if any")
    has_diagram: bool = Field(default=False, description="True if this question contains or references a diagram/figure")


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
    Skips blank/cover pages automatically.
    """
    DPI = 300
    matrix = fitz.Matrix(DPI / 72, DPI / 72)
    page_images = []

    for page_num in range(len(doc)):
        page = doc[page_num]

        # Skip near-blank pages (covers, separators)
        text_len = len(page.get_text("text").strip())
        img_count = len(page.get_images())
        if text_len < 30 and img_count == 0:
            continue

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

_VISION_SYSTEM_INSTRUCTION = """You are an elite academic document parsing engine optimized for high-stakes professional exams (CFA, FRM, JEE, NEET). You are processing high-resolution 300 DPI page images. Your output must strictly be a raw JSON array of objects matching the schema.

STRICT INGESTION MANDATES:

1. POSITION-BASED OPTION SPLITTING:
   - Identify option layouts instantly. Even if options are printed horizontally side-by-side or in multi-column grids (e.g., "A. 12V  B. 24V  C. 36V"), you must forcefully split and tokenize them into clean, separate key-value pairs inside the JSON mapping: {"A": "12V", "B": "24V", ...}. NEVER concatenate multiple choices into a single string.

2. MATH & SCIENTIFIC SYMBOL PRESERVATION:
   - Preserve all mathematical, thermodynamic, engineering, and circuit operators natively as clean Unicode string representations (e.g., π, Δ, Ω, α, β, γ, θ, ∞, ±, √, ∑, ∫).
   - Use strict HTML <sup> and <sub> formatting for structural superscripts and subscripts. Never output empty box placeholders like '□'.

3. STRICT INLINE IMAGE & DIAGRAM ANCHORING (NO COORDINATES):
   - Do NOT guess or output pixel/percentage bounding box coordinates for diagrams, graphs, circuits, or figures. Coordinates cause cropping failures.
   - Instead, the absolute instant you detect a visual element (diagram, chart, graph, illustration) inside a question or vignette, you MUST insert a hard structural token string: `[[DIAGRAM:page_X_index_Y]]` directly into the text field at the exact logical position where the visual appears (where X is the 0-indexed page number and Y is the sequential index of the image on that specific page, starting from 1).
   - Mark `has_diagram = true` whenever you inject this token.

4. TEXT-BASED DATA GRIDS (TABLES):
   - If a visual block represents a structured text data grid or financial statement table, do NOT treat it as a diagram image. Re-render it completely as a standard, responsive HTML <table> (using <tr>, <th>, <td> tags with inline borders) embedded directly within the core question text string.

5. SEQUENTIAL FLOW SEGMENTATION:
   - Truncate parsing sequences instantly the moment solution anchors like "Answer Key", "Solutions", "Explanations", or "Answer Sheet" appear on the current workspace. Do not ingest past this boundary.

OUTPUT FORMAT: Return only a clean, structurally valid JSON array conforming strictly to the requested QuestionModel schema. Do not wrap in markdown blocks like ```json."""


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
            f"as percentages of page dimensions in the diagram_bbox field. "
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


def _merge_bboxes(rects: list, threshold: float = 15.0) -> list:
    """
    Merge rects that overlap or are very close vertically/horizontally.
    """
    if not rects:
        return []
    
    rects = sorted(rects, key=lambda r: r.y0)
    merged = []
    
    for r in rects:
        if not merged:
            merged.append(fitz.Rect(r))
            continue
        
        placed = False
        for idx, m in enumerate(merged):
            h_overlap = not (r.x1 < m.x0 - threshold or r.x0 > m.x1 + threshold)
            v_overlap = not (r.y1 < m.y0 - threshold or r.y0 > m.y1 + threshold)
            
            if h_overlap and v_overlap:
                merged[idx] = m | r  # Union
                placed = True
                break
        
        if not placed:
            merged.append(fitz.Rect(r))
            
    return merged


def _extract_all_page_media(doc) -> list[dict]:
    """
    Unconditionally extract all visual assets from every page in the PDF.
    Does NOT depend on Gemini output at all.
    Saves cropped PNG files as page_{page_num}_img_{index}.png (1-indexed).

    Returns list of media metadata:
      {'page': int, 'index': int, 'img_name': str}
    """
    HEADER_FRAC = 0.12
    FOOTER_FRAC = 0.90
    extracted_metadata = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        page_w = page.rect.width
        page_h = page.rect.height
        page_area = page_w * page_h
        HEADER_Y = page_h * HEADER_FRAC
        FOOTER_Y = page_h * FOOTER_FRAC

        visual_bboxes = []

        # Raster images
        try:
            for img_info in page.get_images():
                for r in page.get_image_rects(img_info[0]):
                    if (r.y0 < HEADER_Y or r.y1 > FOOTER_Y) and (r.height < 30 or r.width < 100):
                        continue
                    if r.width < 20 or r.height < 20:
                        continue
                    visual_bboxes.append(r)
        except Exception:
            pass

        # Image blocks from text dict
        try:
            blocks_dict = page.get_text('dict')['blocks']
            for b in blocks_dict:
                if b['type'] == 1:  # image block
                    r = fitz.Rect(b['bbox'])
                    if (r.y0 < HEADER_Y or r.y1 > FOOTER_Y) and (r.height < 30 or r.width < 100):
                        continue
                    if r.width < 20 or r.height < 20:
                        continue
                    visual_bboxes.append(r)
        except Exception:
            pass

        # Vector drawings (diagrams, circuits, graphs)
        try:
            for drw in page.get_drawings():
                r = drw['rect']
                if (r.y0 < HEADER_Y or r.y1 > FOOTER_Y) and (r.height < 30 or r.width < 100):
                    continue
                if r.width < 25 or r.height < 20:
                    continue
                if (r.width * r.height) >= (page_area * 0.7):
                    continue  # Skip full-page backgrounds
                visual_bboxes.append(r)
        except Exception:
            pass

        if not visual_bboxes:
            continue

        # Merge overlapping bounding boxes
        merged = _merge_bboxes(visual_bboxes)
        
        # Sort vertically to ensure stable 1-based indexing on the page
        merged = sorted(merged, key=lambda rect: rect.y0)

        # Crop and save each visual element
        for idx, rect in enumerate(merged):
            index = idx + 1
            clip = (rect + (-5, -5, 5, 5)).intersect(page.rect)
            if clip.is_empty or clip.width < 25 or clip.height < 20:
                continue

            try:
                pix = page.get_pixmap(clip=clip, dpi=150)
                img_name = f'page_{page_num}_img_{index}.png'
                img_path = os.path.join(STATIC_MEDIA_DIR, img_name)
                pix.save(img_path)
                
                extracted_metadata.append({
                    'page': page_num,
                    'index': index,
                    'img_name': img_name,
                    'bbox': (rect.x0, rect.y0, rect.x1, rect.y1)
                })
                print(f"[MEDIA] Saved visual: {img_name}", flush=True)
            except Exception as e:
                print(f"[MEDIA ERROR] Failed page {page_num} index {index}: {e}", flush=True)

    print(f'[MEDIA] Completed unconditional extraction. Saved {len(extracted_metadata)} page images.', flush=True)
    return extracted_metadata


def _replace_diagram_tokens(questions: list[dict]) -> None:
    """
    Search for [[DIAGRAM:page_X_index_Y]] inside question text and vignette,
    and replace them inline with a responsive HTML image tag pointing to the saved PNG.
    """
    import re
    pattern = re.compile(r'\[\[DIAGRAM:page_(\d+)_index_(\d+)\]\]', re.I)

    for q in questions:
        if 'media' not in q or not q['media']:
            q['media'] = {}

        # 1. Update text
        text = q.get('text', '')
        matches = pattern.findall(text)
        for match in matches:
            page_num = match[0]
            img_index = match[1]
            media_key = f"page_{page_num}_index_{img_index}"
            img_name = f"page_{page_num}_img_{img_index}.png"
            img_path = os.path.join(STATIC_MEDIA_DIR, img_name)

            if os.path.exists(img_path):
                html = (
                    f'<div class="diagram-container" style="margin:15px 0;text-align:left;">'
                    f'<img src="/static/extracted_media/{img_name}" '
                    f'style="max-width:100%;border-radius:8px;'
                    f'box-shadow:0 4px 12px rgba(0,0,0,0.18);" '
                    f'alt="Diagram" loading="lazy">'
                    f'</div>'
                )
                q['media'][media_key] = html
                text = re.sub(
                    rf'\[\[DIAGRAM:page_{page_num}_index_{img_index}\]\]',
                    html, text, flags=re.I
                )
        q['text'] = text

        # 2. Update vignette
        vig = q.get('vignette') or ''
        if vig:
            vig_matches = pattern.findall(vig)
            for match in vig_matches:
                page_num = match[0]
                img_index = match[1]
                media_key = f"page_{page_num}_index_{img_index}"
                img_name = f"page_{page_num}_img_{img_index}.png"
                img_path = os.path.join(STATIC_MEDIA_DIR, img_name)

                if os.path.exists(img_path):
                    html = (
                        f'<div class="diagram-container" style="margin:15px 0;text-align:left;">'
                        f'<img src="/static/extracted_media/{img_name}" '
                        f'style="max-width:100%;border-radius:8px;'
                        f'box-shadow:0 4px 12px rgba(0,0,0,0.18);" '
                        f'alt="Diagram" loading="lazy">'
                        f'</div>'
                    )
                    q['media'][media_key] = html
                    vig = re.sub(
                        rf'\[\[DIAGRAM:page_{page_num}_index_{img_index}\]\]',
                        html, vig, flags=re.I
                    )
            q['vignette'] = vig


def _find_question_y_position(page, stem: str) -> float | None:
    """
    Search for a question stem on a page at multiple resolutions
    and return the vertical y0 position of the first match.
    """
    if not stem:
        return None
    # Strip HTML tags
    clean = re.sub(r'<[^>]+>', '', stem)
    # Normalize spaces
    clean = re.sub(r'\s+', ' ', clean).strip()
    
    # Try different substring lengths directly on the normalized text
    for length in [35, 20, 12]:
        term = clean[:length].strip()
        if len(term) >= 6:
            rects = page.search_for(term)
            if rects:
                return rects[0].y0
                
    # Fallback: try first 3 words
    words = [w.strip() for w in clean.split() if w.strip()]
    if len(words) >= 2:
        term = ' '.join(words[:3])
        rects = page.search_for(term)
        if rects:
            return rects[0].y0
            
    return None


def _inject_pdf_tables(doc, questions: list[dict]) -> None:
    """
    Extract digital tables via pdfplumber from each page,
    locate their corresponding questions on that page by vertical position,
    and inject the formatted HTML table into the question stem.
    """
    try:
        import pdfplumber
        pl_pdf = pdfplumber.open(doc.name)
    except Exception as e:
        print(f"[TABLE INJECTION ERROR] Failed to open pdfplumber: {e}", flush=True)
        return

    # Group questions by page_number
    page_to_qs = {}
    for q in questions:
        p_num = q.get('page_number', 0)
        page_to_qs.setdefault(p_num, []).append(q)

    HEADER_FRAC = 0.12
    FOOTER_FRAC = 0.90

    for page_num, qs in page_to_qs.items():
        if page_num >= len(pl_pdf.pages):
            continue

        page = doc[page_num]
        page_h = page.rect.height
        HEADER_Y = page_h * HEADER_FRAC
        FOOTER_Y = page_h * FOOTER_FRAC

        try:
            pl_page = pl_pdf.pages[page_num]
            tables = pl_page.find_tables()
            if not tables:
                continue

            print(f"[TABLE INJECT] Found {len(tables)} tables on page {page_num}", flush=True)

            # 1. Determine vertical positions of questions on this page
            q_coords = []
            for q in qs:
                y_pos = _find_question_y_position(page, q.get('original_stem', q.get('text', '')))
                if y_pos is None:
                    q_coords.append((q, -1))
                else:
                    q_coords.append((q, y_pos))

            # Distribute estimated ones evenly based on order if search failed
            none_count = sum(1 for item in q_coords if item[1] == -1)
            if none_count > 0:
                step = (FOOTER_Y - HEADER_Y) / (len(qs) + 1)
                curr_y = HEADER_Y + step
                new_q_coords = []
                for q, y in q_coords:
                    if y == -1:
                        new_q_coords.append((q, curr_y))
                        curr_y += step
                    else:
                        new_q_coords.append((q, y))
                q_coords = new_q_coords

            # Sort questions vertically
            q_coords.sort(key=lambda item: item[1])

            # 2. Match each table to the nearest question
            for tbl in tables:
                bx = tbl.bbox # (x0, y0, x1, y1) in points
                if bx[1] < HEADER_Y or bx[3] > FOOTER_Y:
                    continue # Skip headers and footers
                
                extracted = tbl.extract()
                if not extracted or not any(any(cell for cell in row) for row in extracted):
                    continue

                # Format the table as HTML
                html = (
                    '<div class="table-responsive" style="overflow-x:auto;margin:15px 0;">'
                    '<table border="1" style="border-collapse:collapse;width:100%;font-size:14px;'
                    'background:rgba(255,255,255,0.05);">'
                )
                for r_idx, row in enumerate(extracted):
                    html += '<tr>'
                    for cell in row:
                        ct = str(cell).replace('\n', '<br>') if cell else ''
                        tag = 'th' if r_idx == 0 else 'td'
                        bg = ' background:rgba(128,128,128,0.12);' if r_idx == 0 else ''
                        html += f'<{tag} style="padding:9px 12px;border:1px solid rgba(128,128,128,0.3);{bg}">{ct}</{tag}>'
                    html += '</tr>'
                html += '</table></div>'

                # Table y center coordinate
                tbl_y = (bx[1] + bx[3]) / 2

                # Find the question closest to tbl_y on the page
                best_q = None
                best_dist = float('inf')
                best_q_y = 0
                for q, y in q_coords:
                    dist = abs(y - tbl_y)
                    if dist < best_dist:
                        best_dist = dist
                        best_q = q
                        best_q_y = y

                if best_q:
                    current_text = best_q.get('text', '')
                    
                    # Avoid duplicate table injection (if Gemini or parser already did it)
                    cell_sample = ""
                    for row in extracted[:2]:
                        for cell in row:
                            if cell and len(str(cell)) > 3:
                                cell_sample = str(cell).strip()
                                break
                        if cell_sample:
                            break
                            
                    if cell_sample and cell_sample in current_text:
                        print(f"[TABLE INJECT] Table sample '{cell_sample}' already in Q{best_q['id']}, skipping", flush=True)
                        continue

                    if tbl_y < best_q_y:
                        # Table sits above the question
                        best_q['text'] = html + "\n" + current_text
                    else:
                        best_q['text'] = current_text + "\n" + html
                    print(f"[TABLE INJECT] Injected table on page {page_num} into Q{best_q['id']}", flush=True)

        except Exception as ex:
            print(f"[TABLE INJECT ERROR] Failed page {page_num}: {ex}", flush=True)

    try:
        pl_pdf.close()
    except Exception:
        pass


def _inject_unreferenced_media(doc, questions: list[dict], media_metadata: list[dict]) -> None:
    """
    Identifies visual media (diagrams, image tables, charts) that were extracted
    but are not referenced anywhere in the question text or vignettes.
    Maps them to the vertically nearest question on the same page and prepends/appends them.
    """
    # Group media by page
    page_to_media = {}
    for item in media_metadata:
        page_to_media.setdefault(item['page'], []).append(item)

    # Group questions by page
    page_to_qs = {}
    for q in questions:
        page_to_qs.setdefault(q.get('page_number', 0), []).append(q)

    HEADER_FRAC = 0.12
    FOOTER_FRAC = 0.90

    for page_num, media_list in page_to_media.items():
        qs = page_to_qs.get(page_num, [])
        if not qs:
            continue

        page = doc[page_num]
        page_h = page.rect.height
        HEADER_Y = page_h * HEADER_FRAC
        FOOTER_Y = page_h * FOOTER_FRAC

        # Determine vertical coordinates of questions on this page
        q_coords = []
        for q in qs:
            y_pos = _find_question_y_position(page, q.get('original_stem', q.get('text', '')))
            if y_pos is None:
                q_coords.append((q, -1))
            else:
                q_coords.append((q, y_pos))

        # Distribute estimated ones evenly based on order if search failed
        none_count = sum(1 for item in q_coords if item[1] == -1)
        if none_count > 0:
            step = (FOOTER_Y - HEADER_Y) / (len(qs) + 1)
            curr_y = HEADER_Y + step
            new_q_coords = []
            for q, y in q_coords:
                if y == -1:
                    new_q_coords.append((q, curr_y))
                    curr_y += step
                else:
                    new_q_coords.append((q, y))
            q_coords = new_q_coords

        # Sort questions vertically
        q_coords.sort(key=lambda item: item[1])

        for media in media_list:
            img_name = media['img_name']
            
            # Check if this image name or token is already referenced in any question on this page
            referenced = False
            token_sig_1 = f"page_{media['page']}_img_{media['index']}.png"
            token_sig_2 = f"page_{media['page']}_index_{media['index']}"
            for q in qs:
                q_text_all = (q.get('text', '') + ' ' + (q.get('vignette') or '')).lower()
                if token_sig_1.lower() in q_text_all or token_sig_2.lower() in q_text_all:
                    referenced = True
                    break
            
            if referenced:
                continue

            # It's unreferenced! Map it to the closest question vertically.
            bbox = media['bbox']
            media_y = (bbox[1] + bbox[3]) / 2

            best_q = None
            best_dist = float('inf')
            best_q_y = 0
            for q, y in q_coords:
                dist = abs(y - media_y)
                if dist < best_dist:
                    best_dist = dist
                    best_q = q
                    best_q_y = y

            if best_q:
                # Build HTML image tag
                html = (
                    f'<div class="diagram-container" style="margin:15px 0;text-align:left;">'
                    f'<img src="/static/extracted_media/{img_name}" '
                    f'style="max-width:100%;border-radius:8px;'
                    f'box-shadow:0 4px 12px rgba(0,0,0,0.18);" '
                    f'alt="Diagram" loading="lazy">'
                    f'</div>'
                )
                
                current_text = best_q.get('text', '')
                if media_y < best_q_y:
                    # Prepend if visual block is above question text
                    best_q['text'] = html + "\n" + current_text
                else:
                    # Append if visual block is below question text
                    best_q['text'] = current_text + "\n" + html
                print(f"[MEDIA INJECT] Injected unreferenced visual '{img_name}' on page {page_num} into Q{best_q['id']}", flush=True)


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
            'has_diagram': item.get('has_diagram', False),
            'diagram_bbox': item.get('diagram_bbox'),
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

    1. UNCONDITIONALLY extract and save all visuals as page_X_img_Y.png
    2. Render pages to 300 DPI images for Gemini
    3. Send images to Gemini 2.0 Flash for question text structuring and token injection
    4. Validate and auto-correct with Pydantic
    5. Resolve inline diagram tokens [[DIAGRAM:page_X_index_Y]] on the backend
    """
    doc = fitz.open(pdf_path)

    # Parse global answer key from text layer (fallback source)
    raw_text = ""
    for page in doc:
        t = page.get_text("text")
        if t:
            raw_text += t + "\n"
    global_answers = _parse_global_answer_key(raw_text)

    # Stage 1: UNCONDITIONALLY extract ALL visual media from every page
    print("[PIPELINE] Extracting all images, diagrams, and tables from PDF...")
    media_metadata = _extract_all_page_media(doc)

    # Stage 2: Render pages to high-DPI images for Gemini
    print(f"[PIPELINE] Rendering {len(doc)} pages at 300 DPI...")
    page_images = _render_pages_to_images(doc)
    print(f"[PIPELINE] {len(page_images)} pages rendered (blank pages skipped).")

    if not page_images:
        doc.close()
        return []

    # Stage 3: Gemini Vision structured extraction (for question text/options)
    print("[PIPELINE] Sending page images to Gemini 2.0 Flash...")
    raw_questions = _call_gemini_vision(page_images)
    print(f"[PIPELINE] Gemini returned {len(raw_questions)} questions.")

    # If Gemini failed, use fallback text parser
    if not raw_questions:
        print("[PIPELINE] Gemini returned no results. Falling back to text parser...")
        questions = _fallback_text_parser(doc)
    else:
        # Stage 4: Validate and auto-correct
        print("[PIPELINE] Running Pydantic validation and auto-correction...")
        questions = _validate_and_correct(raw_questions, global_answers)
        print(f"[PIPELINE] {len(questions)} questions validated.")

    # Stage 5: Resolve inline [[DIAGRAM:page_X_index_Y]] tokens
    print("[PIPELINE] Resolving diagram tokens...")
    _replace_diagram_tokens(questions)
    print("[PIPELINE] Diagram token resolution complete.")

    # Stage 6: Inject PDF tables (ALWAYS runs as layout-aware safety layer)
    print("[PIPELINE] Injecting text-based PDF tables...")
    _inject_pdf_tables(doc, questions)
    print("[PIPELINE] PDF table injection complete.")

    # Stage 7: Inject unreferenced visual media (layouts/diagrams/image-tables)
    print("[PIPELINE] Injecting unreferenced visual media...")
    _inject_unreferenced_media(doc, questions, media_metadata)
    print("[PIPELINE] Unreferenced visual media injection complete.")

    # Clean up internal fields not needed by frontend
    for q in questions:
        q.pop('has_diagram', None)
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