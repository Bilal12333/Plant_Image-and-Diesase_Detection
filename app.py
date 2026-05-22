import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from groq import Groq
import io
import tempfile
import os
import base64
import json
import datetime
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(    page_title="PlantAI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>
.stApp { background-color: #0f172a; color: #f8fafc; }
.block-container { padding-top: 2rem; max-width: 1200px; }
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
[data-testid="stFileUploader"] {
    background: #111827;
    border: 2px dashed #334155;
    border-radius: 18px;
    padding: 20px;
}
.card {
    background: #111827;
    border-radius: 20px;
    padding: 24px;
    margin-bottom: 20px;
    border: 1px solid rgba(255,255,255,0.05);
    box-shadow: 0 8px 20px rgba(0,0,0,0.25);
}
.badge {
    display:inline-block;
    padding: 6px 14px;
    border-radius:999px;
    background: rgba(34,197,94,0.15);
    color:#4ade80;
    font-weight:600;
    font-size:14px;
}
.warning-card {
    background: #1c1107;
    border-radius: 20px;
    padding: 24px;
    margin-bottom: 20px;
    border: 1px solid rgba(245,158,11,0.3);
    box-shadow: 0 8px 20px rgba(0,0,0,0.25);
}
.error-card {
    background: #1c0a0a;
    border-radius: 20px;
    padding: 24px;
    margin-bottom: 20px;
    border: 1px solid rgba(239,68,68,0.3);
    box-shadow: 0 8px 20px rgba(0,0,0,0.25);
}
.info-card {
    background: #0c1a2e;
    border-radius: 20px;
    padding: 24px;
    margin-bottom: 20px;
    border: 1px solid rgba(59,130,246,0.3);
    box-shadow: 0 8px 20px rgba(0,0,0,0.25);
}
.stButton > button {
    background: linear-gradient(135deg, #22c55e, #16a34a);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.75rem 1.5rem;
    font-weight: 600;
}
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #22c55e, #3b82f6);
}
h1, h2, h3 { color: white; }
p { color: #e2e8f0; }
</style>
""", unsafe_allow_html=True)

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.markdown("## 🌿 PlantAI")
    st.markdown("AI-powered plant disease detection system.")
    st.divider()
    st.markdown("### Supported Plants")
    st.markdown("- 🍅 Tomato\n- 🥔 Potato\n- 🫑 Pepper Bell")
    st.divider()
    st.markdown("### Validation Pipeline")
    st.markdown("1. 🔵 Blur check\n2. 🟢 Green content\n3. 👁️ Vision plant ID\n4. 🤖 Disease model\n5. 🧠 AI analysis")

# =========================================================
# HEADER
# =========================================================

st.markdown(
    "<div style='padding-bottom:20px;'>"
    "<h1 style='font-size:3rem; margin-bottom:0;'>🌿 PlantAI</h1>"
    "<p style='color:#94a3b8; font-size:1.1rem;'>Deep Learning Plant Disease Detection System</p>"
    "</div>",
    unsafe_allow_html=True
)

# =========================================================
# GROQ CLIENT
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("plant_disease_model_v3.keras")

model = load_model()

# =========================================================
# CLASS NAMES
# =========================================================

CLASS_NAMES = [
    "Pepper__bell___Bacterial_spot",                  # 0
    "Pepper__bell___healthy",                          # 1
    "Potato___Early_blight",                           # 2
    "Potato___Late_blight",                            # 3
    "Potato___healthy",                                # 4
    "Tomato_Bacterial_spot",                           # 5
    "Tomato_Early_blight",                             # 6
    "Tomato_Late_blight",                              # 7
    "Tomato_Leaf_Mold",                                # 8
    "Tomato_Septoria_leaf_spot",                       # 9
    "Tomato_Spider_mites_Two_spotted_spider_mite",     # 10
    "Tomato__Target_Spot",                             # 11
    "Tomato__Tomato_YellowLeaf__Curl_Virus",           # 12
    "Tomato__Tomato_mosaic_virus",                     # 13
    "Tomato_healthy"                                   # 14
]

# Strict per-plant allowed class indices
PLANT_CLASS_MAP = {
    "tomato": [5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
    "potato": [2, 3, 4],
    "pepper": [0, 1],
}

IMG_SIZE = (224, 224)

# =========================================================
# HELPERS
# =========================================================

def clean_class_name(name):
    return name.replace("___", " - ").replace("__", " ").replace("_", " ")

def image_to_base64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=85)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

# =========================================================
# PREPROCESSING
# =========================================================

def preprocess_image(image):
    image = np.array(image)
    image = cv2.resize(image, IMG_SIZE)
    image = image.astype(np.float32) / 255.0
    mean  = np.array([0.485, 0.456, 0.406])
    std   = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std
    return np.expand_dims(image, axis=0)

# =========================================================
# BASIC CHECKS
# =========================================================

def is_blurry(image, threshold=80):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < threshold

def has_enough_green(image, green_threshold=0.08):
    img     = np.array(image)
    r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
    return (np.sum((g > r) & (g > b)) / img[:,:,0].size) > green_threshold

# =========================================================
# GROQ VISION — HARDENED PLANT VALIDATION
# =========================================================

def validate_plant_with_vision(image: Image.Image):
    """
    Two-pass vision validation:

    Pass 1 — Ask the model to identify the plant and whether it is supported.
    Pass 2 — If Pass 1 returns plant_type='other' but is_supported=True
             (contradictory), re-ask with a stricter prompt to resolve.

    Returns:
        is_valid       bool
        plant_type     str   'tomato' | 'potato' | 'pepper' | 'other'
        detected_plant str
        reason         str
        vision_error   str   non-empty on API failure

    FAIL-CLOSED: any exception → is_valid=False
    """
    b64 = image_to_base64(image)

    # ── Pass 1 ──────────────────────────────────────────
    prompt_1 = """You are a strict botanical expert specialising in crop leaves.

Examine the image carefully. Reply ONLY with valid JSON — no markdown, no extra text.

{
  "is_supported_plant": true or false,
  "plant_type": "tomato" | "potato" | "pepper" | "other",
  "detected_plant": "<specific common name of what you see>",
  "reason": "<one sentence>"
}

Critical rules:
- is_supported_plant = true ONLY for a LEAF of Tomato, Potato, or Pepper Bell.
- plant_type MUST match: tomato/potato/pepper when is_supported_plant=true, else "other".
- Tomato and Potato leaves look very similar (both Solanaceae) — distinguish carefully:
    Tomato leaf: pinnately compound with MANY small leaflets, strongly aromatic scent cue
    Potato leaf: also compound but leaflets are rounder and more uniform in size
- A healthy leaf with NO disease is still valid — do NOT require visible disease.
- Green colour alone does NOT qualify — identify the actual species.
- false for: fruits, vegetables, flowers, trees, animals, objects, non-leaf images."""

    try:
        def call_vision(prompt_text):
            return client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                        {"type": "text", "text": prompt_text}
                    ]
                }],
                max_tokens=250,
                temperature=0.0
            )

        def parse_response(raw):
            raw = raw.strip()
            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.lower().startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()
            return json.loads(raw)

        # Pass 1
        r1   = call_vision(prompt_1)
        raw1 = r1.choices[0].message.content
        d1   = parse_response(raw1)

        is_valid       = bool(d1.get("is_supported_plant", False))
        plant_type     = str(d1.get("plant_type", "other")).lower().strip()
        detected_plant = str(d1.get("detected_plant", "Unknown"))
        reason         = str(d1.get("reason", ""))

        # ── Consistency check ────────────────────────────
        # If vision says supported=True but plant_type='other', or
        # says supported=False but plant_type is a known plant → re-ask
        contradiction = (
            (is_valid and plant_type not in PLANT_CLASS_MAP) or
            (not is_valid and plant_type in PLANT_CLASS_MAP)
        )

        if contradiction:
            prompt_2 = f"""You previously analysed this leaf image and gave a contradictory answer:
is_supported_plant={is_valid}, plant_type="{plant_type}"

These are inconsistent. Please re-examine carefully and answer again.
Reply ONLY with valid JSON:

{{
  "is_supported_plant": true or false,
  "plant_type": "tomato" | "potato" | "pepper" | "other",
  "detected_plant": "<specific name>",
  "reason": "<one sentence>"
}}

Remember: plant_type must be tomato/potato/pepper when is_supported_plant=true, and "other" when false."""

            r2   = call_vision(prompt_2)
            raw2 = r2.choices[0].message.content
            d2   = parse_response(raw2)

            is_valid       = bool(d2.get("is_supported_plant", False))
            plant_type     = str(d2.get("plant_type", "other")).lower().strip()
            detected_plant = str(d2.get("detected_plant", detected_plant))
            reason         = str(d2.get("reason", reason))

        # Final safety: if is_valid=True but plant_type still not in map, force-invalid
        if is_valid and plant_type not in PLANT_CLASS_MAP:
            is_valid   = False
            plant_type = "other"
            reason     = "Plant type could not be confirmed as Tomato, Potato, or Pepper Bell."

        return is_valid, plant_type, detected_plant, reason, ""

    except Exception as e:
        return False, "unknown", "Unknown", "Vision validation could not be completed.", str(e)

# =========================================================
# PLANT-AWARE PREDICTION  (STRICT — no fallback to full probs)
# =========================================================

def predict_with_plant_filter(probs, plant_type):
    """
    Restricts softmax scores to ONLY the confirmed plant's classes.
    This eliminates cross-family errors like:
      healthy Tomato leaf → Potato Late Blight

    Raises ValueError if plant_type is not in PLANT_CLASS_MAP,
    so the caller must always pass a validated plant_type.
    """
    allowed = PLANT_CLASS_MAP.get(plant_type)
    if not allowed:
        raise ValueError(f"Unknown plant_type '{plant_type}' passed to predict_with_plant_filter")

    # Mask: keep only allowed classes
    masked = np.zeros_like(probs)
    for i in allowed:
        masked[i] = probs[i]

    # Re-normalise
    total = masked.sum()
    if total > 0:
        masked = masked / total
    else:
        # Fallback: uniform over allowed classes (shouldn't happen in practice)
        for i in allowed:
            masked[i] = 1.0 / len(allowed)

    top_idx   = int(np.argmax(masked))
    top_class = CLASS_NAMES[top_idx]
    top_prob  = float(masked[top_idx])

    top3_idx = np.argsort(masked)[-3:][::-1]
    top3     = [(clean_class_name(CLASS_NAMES[i]), float(masked[i]))
                for i in top3_idx if masked[i] > 0]

    return clean_class_name(top_class), top_prob, top3

# =========================================================
# AI DISEASE ANALYSIS
# =========================================================

def get_ai_analysis(disease_name):
    prompt = f"""
The detected plant disease is: {disease_name}

Please explain:
1. What this disease is
2. Main causes
3. Symptoms
4. Treatment methods
5. Prevention tips

Keep the explanation beginner-friendly.
"""
    try:
        cc = client.chat.completions.create(
            messages=[
                {"role": "system",
                 "content": ("You are PlantAI, a professional plant disease expert. "
                             "Explain diseases clearly. Give practical advice.")},
                {"role": "user", "content": prompt}
            ],
            model="llama-3.1-8b-instant"
        )
        return cc.choices[0].message.content
    except Exception as e:
        return f"AI analysis failed: {str(e)}"

# =========================================================
# PDF REPORT
# =========================================================

def generate_pdf_report(image, clean_name, top_prob, top3, ai_analysis, status):
    buffer = io.BytesIO()
    doc    = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5 * inch)
    styles = getSampleStyleSheet()
    story  = []

    title_style   = ParagraphStyle('Title',   parent=styles['Title'],   fontSize=22, textColor=colors.HexColor('#22c55e'), spaceAfter=6)
    heading_style = ParagraphStyle('Heading', parent=styles['Heading2'], fontSize=13, textColor=colors.HexColor('#1e40af'), spaceBefore=12, spaceAfter=4)
    normal_style  = ParagraphStyle('Normal',  parent=styles['Normal'],  fontSize=10, leading=14)
    label_style   = ParagraphStyle('Label',   parent=styles['Normal'],  fontSize=10, textColor=colors.HexColor('#6b7280'))

    story.append(Paragraph("PlantAI - Disease Detection Report", title_style))
    story.append(Paragraph(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", label_style))
    story.append(Spacer(1, 12))

    img_buf = io.BytesIO()
    image.save(img_buf, format='PNG')
    img_buf.seek(0)
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
        tmp.write(img_buf.read())
        tmp_path = tmp.name

    rl_image     = RLImage(tmp_path, width=2.2 * inch, height=2.2 * inch)
    status_color = '#22c55e' if 'healthy' in status.lower() else '#ef4444'

    result_data = [
        [Paragraph("<b>Disease</b>",    label_style), Paragraph(clean_name, normal_style)],
        [Paragraph("<b>Status</b>",     label_style), Paragraph(f"<font color='{status_color}'>{status}</font>", normal_style)],
        [Paragraph("<b>Confidence</b>", label_style), Paragraph(f"{top_prob:.2%}", normal_style)],
    ]
    result_table = Table(result_data, colWidths=[1.2*inch, 3*inch])
    result_table.setStyle(TableStyle([
        ('VALIGN',         (0,0), (-1,-1), 'TOP'),
        ('ROWBACKGROUNDS', (0,0), (-1,-1), [colors.HexColor('#f8fafc'), colors.white]),
        ('GRID',           (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
        ('PADDING',        (0,0), (-1,-1), 6),
    ]))

    layout_table = Table([[rl_image, result_table]], colWidths=[2.4*inch, 4.2*inch])
    layout_table.setStyle(TableStyle([
        ('VALIGN',      (0,0), (-1,-1), 'TOP'),
        ('LEFTPADDING', (1,0), (1, 0),  16),
    ]))
    story.append(layout_table)
    story.append(Spacer(1, 16))

    story.append(Paragraph("Top Predictions", heading_style))
    pred_data = [["Disease", "Confidence"]]
    for cls, p in top3:
        pred_data.append([cls, f"{p:.2%}"])
    pred_table = Table(pred_data, colWidths=[4.5*inch, 1.5*inch])
    pred_table.setStyle(TableStyle([
        ('BACKGROUND',     (0,0), (-1, 0), colors.HexColor('#1e40af')),
        ('TEXTCOLOR',      (0,0), (-1, 0), colors.white),
        ('FONTNAME',       (0,0), (-1, 0), 'Helvetica-Bold'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#f0fdf4'), colors.white]),
        ('GRID',           (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
        ('PADDING',        (0,0), (-1,-1), 8),
    ]))
    story.append(pred_table)
    story.append(Spacer(1, 16))

    story.append(Paragraph("AI Expert Analysis", heading_style))
    for line in ai_analysis.split('\n'):
        line = line.strip()
        if line:
            story.append(Paragraph(line, normal_style))
            story.append(Spacer(1, 4))

    doc.build(story)
    buffer.seek(0)
    os.unlink(tmp_path)
    return buffer

# =========================================================
# FILE UPLOADER
# =========================================================

uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "jpeg", "png"])

# =========================================================
# MAIN APP
# =========================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    # ── Step 1 · Blur ────────────────────────────────────
    if is_blurry(image):
        st.markdown(
            "<div class='error-card'>"
            "<h3 style='color:#ef4444;'>⚠️ Blurry Image Detected</h3>"
            "<p>Please upload a sharper, well-focused plant leaf photo.</p>"
            "</div>", unsafe_allow_html=True)
        st.stop()

    # ── Step 2 · Green content ───────────────────────────
    if not has_enough_green(image):
        st.markdown(
            "<div class='error-card'>"
            "<h3 style='color:#ef4444;'>❌ Invalid Plant Image</h3>"
            "<p>Not enough green content detected. Please upload a plant leaf photo.</p>"
            "</div>", unsafe_allow_html=True)
        st.stop()

    # ── Step 3 · Vision validation ───────────────────────
    with st.spinner("👁️ AI Vision is identifying the plant type..."):
        is_valid, plant_type, detected_plant, reason, vision_error = validate_plant_with_vision(image)

    if not is_valid:
        if vision_error:
            st.markdown(
                "<div class='error-card'>"
                "<h3 style='color:#ef4444;'>⚠️ Vision Validation Error</h3>"
                "<p>Plant type could not be verified. Please try again.</p>"
                f"<p style='color:#64748b; font-size:0.8rem;'>Error: {vision_error}</p>"
                "</div>", unsafe_allow_html=True)
        else:
            st.markdown(
                "<div class='warning-card'>"
                "<h3 style='color:#f59e0b;'>🌱 Unsupported Plant Detected</h3>"
                f"<p>Detected: <b style='color:#fbbf24;'>{detected_plant}</b></p>"
                f"<p style='color:#94a3b8; font-size:0.9rem;'>{reason}</p>"
                "<p style='margin-top:12px;'>PlantAI supports only:</p>"
                "<ul style='color:#e2e8f0; line-height:2;'>"
                "<li>🍅 <b>Tomato</b> leaves</li>"
                "<li>🥔 <b>Potato</b> leaves</li>"
                "<li>🫑 <b>Pepper Bell</b> leaves</li>"
                "</ul>"
                "</div>", unsafe_allow_html=True)
        st.stop()

    # ── Confirmed plant banner ───────────────────────────
    plant_emoji = {"tomato": "🍅", "potato": "🥔", "pepper": "🫑"}.get(plant_type, "🌿")
    st.markdown(
        f"<div class='info-card'>"
        f"<p style='margin:0; color:#93c5fd;'>👁️ Vision confirmed: "
        f"<b style='color:#60a5fa;'>{plant_emoji} {detected_plant}</b> — "
        f"running disease analysis...</p>"
        f"</div>",
        unsafe_allow_html=True
    )

    # ── Step 4 · Disease model + strict plant filter ─────
    img   = preprocess_image(image)
    preds = model.predict(img, verbose=0)
    probs = preds[0]

    try:
        clean_name, top_prob, top3 = predict_with_plant_filter(probs, plant_type)
    except ValueError as e:
        st.markdown(
            "<div class='error-card'>"
            "<h3 style='color:#ef4444;'>⚠️ Prediction Error</h3>"
            f"<p>{str(e)}</p>"
            "</div>", unsafe_allow_html=True)
        st.stop()

    status       = "Healthy Plant" if "healthy" in clean_name.lower() else "Disease Detected"
    status_color = "#22c55e"      if "healthy" in clean_name.lower() else "#ef4444"

    # ── Step 5 · Display results ─────────────────────────
    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.image(image, caption="Uploaded Leaf Image", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='badge'>Prediction Result</div>", unsafe_allow_html=True)
        st.markdown(f"<h2 style='margin-top:15px;'>{clean_name}</h2>", unsafe_allow_html=True)
        st.markdown(f"<p style='color:{status_color}; font-weight:600;'>{status}</p>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='color:#22c55e;'>Confidence: {top_prob:.2%}</h3>", unsafe_allow_html=True)
        st.progress(float(top_prob))
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Step 6 · Top predictions ─────────────────────────
    st.markdown("## 📊 Top Predictions")
    for cls, p in top3:
        st.markdown(
            f"<div style='background:linear-gradient(135deg,#111827,#1e293b);"
            f"padding:16px;border-radius:16px;margin-bottom:12px;'>"
            f"<div style='display:flex;justify-content:space-between;margin-bottom:8px;'>"
            f"<span style='font-weight:600;color:white;'>{cls}</span>"
            f"<span style='color:#22c55e;font-weight:600;'>{p:.2%}</span>"
            f"</div></div>",
            unsafe_allow_html=True
        )
        st.progress(float(p))

    # ── Step 7 · AI analysis ─────────────────────────────
    with st.spinner("🧠 AI Expert is analyzing disease..."):
        ai_analysis = get_ai_analysis(clean_name)

    st.markdown("## 🧠 AI Expert Analysis")
    st.markdown(
        f"<div style='background:#111827;border-radius:20px;padding:24px;"
        f"border:1px solid rgba(255,255,255,0.05);'>{ai_analysis}</div>",
        unsafe_allow_html=True
    )

    # ── Step 8 · PDF ─────────────────────────────────────
    pdf_buffer = generate_pdf_report(image, clean_name, top_prob, top3, ai_analysis, status)
    st.download_button(
        label="📄 Download PDF Report",
        data=pdf_buffer,
        file_name=f"PlantAI_Report_{clean_name.replace(' ', '_')}.pdf",
        mime="application/pdf"
    )

# =========================================================
# FOOTER
# =========================================================

st.markdown(
    "<hr style='margin-top:50px;border:1px solid #1e293b;'>"
    "<p style='text-align:center;color:#94a3b8;'>PlantAI • Deep Learning Plant Disease Detection</p>",
    unsafe_allow_html=True
)