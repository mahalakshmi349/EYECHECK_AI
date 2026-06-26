import os
import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import streamlit as st

# ── CONFIG ──────────────────────────────────────────────
RESNET_PATH = os.path.join("model", "resnet50_dr.pth")
EFFNET_PATH = os.path.join("model", "efficientnet_dr.pth")
IMG_SIZE    = 224
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["No DR", "Mild DR", "Moderate DR"]

# Image quality thresholds
BLUR_THRESHOLD  = 80.0
BRIGHTNESS_MIN  = 40
BRIGHTNESS_MAX  = 215
CONTRAST_MIN    = 30

CLASS_INFO = {
    "No DR": {
        "color": "#0a9e7e", "bg": "#edfaf6", "border": "#b2ead9",
        "icon": "✦", "emoji": "🟢",
        "meaning": "No signs of diabetic retinopathy detected in this fundus image.",
        "action": "Continue routine annual eye exams. Maintain good glycemic and BP control.",
        "urgency": "Routine Follow-up",
        "urgency_bg": "#d4f5ea", "urgency_color": "#0a7a60",
    },
    "Mild DR": {
        "color": "#b07d0a", "bg": "#fdf8e8", "border": "#f0d98a",
        "icon": "◈", "emoji": "🟡",
        "meaning": "Early retinopathy — microaneurysms likely present in the retina.",
        "action": "Ophthalmology review within 6–12 months. Tighten blood sugar management.",
        "urgency": "Non-Urgent Referral",
        "urgency_bg": "#fdf0c0", "urgency_color": "#8a5e00",
    },
    "Moderate DR": {
        "color": "#c0420a", "bg": "#fff1eb", "border": "#f5bfa0",
        "icon": "◉", "emoji": "🔴",
        "meaning": "Moderate DR — hemorrhages or exudates detected. Risk of vision loss.",
        "action": "Prompt referral to retinal specialist within 1–3 months.",
        "urgency": "Urgent Referral",
        "urgency_bg": "#ffe0d0", "urgency_color": "#a03000",
    },
}
BAR_COLORS = {"No DR": "#0a9e7e", "Mild DR": "#d4a017", "Moderate DR": "#d4500a"}
# ────────────────────────────────────────────────────────

st.set_page_config(
    page_title="EyeCheck AI · DR Screening",
    page_icon="👁", layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,700;1,500&family=Nunito:wght@400;600;700&family=Fira+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Nunito', sans-serif !important;
    background-color: #f7f3ee !important;
    color: #2c2c2c;
}
.main, .block-container {
    background-color: #f7f3ee !important;
    padding-top: 0.6rem !important;
    padding-left: 1.8rem !important;
    padding-right: 1.8rem !important;
    max-width: 100% !important;
}

/* HERO */
.hero {
    background: linear-gradient(120deg,#ffffff 0%,#eef6f3 60%,#fef9f0 100%);
    border: 1px solid #e2ddd7; border-radius: 14px;
    padding: 0.9rem 1.6rem; margin-bottom: 0.75rem;
    display: flex; align-items: center; justify-content: space-between;
    box-shadow: 0 2px 14px rgba(0,0,0,0.04);
    position: relative; overflow: hidden;
}
.hero::after {
    content:'👁'; position:absolute; right:10px; top:-12px;
    font-size:90px; opacity:0.055; pointer-events:none;
}
.hero-tag {
    font-family:'Fira Mono',monospace; font-size:0.60rem;
    letter-spacing:3px; text-transform:uppercase; color:#0a9e7e;
    background:#edfaf6; border:1px solid #b2ead9;
    padding:2px 10px; border-radius:20px; display:inline-block; margin-bottom:0.3rem;
}
.hero-title {
    font-family:'Playfair Display',serif; font-size:1.65rem;
    font-weight:700; color:#1a1a2e; line-height:1.1; margin-bottom:0.2rem;
}
.hero-title em { font-style:italic; color:#0a9e7e; }
.hero-sub { font-size:0.76rem; color:#6b6b6b; line-height:1.5; max-width:480px; }
.hero-badges { display:flex; gap:0.5rem; flex-shrink:0; flex-wrap:wrap; }
.badge {
    font-family:'Fira Mono',monospace; font-size:0.60rem;
    padding:3px 9px; border-radius:20px; border:1px solid; white-space:nowrap;
}

/* SCALE */
.scale-strip { display:flex; gap:7px; margin-bottom:0.75rem; }
.scale-item { flex:1; padding:7px 12px; border-radius:9px; border:1px solid; }
.scale-name { font-weight:700; font-size:0.76rem; margin-bottom:1px; }
.scale-desc { font-size:0.66rem; opacity:0.72; }

/* UPLOAD */
.upload-wrap {
    background:#ffffff; border:1.5px dashed #c8bfb5;
    border-radius:12px; padding:0.7rem 1rem 0.4rem;
    margin-bottom:0.7rem; box-shadow:0 1px 5px rgba(0,0,0,0.03);
}
.upload-title { font-size:0.85rem; font-weight:700; color:#1a1a2e; margin-bottom:0.1rem; }
.upload-hint  { font-size:0.70rem; color:#9a8f85; }

/* QUALITY CARD */
.quality-card {
    border-radius:10px; padding:0.75rem 1rem;
    border:1.5px solid; margin-bottom:0.7rem;
}
.quality-title {
    font-family:'Fira Mono',monospace; font-size:0.65rem;
    letter-spacing:2px; text-transform:uppercase; margin-bottom:0.5rem;
}
.quality-row { display:flex; align-items:center; gap:8px; margin-bottom:5px; }
.quality-label { font-size:0.75rem; font-weight:600; width:88px; flex-shrink:0; }
.quality-bar-bg { flex:1; height:6px; background:#ede8e3; border-radius:3px; overflow:hidden; }
.quality-bar-fill { height:100%; border-radius:3px; }
.quality-val { font-family:'Fira Mono',monospace; font-size:0.68rem; color:#6b6b6b; width:50px; text-align:right; }
.quality-warn {
    background:#fff8e1; border:1px solid #ffe082;
    border-left:3px solid #f9a825; border-radius:0 7px 7px 0;
    padding:0.5rem 0.8rem; font-size:0.72rem; color:#7a5f00;
    line-height:1.5; margin-top:0.4rem;
}
.quality-block {
    background:#fff1eb; border:1px solid #f5bfa0;
    border-left:3px solid #e53935; border-radius:0 7px 7px 0;
    padding:0.55rem 0.8rem; font-size:0.72rem; color:#a03000;
    line-height:1.5; margin-top:0.4rem; font-weight:600;
}

/* PREVIEW */
.preview-wrap {
    background:#ffffff; border:1px solid #e2ddd7;
    border-radius:10px; padding:0.7rem;
    margin-bottom:0.7rem; box-shadow:0 1px 5px rgba(0,0,0,0.03);
}
.preview-meta {
    font-family:'Fira Mono',monospace;
    font-size:0.66rem; color:#9a8f85; margin-top:0.35rem;
}

/* BUTTON */
.stButton > button {
    background: linear-gradient(135deg,#0a9e7e,#07c49a) !important;
    color:white !important; border:none !important;
    border-radius:9px !important; padding:0.5rem 1.5rem !important;
    font-family:'Nunito',sans-serif !important;
    font-weight:700 !important; font-size:0.88rem !important;
    width:100% !important;
    box-shadow:0 3px 12px rgba(10,158,126,0.28) !important;
    transition:all 0.2s !important;
}
.stButton > button:hover {
    transform:translateY(-1px) !important;
    box-shadow:0 5px 16px rgba(10,158,126,0.38) !important;
}

/* RESULT */
.result-card {
    border-radius:12px; padding:0.9rem 1.1rem;
    border:1.5px solid; margin:0.5rem 0 0.3rem;
    position:relative; overflow:hidden;
    box-shadow:0 2px 10px rgba(0,0,0,0.05);
}
.result-card::after {
    content:attr(data-icon); position:absolute; right:12px; top:8px;
    font-size:40px; opacity:0.10;
    font-family:'Playfair Display',serif; pointer-events:none;
}
.urgency-pill {
    display:inline-block; font-family:'Fira Mono',monospace;
    font-size:0.60rem; letter-spacing:2px; text-transform:uppercase;
    padding:2px 10px; border-radius:20px; margin-bottom:0.4rem; font-weight:500;
}
.result-grade {
    font-family:'Playfair Display',serif; font-size:1.5rem;
    font-weight:700; margin-bottom:0.25rem; line-height:1.15;
}
.result-meaning { font-size:0.78rem; line-height:1.5; margin-bottom:0.4rem; opacity:0.88; }
.result-action  { font-size:0.76rem; font-weight:600; opacity:0.92; }

/* ENSEMBLE ROWS */
.ensemble-row {
    display:flex; align-items:center; gap:8px;
    background:#faf8f5; border:1px solid #e2ddd7;
    border-radius:8px; padding:0.45rem 0.8rem; margin:0.3rem 0;
}
.ensemble-label { font-family:'Fira Mono',monospace; font-size:0.66rem; color:#9a8f85; width:115px; }
.ensemble-val   { font-family:'Fira Mono',monospace; font-size:0.74rem; font-weight:600; }

/* CONFIDENCE */
.conf-header {
    font-family:'Fira Mono',monospace; font-size:0.62rem;
    letter-spacing:2px; text-transform:uppercase;
    color:#9a8f85; margin:0.65rem 0 0.4rem;
}
.conf-row { display:flex; align-items:center; gap:8px; margin-bottom:5px; }
.conf-name { font-size:0.76rem; font-weight:600; width:100px; flex-shrink:0; color:#2c2c2c; }
.conf-bar-bg { flex:1; height:6px; background:#ede8e3; border-radius:3px; overflow:hidden; }
.conf-bar-fill { height:100%; border-radius:3px; }
.conf-pct { font-family:'Fira Mono',monospace; font-size:0.70rem; color:#6b6b6b; width:38px; text-align:right; }

/* GRADCAM */
.gradcam-header {
    font-family:'Fira Mono',monospace; font-size:0.62rem;
    letter-spacing:2px; text-transform:uppercase;
    color:#9a8f85; margin:0.65rem 0 0.35rem;
}
.gradcam-note { font-size:0.70rem; color:#9a8f85; line-height:1.4; margin-top:0.3rem; }

/* DISCLAIMER */
.disclaimer {
    background:#faf8f5; border:1px solid #e2ddd7;
    border-left:3px solid #e0a050; border-radius:0 6px 6px 0;
    padding:0.5rem 0.8rem; font-size:0.70rem;
    color:#7a6a55; line-height:1.5; margin-top:0.6rem;
}

/* INFO TILES */
.info-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:8px; margin-top:0.7rem; }
.info-tile {
    background:#ffffff; border:1px solid #e2ddd7;
    border-radius:10px; padding:0.75rem;
    box-shadow:0 1px 4px rgba(0,0,0,0.03);
}
.info-tile-icon  { font-size:1rem; margin-bottom:4px; }
.info-tile-title { font-size:0.74rem; font-weight:700; color:#1a1a2e; margin-bottom:2px; }
.info-tile-body  { font-size:0.68rem; color:#7a7a7a; line-height:1.45; }

/* FOOTER */
.footer {
    text-align:center; font-family:'Fira Mono',monospace;
    font-size:0.62rem; color:#b0a898;
    margin-top:0.8rem; padding-top:0.6rem; border-top:1px solid #e2ddd7;
}

#MainMenu, footer, header { visibility:hidden; }
.stFileUploader label { display:none !important; }
section[data-testid="stSidebar"] { display:none; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
#  IMAGE QUALITY CHECKER
# ══════════════════════════════════════════════════════
def check_image_quality(image: Image.Image):
    img_np     = np.array(image.convert("RGB"))
    gray       = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = float(gray.mean())
    contrast   = float(gray.std())

    issues = []
    if blur_score < BLUR_THRESHOLD:
        issues.append(f"Image too blurry (score: {blur_score:.1f})")
    if brightness < BRIGHTNESS_MIN:
        issues.append(f"Image too dark (brightness: {brightness:.1f})")
    if brightness > BRIGHTNESS_MAX:
        issues.append(f"Image overexposed (brightness: {brightness:.1f})")
    if contrast < CONTRAST_MIN:
        issues.append(f"Low contrast / washed out (contrast: {contrast:.1f})")

    critical = blur_score < BLUR_THRESHOLD / 2 or brightness < 20 or brightness > 240
    if critical or len(issues) >= 2:
        status = "poor"
    elif len(issues) == 1:
        status = "warn"
    else:
        status = "good"

    return status, {"blur": blur_score, "brightness": brightness, "contrast": contrast}, issues


def qbar_color(value, low, high):
    norm = max(0.0, min(1.0, (value - low) / (high - low + 1e-6)))
    if norm > 0.6: return "#0a9e7e"
    if norm > 0.3: return "#d4a017"
    return "#d4500a"


# ══════════════════════════════════════════════════════
#  MODEL LOADING  ← MATCHES train_model.py ARCHITECTURE
# ══════════════════════════════════════════════════════
@st.cache_resource
def load_models():
    # ── ResNet50 — must match improved train_model.py head ──
    resnet = models.resnet50(weights=None)
    resnet.fc = nn.Sequential(
        nn.BatchNorm1d(resnet.fc.in_features),
        nn.Dropout(0.4),
        nn.Linear(resnet.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 3)
    )
    if not os.path.exists(RESNET_PATH):
        st.error("❌ resnet50_dr.pth not found. Run `python train_model.py` first.")
        st.stop()
    resnet.load_state_dict(torch.load(RESNET_PATH, map_location=DEVICE))
    resnet.eval().to(DEVICE)

    # ── EfficientNet-B4 — must match improved train_model.py head ──
    effnet = models.efficientnet_b4(weights=None)
    in_f   = effnet.classifier[1].in_features
    effnet.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_f, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 3)
    )
    if not os.path.exists(EFFNET_PATH):
        st.error("❌ efficientnet_dr.pth not found. Run `python train_model.py` first.")
        st.stop()
    effnet.load_state_dict(torch.load(EFFNET_PATH, map_location=DEVICE))
    effnet.eval().to(DEVICE)

    return resnet, effnet


# ══════════════════════════════════════════════════════
#  INFERENCE + WEIGHTED ENSEMBLE
# ══════════════════════════════════════════════════════
def preprocess(image: Image.Image):
    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    return tf(image).unsqueeze(0).to(DEVICE)

def ensemble_predict(image, resnet, effnet):
    tensor = preprocess(image)
    with torch.no_grad():
        p_r = F.softmax(resnet(tensor), dim=1).squeeze().cpu().numpy()
        p_e = F.softmax(effnet(tensor), dim=1).squeeze().cpu().numpy()
    # 70/30 weighted — ResNet is stronger (87.7% vs 85.2%)
    ens   = (0.7 * p_r) + (0.3 * p_e)
    grade = CLASS_NAMES[int(np.argmax(ens))]
    return grade, ens, p_r, p_e


# ══════════════════════════════════════════════════════
#  GRADCAM
# ══════════════════════════════════════════════════════
class GradCAM:
    def __init__(self, model, target_layer):
        self.gradients   = None
        self.activations = None
        target_layer.register_forward_hook(
            lambda _, __, o: setattr(self, "activations", o.detach())
        )
        target_layer.register_full_backward_hook(
            lambda _, __, g: setattr(self, "gradients", g[0].detach())
        )
        self.model = model

    def generate(self, tensor, class_idx):
        self.model.zero_grad()
        out = self.model(tensor)
        out[0, class_idx].backward()
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam     = (weights * self.activations).sum(dim=1).squeeze()
        cam     = F.relu(torch.tensor(cam.numpy())).numpy()
        cam    -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam

def apply_gradcam(image: Image.Image, resnet, class_idx: int) -> Image.Image:
    gcam   = GradCAM(resnet, resnet.layer4[-1])
    tensor = preprocess(image)
    tensor.requires_grad_(True)
    cam    = gcam.generate(tensor, class_idx)
    img_np  = np.array(image.resize((IMG_SIZE, IMG_SIZE)))
    cam_res = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_res), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img_np, 0.55, heatmap, 0.45, 0)
    return Image.fromarray(overlay)


# ══════════════════════════════════════════════════════
#  PAGE UI
# ══════════════════════════════════════════════════════

# ── Hero ──
st.markdown("""
<div class="hero">
  <div>
    <div class="hero-tag">AI · Ophthalmology · Early Detection</div>
    <div class="hero-title">EyeCheck <em>AI</em> &nbsp;·&nbsp; DR Screening</div>
    <div class="hero-sub">
      ResNet50 (87.7%) + EfficientNet-B4 (85.2%) ensemble · GradCAM explainability ·
      Image quality gating — reliable low-cost community screening.
    </div>
  </div>
  <div class="hero-badges">
    <div class="badge" style="background:#edfaf6;border-color:#b2ead9;color:#0a7a60;">ResNet50 87.7%</div>
    <div class="badge" style="background:#eef3fd;border-color:#b2caea;color:#1a4a8a;">EfficientNet 85.2%</div>
    <div class="badge" style="background:#fdf8e8;border-color:#f0d98a;color:#8a5e00;">GradCAM XAI</div>
    <div class="badge" style="background:#f3eef9;border-color:#d0b2ea;color:#5a1a8a;">Quality Check</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Scale ──
st.markdown("""
<div class="scale-strip">
  <div class="scale-item" style="background:#edfaf6;border-color:#b2ead9;">
    <div class="scale-name" style="color:#0a7a60;">🟢 No DR</div>
    <div class="scale-desc" style="color:#0a9e7e;">Healthy retina</div>
  </div>
  <div class="scale-item" style="background:#fdf8e8;border-color:#f0d98a;">
    <div class="scale-name" style="color:#8a5e00;">🟡 Mild DR</div>
    <div class="scale-desc" style="color:#b07d0a;">Microaneurysms</div>
  </div>
  <div class="scale-item" style="background:#fff1eb;border-color:#f5bfa0;">
    <div class="scale-name" style="color:#a03000;">🔴 Moderate DR</div>
    <div class="scale-desc" style="color:#c0420a;">Hemorrhages / Exudates</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Two Columns ──
left, right = st.columns([1, 1], gap="medium")

with left:
    st.markdown("""
    <div class="upload-wrap">
      <div class="upload-title">📤 Upload Fundus Image</div>
      <div class="upload-hint">JPG · PNG · WEBP &nbsp;|&nbsp; Max 10 MB</div>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader("fundus", type=["jpg","jpeg","png","webp"],
                                label_visibility="collapsed")

    if uploaded:
        image = Image.open(uploaded).convert("RGB")

        st.markdown('<div class="preview-wrap">', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown(
            f'<div class="preview-meta">📎 {uploaded.name} &nbsp;·&nbsp; '
            f'{image.size[0]}×{image.size[1]}px &nbsp;·&nbsp; {uploaded.size//1024} KB</div>',
            unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Quality Check ──
        quality_status, qm, issues = check_image_quality(image)
        blur_pct   = min(100, qm["blur"] / 300 * 100)
        bright_pct = min(100, qm["brightness"] / 255 * 100)
        cont_pct   = min(100, qm["contrast"] / 80 * 100)

        iq_bg     = {"good": "#edfaf6", "warn": "#fdf8e8", "poor": "#fff1eb"}[quality_status]
        iq_border = {"good": "#b2ead9", "warn": "#f0d98a", "poor": "#f5bfa0"}[quality_status]
        iq_title  = {"good": "#0a7a60", "warn": "#8a5e00", "poor": "#a03000"}[quality_status]
        iq_label  = {"good": "✅ Good Quality", "warn": "⚠️ Acceptable", "poor": "❌ Poor Quality"}[quality_status]

        warn_html  = "".join(f'<div class="quality-warn">⚠️ {i}</div>' for i in issues) if quality_status == "warn" else ""
        block_html = "".join(f'<div class="quality-block">🚫 {i}</div>' for i in issues) if quality_status == "poor" else ""

        st.markdown(f"""
        <div class="quality-card" style="background:{iq_bg};border-color:{iq_border};">
          <div class="quality-title" style="color:{iq_title};">
            Image Quality &nbsp;·&nbsp; {iq_label}
          </div>
          <div class="quality-row">
            <div class="quality-label">Sharpness</div>
            <div class="quality-bar-bg">
              <div class="quality-bar-fill" style="width:{blur_pct:.1f}%;background:{qbar_color(qm['blur'],0,300)};"></div>
            </div>
            <div class="quality-val">{qm['blur']:.1f}</div>
          </div>
          <div class="quality-row">
            <div class="quality-label">Brightness</div>
            <div class="quality-bar-bg">
              <div class="quality-bar-fill" style="width:{bright_pct:.1f}%;background:{qbar_color(qm['brightness'],BRIGHTNESS_MIN,BRIGHTNESS_MAX)};"></div>
            </div>
            <div class="quality-val">{qm['brightness']:.1f}</div>
          </div>
          <div class="quality-row">
            <div class="quality-label">Contrast</div>
            <div class="quality-bar-bg">
              <div class="quality-bar-fill" style="width:{cont_pct:.1f}%;background:{qbar_color(qm['contrast'],0,80)};"></div>
            </div>
            <div class="quality-val">{qm['contrast']:.1f}</div>
          </div>
          {warn_html}{block_html}
        </div>
        """, unsafe_allow_html=True)

        if quality_status == "poor":
            st.error("❌ Image quality too poor for reliable diagnosis. Please upload a clearer fundus photograph.")
            run = False
        else:
            if quality_status == "warn":
                st.warning("⚠️ Image quality is suboptimal. Results may be less reliable.")
            run = st.button("👁  Run Ensemble + GradCAM Analysis")
    else:
        st.markdown("""
        <div class="info-grid">
          <div class="info-tile">
            <div class="info-tile-icon">🔬</div>
            <div class="info-tile-title">Dual Ensemble</div>
            <div class="info-tile-body">ResNet50 + EfficientNet-B4 averaged for robust predictions.</div>
          </div>
          <div class="info-tile">
            <div class="info-tile-icon">🗺️</div>
            <div class="info-tile-title">GradCAM Heatmap</div>
            <div class="info-tile-body">Visual map of retinal regions the AI focused on.</div>
          </div>
          <div class="info-tile">
            <div class="info-tile-icon">🛡️</div>
            <div class="info-tile-title">Quality Gating</div>
            <div class="info-tile-body">Blurry or dark images are flagged before analysis.</div>
          </div>
        </div>
        """, unsafe_allow_html=True)
        run = False

with right:
    if uploaded and run:
        with st.spinner("Running ensemble inference + GradCAM..."):
            resnet, effnet       = load_models()
            grade, ens, p_r, p_e = ensemble_predict(image, resnet, effnet)
            info                 = CLASS_INFO[grade]
            class_idx            = CLASS_NAMES.index(grade)
            gradcam_img          = apply_gradcam(image, resnet, class_idx)

        # Result card
        st.markdown(f"""
        <div class="result-card" data-icon="{info['icon']}"
             style="background:{info['bg']};border-color:{info['border']};">
          <div class="urgency-pill"
               style="background:{info['urgency_bg']};color:{info['urgency_color']};">
            {info['urgency']}
          </div>
          <div class="result-grade" style="color:{info['color']};">
            {info['emoji']} &nbsp; {grade}
          </div>
          <div class="result-meaning" style="color:{info['color']};">{info['meaning']}</div>
          <div class="result-action"  style="color:{info['color']};">📋 &nbsp;{info['action']}</div>
        </div>
        """, unsafe_allow_html=True)

        # Per-model scores
        st.markdown("""
        <div class="ensemble-row">
          <span class="ensemble-label">ResNet50 (70%)</span>
          <span class="ensemble-val" style="color:#1a5a9a;">
            {r0:.0f}% No DR &nbsp;|&nbsp; {r1:.0f}% Mild &nbsp;|&nbsp; {r2:.0f}% Moderate
          </span>
        </div>
        <div class="ensemble-row">
          <span class="ensemble-label">EfficientNet (30%)</span>
          <span class="ensemble-val" style="color:#6a1a8a;">
            {e0:.0f}% No DR &nbsp;|&nbsp; {e1:.0f}% Mild &nbsp;|&nbsp; {e2:.0f}% Moderate
          </span>
        </div>
        """.format(
            r0=p_r[0]*100, r1=p_r[1]*100, r2=p_r[2]*100,
            e0=p_e[0]*100, e1=p_e[1]*100, e2=p_e[2]*100,
        ), unsafe_allow_html=True)

        # Ensemble confidence bars
        st.markdown('<div class="conf-header">Ensemble Confidence (70/30 weighted)</div>',
                    unsafe_allow_html=True)
        bars = ""
        for i, name in enumerate(CLASS_NAMES):
            pct = float(ens[i]) * 100
            bars += f"""
            <div class="conf-row">
              <div class="conf-name">{name}</div>
              <div class="conf-bar-bg">
                <div class="conf-bar-fill"
                     style="width:{pct:.1f}%;background:{BAR_COLORS[name]};"></div>
              </div>
              <div class="conf-pct">{pct:.1f}%</div>
            </div>"""
        st.markdown(bars, unsafe_allow_html=True)

        # GradCAM
        st.markdown('<div class="gradcam-header">GradCAM — Retinal Attention Map</div>',
                    unsafe_allow_html=True)
        st.image(gradcam_img, use_container_width=True)
        st.markdown(f"""
        <div class="gradcam-note">
          🔴 Red/yellow = regions the AI focused on to predict <strong>{grade}</strong>.
          These correspond to lesion-prone zones in the retina.
        </div>
        """, unsafe_allow_html=True)

        # Disclaimer
        st.markdown("""
        <div class="disclaimer">
          ⚠️ <strong>Clinical Disclaimer:</strong> AI screening aid only — not a clinical
          diagnosis. All findings must be confirmed by a certified ophthalmologist.
        </div>
        """, unsafe_allow_html=True)

    elif not uploaded:
        st.markdown("""
        <div style="display:flex;flex-direction:column;align-items:center;
                    justify-content:center;background:#ffffff;border:1px solid #e2ddd7;
                    border-radius:12px;padding:2rem;text-align:center;
                    color:#b0a898;min-height:300px;">
          <div style="font-size:2.5rem;margin-bottom:0.6rem;opacity:0.35;">👁</div>
          <div style="font-family:'Fira Mono',monospace;font-size:0.72rem;
                      letter-spacing:2px;text-transform:uppercase;">
            Awaiting image upload
          </div>
          <div style="font-size:0.74rem;margin-top:0.4rem;color:#c8bfb5;">
            Upload a fundus photo on the left to begin
          </div>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ──
st.markdown("""
<div class="footer">
  EyeCheck AI &nbsp;·&nbsp; ResNet50 87.7% + EfficientNet-B4 85.2% Ensemble &nbsp;·&nbsp;
  GradCAM XAI &nbsp;·&nbsp; Image Quality Gating &nbsp;·&nbsp; APTOS 2019
</div>
""", unsafe_allow_html=True)