import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
import io
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# CONFIGURATION PAGE
st.set_page_config(
    page_title="Dogs vs Cats CLASSIFICATION",
    page_icon="🐾",
    layout="centered"
)

# CONSTANTES
IMG_SIZE = (128, 128) #Toutes les images sont redimensionnées
MODEL_PATH = "model_classification_DL.h5" #chemin du moodèle

CLASS_INFO = {
    0: {"label": "Chat",  "emoji": "🐱", "color": "#F4B53D", "stamp": "FÉLIN CONFIRMÉ"},
    1: {"label": "Chien", "emoji": "🐶", "color": "#3FC1B0", "stamp": "CANIN CONFIRMÉ"},
}

#STYLE CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fredoka:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: radial-gradient(circle at 15% 0%, #262849 0%, #1A1B2E 55%, #14152480 100%);
    color: #F1EFE7;
}

#MainMenu, header, footer {visibility: hidden;}

/* Hero */
.hero-wrap { text-align: center; padding: 1.4rem 0 0.6rem 0; }
.hero-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.96rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #9C9AD1;
}
.hero-title {
    font-family: 'Fredoka', sans-serif;
    font-weight: 700;
    font-size: 4.4rem;
    margin: 0.2rem 0 0.3rem 0;
    background: linear-gradient(90deg, #F4B53D, #3FC1B0);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
}
.hero-tag {
    font-size: 0.85rem;
    font-family: 'JetBrains Mono', monospace;
    color: #C9C7E8;
    max-width: 30rem;
    margin: 0 auto;
    text-align: center;
}

/* Zone de dépôt */
[data-testid="stFileUploader"] {
    border: 1.5px dashed #4B4C7A;
    border-radius: 16px;
    background: #1F2038;
    padding: 0.6rem;
    transition: border-color 0.25s ease;
}
[data-testid="stFileUploader"]:hover { border-color: #3FC1B0; }

/* Cadre "scan" de l'image */
.scan-frame {
    border-radius: 18px;
    overflow: hidden;
    border: 1px solid #34355C;
    box-shadow: 0 12px 30px -12px rgba(0,0,0,0.6);
    margin-top: 0.8rem;
}

/*  Carte résultat */
.result-card {
    margin-top: 1.2rem;
    background: linear-gradient(155deg, #232442 0%, #1D1E38 100%);
    border: 1px solid #34355C;
    border-radius: 18px;
    padding: 1.4rem 1.5rem;
    display: flex;
    align-items: center;
    gap: 1.1rem;
    animation: cardIn 0.5s ease;
}
@keyframes cardIn {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}
.result-species {
    font-family: 'Fredoka', sans-serif;
    font-size: 1.6rem;
    font-weight: 600;
    margin: 0.1rem 0 0.5rem 0;
}
.result-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: #9C9AD1;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* Tampon de confiance */
.stamp {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    font-size: 0.68rem;
    letter-spacing: 0.05em;
    border: 2px solid currentColor;
    border-radius: 999px;
    padding: 0.35rem 0.8rem;
    display: inline-block;
}
@keyframes stampIn {
    from { opacity: 0; scale(1.4); }
    to { opacity: 1; scale(1); }
}

/* Jauge de confiance  */
.meter-track {
    height: 9px;
    width: 100%;
    background: #14152e;
    border-radius: 999px;
    overflow: hidden;
    margin-top: 0.8rem;
}
.meter-fill {
    height: 100%;
    border-radius: 999px;
    transition: width 0.6s ease;
}

/* Fiche technique  */
.debug-box {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: #B9B7E0;
    line-height: 1.7;
}

/* Pied de page */
.footer-caption {
    text-align: center;
    color: #6D6C97;
    font-size: 0.72rem;
    margin-top: 2.2rem;
    letter-spacing: 0.04em;
}
</style>
""", unsafe_allow_html=True)

# LOAD MODEL (CACHE) Pour charger le modèle une seule fois
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# HERO
st.markdown("""
<div class="hero-wrap">
    <div class="hero-eyebrow">CLASSIFICATION CATS & DOGS EN UTILISANT LE DEEP LEARNING · CNN + MobileNetV2</div>
    <div class="hero-title">🐾 Dogs vs Cats</div>
    <p class="hero-tag">UPLOADER UNE IMAGE - LE MODÈLE L'EXAMINE ET DÉLIVRE SA CLASSIFICATION EN QUELQUES SECONDES !</p>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Uploader une image (jpg, jpeg, png, webp)",
    type=["jpg", "jpeg", "png", "webp"],
    label_visibility="collapsed"
)

#PREDICTION
if uploaded_file is not None:

    # Lecture image SAFE 
    try:
        image = Image.open(io.BytesIO(uploaded_file.read())).convert("RGB")
    except UnidentifiedImageError:
        st.error("Image invalide ou corrompue. Essaie une autre image.")
        st.stop()

    # Affichage de l'image dans un cadre "scan"
    st.markdown('<div class="scan-frame">', unsafe_allow_html=True)
    st.image(image, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    #Preprocessing MobileNetV2
    img = image.resize(IMG_SIZE)
    img_array = np.array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    with st.spinner("Analyse en cours..."):
        proba = model.predict(img_array, verbose=0)[0][0]

    predicted_class = 1 if proba > 0.5 else 0
    confidence = proba if predicted_class == 1 else 1 - proba
    info = CLASS_INFO[predicted_class]

    if confidence > 0.90:
        verdict, vcolor = "CONFIANCE ÉLEVÉE", "#3FC1B0"
    elif confidence > 0.70:
        verdict, vcolor = "CONFIANCE MOYENNE", "#F4B53D"
    else:
        verdict, vcolor = "À VÉRIFIER", "#FF6B6B"

    #CARTE RÉSULTAT
    st.markdown(f"""
    <div class="result-card">
        <div style="font-size:2.6rem;">{info['emoji']}</div>
        <div style="flex:1;">
            <div class="result-sub">Résultat de l'examen</div>
            <p class="result-species" style="color:{info['color']};">{info['label']}</p>
            <span class="stamp" style="color:{vcolor};">{verdict} · {confidence*100:.1f}%</span>
            <div class="meter-track">
                <div class="meter-fill" style="width:{confidence*100:.1f}%; background:{info['color']};"></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("Fiche technique"):
        st.markdown(f"""
        <div class="debug-box">
        Score brut (probabilité Chien) : {proba:.4f}<br>
        Préprocessing : MobileNetV2 preprocess_input<br>
        Dimensions d'entrée : (1, 128, 128, 3)
        </div>
        """, unsafe_allow_html=True)

#FOOTER
st.markdown("""
<div class="footer-caption">PROJET DEEP LEARNING — CNN + TRANSFER LEARNING (MOBILENETV2) — DOGS VS CATS CLASSIFICATION</div>
""", unsafe_allow_html=True)
