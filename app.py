import streamlit as st
import tensorflow as tf
import pickle
import re
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Page configuration
st.set_page_config(
    page_title="Sentiment Analysis - Business Review",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Constants
MAX_LEN = 128
MODEL_PATH = "kaggle/working/model_outputs/CNN + BiLSTM_best.h5"
TOKENIZER_PATH = "kaggle/working/model_outputs/tokenizer.pkl"
LABEL_ENCODER_PATH = "kaggle/working/model_outputs/label_encoder.pkl"

# Initialize session state for language
if "language" not in st.session_state:
    st.session_state.language = "en"  # Default to English

# Translations dictionary
TRANSLATIONS = {
    "en": {
        # Navigation
        "about": "About",
        "settings": "Settings",
        "language": "Language",
        # Hero section
        "title": "Sentiment Analysis",
        "subtitle": "🇮🇩 Indonesian Business Review Analyzer",
        "description": "Powered by CNN + BiLSTM with Attention • Real-time AI predictions with 74.97% accuracy",
        # Sidebar
        "about_title": "About This Tool",
        "about_text": "This AI-powered sentiment analysis model classifies Indonesian business reviews into three categories:",
        "model_info": "Model Information",
        "architecture": "Architecture",
        "accuracy": "Accuracy",
        "dataset_size": "Dataset Size",
        "samples": "samples",
        "try_examples": "Try Examples",
        # Sentiments
        "positive": "Positive",
        "neutral": "Neutral",
        "negative": "Negative",
        # Input section
        "input_title": "Enter Your Review Text",
        "input_subtitle": "Type or paste a business review below and click analyze to see the sentiment prediction.",
        "input_placeholder": "Example: Masukkan teks ulasan bisnis Indonesia Anda di sini... ⚡",
        "input_label": "Type or paste your Indonesian business review here:",
        "analyze_button": "🔍 Analyze Sentiment",
        # Results
        "results_title": "Analysis Results",
        "sentiment_label": "Sentiment",
        "confidence_label": "Confidence",
        "distribution_title": "Sentiment Distribution",
        "statistics_title": "Text Statistics",
        "word_count": "Word Count",
        "characters": "Characters",
        "cleaned_words": "Cleaned Words",
        "view_preprocessed": "🔍 View Preprocessed Text",
        # Messages
        "loading_model": "Loading model...",
        "analyzing": "Analyzing sentiment...",
        "empty_warning": "⚠️ Please enter some text to analyze.",
        "error_loading": "Failed to load model. Please check the model files.",
        "error_prediction": "Error loading model artifacts:",
        # Footer
        "powered_by": "Powered by",
        "model_description": "CNN + BiLSTM with Attention Mechanism",
        "copyright": "© 2024",
        # Example texts
        "example_positive": "kloningan daihatsu masuk pasar indonesia epmb menerima hak ekslusif dalam merakit mendistribusi kendaraan merek lingbox pasar malaysia indonesia lingbox mini ev masuk tanah air https co ow ag https co ihkzxaj",
        "example_neutral": "Pemerintah sedang menyiapkan insentif untuk kendaraan bermotor listrik.",
        "example_negative": "Mobil listrik sangat mengecewakan dan tidak layak dibeli. Saya sangat kecewa dengan kualitasnya yang buruk dan harganya terlalu mahal.",
    },
    "id": {
        # Navigation
        "about": "Tentang",
        "settings": "Pengaturan",
        "language": "Bahasa",
        # Hero section
        "title": "Analisis Sentimen",
        "subtitle": "🇮🇩 Penganalisis Ulasan Bisnis Indonesia",
        "description": "Didukung oleh CNN + BiLSTM dengan Attention • Prediksi AI real-time dengan akurasi 74.97%",
        # Sidebar
        "about_title": "Tentang Alat Ini",
        "about_text": "Model analisis sentimen bertenaga AI ini mengklasifikasikan ulasan bisnis Indonesia ke dalam tiga kategori:",
        "model_info": "Informasi Model",
        "architecture": "Arsitektur",
        "accuracy": "Akurasi",
        "dataset_size": "Ukuran Dataset",
        "samples": "sampel",
        "try_examples": "Coba Contoh",
        # Sentiments
        "positive": "Positif",
        "neutral": "Netral",
        "negative": "Negatif",
        # Input section
        "input_title": "Masukkan Teks Ulasan Anda",
        "input_subtitle": "Ketik atau tempel ulasan bisnis di bawah ini dan klik analisis untuk melihat prediksi sentimen.",
        "input_placeholder": "Contoh: Masukkan teks ulasan bisnis Indonesia Anda di sini... ⚡",
        "input_label": "Ketik atau tempel ulasan bisnis Indonesia Anda di sini:",
        "analyze_button": "🔍 Analisis Sentimen",
        # Results
        "results_title": "Hasil Analisis",
        "sentiment_label": "Sentimen",
        "confidence_label": "Kepercayaan",
        "distribution_title": "Distribusi Sentimen",
        "statistics_title": "Statistik Teks",
        "word_count": "Jumlah Kata",
        "characters": "Karakter",
        "cleaned_words": "Kata Bersih",
        "view_preprocessed": "🔍 Lihat Teks Terproses",
        # Messages
        "loading_model": "Memuat model...",
        "analyzing": "Menganalisis sentimen...",
        "empty_warning": "⚠️ Silakan masukkan teks untuk dianalisis.",
        "error_loading": "Gagal memuat model. Silakan periksa file model.",
        "error_prediction": "Kesalahan memuat artefak model:",
        # Footer
        "powered_by": "Didukung oleh",
        "model_description": "CNN + BiLSTM dengan Mekanisme Attention",
        "copyright": "© 2024",
        # Example texts
        "example_positive": "kloningan daihatsu masuk pasar indonesia epmb menerima hak ekslusif dalam merakit mendistribusi kendaraan merek lingbox pasar malaysia indonesia lingbox mini ev masuk tanah air https co ow ag https co ihkzxaj",
        "example_neutral": "Pemerintah sedang menyiapkan insentif untuk kendaraan bermotor listrik.",
        "example_negative": "Mobil listrik sangat mengecewakan dan tidak layak dibeli. Saya sangat kecewa dengan kualitasnya yang buruk dan harganya terlalu mahal.",
    },
}


def get_text(key):
    """Get translated text based on current language"""
    return TRANSLATIONS[st.session_state.language].get(key, key)


# Single professional theme (no light/dark toggle)


# Single professional color palette
def get_theme_colors():
    return {
        "bg_color": "#0b1220",  # deep navy for modern look
        "text_color": "#e6eaf2",  # soft off-white
        "primary_color": "#7c8cf8",  # indigo-400
        "secondary_color": "#5dd39e",  # teal-green accent
        "accent_color": "#f38fb1",  # soft pink accent
        "positive_bg": "rgba(93, 211, 158, 0.15)",
        "positive_border": "#5dd39e",
        "positive_text": "#c9f7e6",
        "positive_glow": "rgba(93, 211, 158, 0.35)",
        "neutral_bg": "rgba(245, 158, 11, 0.15)",
        "neutral_border": "#f59e0b",
        "neutral_text": "#ffe2b5",
        "neutral_glow": "rgba(245, 158, 11, 0.35)",
        "negative_bg": "rgba(239, 68, 68, 0.15)",
        "negative_border": "#ef4444",
        "negative_text": "#ffd2d2",
        "negative_glow": "rgba(239, 68, 68, 0.35)",
        "card_bg": "rgba(16, 23, 42, 0.75)",
        "card_border": "rgba(124, 140, 248, 0.25)",
        "input_bg": "rgba(15, 22, 39, 0.85)",
        "input_border": "rgba(124, 140, 248, 0.35)",
        "shadow": "rgba(0, 0, 0, 0.6)",
    }


# Get current theme colors
colors = get_theme_colors()


# Custom CSS for UI - Modern Design
def get_custom_css():
    colors = get_theme_colors()
    return f"""
    <style>
    /* Import Modern Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Poppins:wght@400;500;600;700&display=swap');
    
    /* Global Theme with Gradient Background */
    body {{
        color: {colors["text_color"]} !important;
        background: {colors["bg_color"]} !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }}
    
    .stApp {{
        background: {colors["bg_color"]} !important;
        color: {colors["text_color"]} !important;
    }}
    
    .main {{
        background: {colors["bg_color"]} !important;
        color: {colors["text_color"]} !important;
    }}
    
    .block-container {{
        background: transparent !important;
        color: {colors["text_color"]} !important;
    }}
    
    /* Hide Streamlit Branding (but keep sidebar toggle) */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    
    /* Style header to match our design */
    header {{
        background: transparent !important;
    }}
    
    header[data-testid="stHeader"] {{
        background: transparent !important;
    }}
    
    /* COMPLETELY HIDE all sidebar toggle buttons - sidebar always visible */
    header button,
    header [data-testid="baseButton-header"],
    [data-testid="stHeader"] button,
    button[kind="header"],
    button[data-testid="collapsedControl"],
    section[data-testid="stSidebar"] button[kind="header"],
    section[data-testid="stSidebar"] > div > button,
    section[data-testid="stSidebar"] [data-testid="collapsedControl"],
    [data-testid="collapsedControl"],
    div[data-testid="collapsedControl"] {{
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        pointer-events: none !important;
        width: 0 !important;
        height: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
        border: none !important;
    }}
    
    /* Force sidebar to always be visible and prevent collapse */
    section[data-testid="stSidebar"] {{
        display: block !important;
        visibility: visible !important;
        opacity: 1 !important;
        position: relative !important;
        transform: translateX(0) !important;
        min-width: 24rem !important;
        max-width: 24rem !important;
    }}
    
    /* Force header area to be visible */
    header,
    [data-testid="stHeader"] {{
        visibility: visible !important;
        opacity: 1 !important;
        background: transparent !important;
        pointer-events: auto !important;
    }}
    
    /* Force toolbar to be visible */
    .stToolbarActions,
    [data-testid="stToolbar"] {{
        visibility: visible !important;
        opacity: 1 !important;
    }}
    
    /* Streamlit toolbar buttons */
    .stToolbar {{
        opacity: 0.7;
        transition: opacity 0.3s ease;
    }}
    
    .stToolbar:hover {{
        opacity: 1;
    }}
    
    /* Hide deploy button if present */
    .stDeployButton {{
        visibility: hidden;
    }}
    
    /* Headers with Better Typography */
    h1, h2, h3, h4, h5, h6 {{
        font-family: 'Poppins', sans-serif;
        font-weight: 700;
        color: {colors["text_color"]};
        letter-spacing: -0.02em;
        line-height: 1.2;
    }}
    
    h1 {{ font-size: 2.5rem; margin-bottom: 0.5rem; }}
    h2 {{ font-size: 2rem; margin-bottom: 0.75rem; }}
    h3 {{ font-size: 1.5rem; margin-bottom: 0.5rem; }}
    
    /* Glassmorphism Card */
    .card {{
        background: {colors["card_bg"]};
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-radius: 24px;
        padding: 2rem;
        border: 1px solid {colors["card_border"]};
        box-shadow: 0 8px 32px {colors["shadow"]};
        margin-bottom: 1.5rem;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }}
    
    .card::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, {colors["primary_color"]}, {colors["secondary_color"]}, {colors["accent_color"]});
        opacity: 0;
        transition: opacity 0.3s ease;
    }}
    
    .card:hover {{
        transform: translateY(-8px) scale(1.01);
        box-shadow: 0 20px 60px {colors["shadow"]};
        border-color: {colors["primary_color"]};
    }}
    
    .card:hover::before {{
        opacity: 1;
    }}
    
    /* Modern Input Fields - Enhanced */
    .stTextArea textarea {{
        background: rgba(20, 30, 48, 0.95) !important;
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 16px !important;
        border: 2px solid rgba(124, 140, 248, 0.4) !important;
        font-size: 16px !important;
        padding: 20px !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        color: #ffffff !important;
        font-family: 'Inter', sans-serif !important;
        line-height: 1.8 !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.05) !important;
        resize: vertical !important;
    }}
    
    .stTextArea textarea::placeholder {{
        color: rgba(230, 234, 242, 0.4) !important;
        font-style: italic !important;
    }}
    
    .stTextArea textarea:hover {{
        border: 2px solid rgba(124, 140, 248, 0.6) !important;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.05) !important;
    }}
    
    .stTextArea textarea:focus {{
        border: 2px solid {colors["primary_color"]} !important;
        box-shadow: 0 0 0 4px rgba(124, 140, 248, 0.15), 0 8px 24px rgba(124, 140, 248, 0.2) !important;
        transform: scale(1.005) !important;
        outline: none !important;
        background: rgba(20, 30, 48, 1) !important;
    }}
    
    /* Text area label */
    .stTextArea label {{
        font-weight: 500 !important;
        color: {colors["text_color"]} !important;
        margin-bottom: 0.5rem !important;
    }}
    
    /* Fix text visibility in all inputs */
    input {{
        color: {colors["text_color"]} !important;
    }}
    
    textarea {{
        color: {colors["text_color"]} !important;
    }}
    
    /* Modern Button with Gradient */
    .stButton > button {{
        background: linear-gradient(135deg, {colors["primary_color"]} 0%, {colors["secondary_color"]} 100%);
        color: white;
        border: none;
        border-radius: 16px;
        font-weight: 600;
        font-size: 16px;
        padding: 14px 32px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 16px rgba(102, 126, 234, 0.3);
        font-family: 'Inter', sans-serif;
        letter-spacing: 0.02em;
        text-transform: none;
    }}
    
    .stButton > button:hover {{
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 8px 24px rgba(102, 126, 234, 0.4);
        background: linear-gradient(135deg, {colors["secondary_color"]} 0%, {colors["primary_color"]} 100%);
    }}
    
    .stButton > button:active {{
        transform: translateY(-1px);
    }}
    
    /* Sentiment Result Boxes - Enhanced */
    .sentiment-positive {{
        background: {colors["positive_bg"]};
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        color: {colors["positive_text"]};
        padding: 2rem;
        border-radius: 20px;
        border: 2px solid {colors["positive_border"]};
        margin: 1.5rem 0;
        box-shadow: 0 8px 32px {colors["positive_glow"]}, inset 0 0 0 1px rgba(255, 255, 255, 0.1);
        transition: all 0.4s ease;
        position: relative;
        overflow: hidden;
    }}
    
    .sentiment-positive::before {{
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, {colors["positive_glow"]} 0%, transparent 70%);
        opacity: 0.3;
        animation: pulse 3s ease-in-out infinite;
    }}
    
    .sentiment-negative {{
        background: {colors["negative_bg"]};
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        color: {colors["negative_text"]};
        padding: 2rem;
        border-radius: 20px;
        border: 2px solid {colors["negative_border"]};
        margin: 1.5rem 0;
        box-shadow: 0 8px 32px {colors["negative_glow"]}, inset 0 0 0 1px rgba(255, 255, 255, 0.1);
        transition: all 0.4s ease;
        position: relative;
        overflow: hidden;
    }}
    
    .sentiment-negative::before {{
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, {colors["negative_glow"]} 0%, transparent 70%);
        opacity: 0.3;
        animation: pulse 3s ease-in-out infinite;
    }}
    
    .sentiment-neutral {{
        background: {colors["neutral_bg"]};
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        color: {colors["neutral_text"]};
        padding: 2rem;
        border-radius: 20px;
        border: 2px solid {colors["neutral_border"]};
        margin: 1.5rem 0;
        box-shadow: 0 8px 32px {colors["neutral_glow"]}, inset 0 0 0 1px rgba(255, 255, 255, 0.1);
        transition: all 0.4s ease;
        position: relative;
        overflow: hidden;
    }}
    
    .sentiment-neutral::before {{
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, {colors["neutral_glow"]} 0%, transparent 70%);
        opacity: 0.3;
        animation: pulse 3s ease-in-out infinite;
    }}
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {{
        background: {colors["card_bg"]} !important;
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-right: 1px solid {colors["card_border"]} !important;
    }}
    
    section[data-testid="stSidebar"] * {{
        color: {colors["text_color"]} !important;
    }}
    
    section[data-testid="stSidebar"] .stButton > button {{
        width: 100%;
        background: {colors["input_bg"]} !important;
        color: {colors["text_color"]} !important;
        border: 1px solid {colors["input_border"]} !important;
        transition: all 0.3s ease;
        font-size: 1.5rem !important;
        padding: 0.75rem !important;
    }}
    
    section[data-testid="stSidebar"] .stButton > button:hover {{
        background: linear-gradient(135deg, {colors["primary_color"]} 0%, {colors["secondary_color"]} 100%) !important;
        color: white !important;
        border-color: transparent !important;
        transform: scale(1.05) !important;
    }}
    
    /* Metrics Cards - Simple card design */
    .metric-card {{
        background: {colors["card_bg"]};
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid {colors["card_border"]};
        transition: all 0.3s ease;
    }}
    
    .metric-card:hover {{
        border-color: {colors["primary_color"]};
        transform: translateY(-2px);
    }}
    
    .metric-value {{
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, {colors["primary_color"]}, {colors["secondary_color"]});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
    }}
    
    .metric-label {{
        font-size: 0.875rem;
        color: {colors["text_color"]};
        opacity: 0.7;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}
    
    /* Enhanced Confidence Gauge */
    .confidence-gauge {{
        width: 100%;
        height: 12px;
        background: {colors["input_bg"]};
        border-radius: 10px;
        overflow: hidden;
        margin: 8px 0;
        box-shadow: inset 0 2px 4px {colors["shadow"]};
        position: relative;
    }}
    
    .confidence-fill {{
        height: 100%;
        border-radius: 10px;
        background: linear-gradient(90deg, {colors["primary_color"]}, {colors["secondary_color"]});
        box-shadow: 0 0 10px rgba(102, 126, 234, 0.5);
        transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
    }}
    
    .confidence-fill::after {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
        animation: shimmer 2s infinite;
    }}
    
    /* Expander - Simple card design */
    div[data-testid="stExpander"],
    .streamlit-expanderHeader,
    details summary {{
        font-size: 0.95rem !important;
        font-weight: 500 !important;
        background: {colors["card_bg"]} !important;
        border-radius: 12px !important;
        border: 1px solid {colors["card_border"]} !important;
        padding: 0.875rem 1rem !important;
        transition: all 0.3s ease !important;
        color: {colors["text_color"]} !important;
    }}
    
    div[data-testid="stExpander"]:hover,
    .streamlit-expanderHeader:hover,
    details summary:hover {{
        border-color: {colors["primary_color"]} !important;
    }}
    
    .streamlit-expanderContent,
    details[open] {{
        background: {colors["input_bg"]} !important;
        border: 1px solid {colors["card_border"]} !important;
        border-top: none !important;
        border-radius: 0 0 12px 12px !important;
        padding: 1rem !important;
        color: #ffffff !important;
    }}
    
    /* Fix all expander text to be white */
    div[data-testid="stExpander"] *,
    .streamlit-expanderContent *,
    details * {{
        color: #ffffff !important;
    }}
    
    /* Code blocks - Enhanced contrast */
    code {{
        background: rgba(15, 22, 39, 0.95) !important;
        color: #ffffff !important;
        padding: 0.2rem 0.4rem !important;
        border-radius: 4px !important;
        border: 1px solid rgba(124, 140, 248, 0.2) !important;
    }}
    
    pre {{
        background: rgba(15, 22, 39, 0.95) !important;
        color: #ffffff !important;
        border: 1px solid {colors["card_border"]} !important;
        border-radius: 8px !important;
        padding: 1rem !important;
    }}
    
    pre code {{
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
    }}
    
    /* Animations */
    @keyframes fadeIn {{
        from {{ 
            opacity: 0; 
            transform: translateY(20px);
        }}
        to {{ 
            opacity: 1; 
            transform: translateY(0);
        }}
    }}
    
    @keyframes pulse {{
        0%, 100% {{ opacity: 0.3; }}
        50% {{ opacity: 0.5; }}
    }}
    
    @keyframes shimmer {{
        0% {{ transform: translateX(-100%); }}
        100% {{ transform: translateX(100%); }}
    }}
    
    @keyframes float {{
        0%, 100% {{ transform: translateY(0px); }}
        50% {{ transform: translateY(-10px); }}
    }}
    
    .fade-in {{
        animation: fadeIn 0.6s cubic-bezier(0.4, 0, 0.2, 1) forwards;
    }}
    
    /* Progress Bar */
    .stProgress > div > div > div > div {{
        background: linear-gradient(90deg, {colors["primary_color"]}, {colors["secondary_color"]}) !important;
    }}
    
    /* Spinners */
    .stSpinner > div {{
        border-top-color: {colors["primary_color"]} !important;
    }}
    
    /* Alerts and Messages */
    .stAlert {{
        background: {colors["card_bg"]} !important;
        color: {colors["text_color"]} !important;
        border: 1px solid {colors["card_border"]} !important;
    }}
    
    /* Markdown text */
    .stMarkdown {{
        color: {colors["text_color"]} !important;
    }}
    
    .stMarkdown p {{
        color: {colors["text_color"]} !important;
    }}
    
    .stMarkdown div {{
        color: {colors["text_color"]} !important;
    }}
    
    /* Labels */
    label {{
        color: {colors["text_color"]} !important;
    }}
    
    /* All paragraph text */
    p {{
        color: {colors["text_color"]} !important;
    }}
    
    /* All div text */
    div {{
        color: {colors["text_color"]};
    }}
    
    /* All spans */
    span {{
        color: inherit;
    }}
    
    /* Select boxes and inputs */
    .stSelectbox > div > div {{
        background: {colors["input_bg"]} !important;
        color: {colors["text_color"]} !important;
        border-color: {colors["input_border"]} !important;
    }}
    
    /* Warnings */
    .stWarning {{
        background: {colors["neutral_bg"]} !important;
        border-left: 4px solid {colors["neutral_border"]} !important;
        color: {colors["text_color"]} !important;
        border-radius: 8px !important;
        padding: 1rem !important;
    }}
    
    /* Errors */
    .stError {{
        background: {colors["negative_bg"]} !important;
        border-left: 4px solid {colors["negative_border"]} !important;
        color: {colors["text_color"]} !important;
        border-radius: 8px !important;
        padding: 1rem !important;
    }}
    
    /* Success */
    .stSuccess {{
        background: {colors["positive_bg"]} !important;
        border-left: 4px solid {colors["positive_border"]} !important;
        color: {colors["text_color"]} !important;
        border-radius: 8px !important;
        padding: 1rem !important;
    }}
    
    /* Info messages */
    .stInfo {{
        background: {colors["card_bg"]} !important;
        border-left: 4px solid {colors["primary_color"]} !important;
        color: {colors["text_color"]} !important;
        border-radius: 8px !important;
        padding: 1rem !important;
    }}
    
    /* Responsive Design */
    @media (max-width: 768px) {{
        /* Adjust sidebar for mobile */
        section[data-testid="stSidebar"] {{
            min-width: 20rem !important;
            max-width: 20rem !important;
        }}
        
        .card {{
            padding: 1.25rem;
            border-radius: 16px;
        }}
        
        h1 {{ font-size: 2rem; }}
        h2 {{ font-size: 1.5rem; }}
        h3 {{ font-size: 1.25rem; }}
        
        .metric-value {{ font-size: 2rem; }}
        
        .stButton > button {{
            padding: 12px 24px;
            font-size: 15px;
        }}
    }}
    
    /* Wider screens - make sidebar even more spacious */
    @media (min-width: 1400px) {{
        section[data-testid="stSidebar"] {{
            min-width: 28rem !important;
            max-width: 28rem !important;
        }}
    }}
    
    /* Scrollbar Styling */
    ::-webkit-scrollbar {{
        width: 10px;
        height: 10px;
    }}
    
    ::-webkit-scrollbar-track {{
        background: {colors["input_bg"]};
        border-radius: 10px;
    }}
    
    ::-webkit-scrollbar-thumb {{
        background: linear-gradient(135deg, {colors["primary_color"]}, {colors["secondary_color"]});
        border-radius: 10px;
    }}
    
    ::-webkit-scrollbar-thumb:hover {{
        background: linear-gradient(135deg, {colors["secondary_color"]}, {colors["primary_color"]});
    }}
    </style>
    """


# Apply custom CSS
st.markdown(get_custom_css(), unsafe_allow_html=True)


@st.cache_resource
def load_model_and_artifacts():
    """Load the trained model, tokenizer, and label encoder"""
    try:
        # Load model
        model = tf.keras.models.load_model(
            MODEL_PATH, custom_objects={"AttentionLayer": AttentionLayer}
        )

        # Load tokenizer
        with open(TOKENIZER_PATH, "rb") as f:
            tokenizer = pickle.load(f)

        # Load label encoder
        with open(LABEL_ENCODER_PATH, "rb") as f:
            label_encoder = pickle.load(f)

        return model, tokenizer, label_encoder
    except Exception as e:
        # Get translation if available, otherwise use default English
        error_msg = TRANSLATIONS.get(
            st.session_state.get("language", "en"), TRANSLATIONS["en"]
        ).get("error_prediction", "Error loading model artifacts:")
        st.error(f"{error_msg} {str(e)}")
        return None, None, None


# Define Attention Layer (same as in training)
class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name="att_weight",
            shape=(input_shape[-1],),
            initializer="random_normal",
            trainable=True,
        )
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs, mask=None):
        e = tf.keras.backend.tanh(inputs)
        e = tf.keras.backend.dot(e, tf.keras.backend.expand_dims(self.W, -1))
        e = tf.keras.backend.squeeze(e, -1)

        if mask is not None:
            mask = tf.keras.backend.cast(mask, tf.keras.backend.floatx())
            e = e * mask + ((1 - mask) * -1e9)

        alpha = tf.keras.backend.softmax(e)
        alpha_exp = tf.keras.backend.expand_dims(alpha, axis=-1)
        context = tf.keras.backend.sum(inputs * alpha_exp, axis=1)
        return context

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def get_config(self):
        config = super().get_config()
        return config


def clean_text(text):
    """Clean and preprocess text (same as training pipeline)"""
    text = str(text)
    text = text.lower()
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    # Remove mentions
    text = re.sub(r"@\w+", " ", text)
    # Remove hashtags
    text = re.sub(r"#", " ", text)
    # Remove non-alphanumeric characters
    text = re.sub(r"[^0-9a-z\s]", " ", text)
    # Collapse spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_text(text, tokenizer):
    """Preprocess text for model prediction"""
    # Clean text
    cleaned = clean_text(text)

    # Tokenize
    sequence = tokenizer.texts_to_sequences([cleaned])

    # Pad sequence
    padded = pad_sequences(sequence, maxlen=MAX_LEN, padding="post", truncating="post")

    return padded, cleaned


def translate_sentiment(indonesian_label):
    """Translate Indonesian sentiment labels based on current language"""
    if st.session_state.language == "en":
        translation = {
            "Positif": "Positive",
            "Netral": "Neutral",
            "Negatif": "Negative",
        }
        return translation.get(indonesian_label, indonesian_label)
    else:
        # Keep Indonesian labels as-is for Indonesian language
        return indonesian_label


def predict_sentiment(text, model, tokenizer, label_encoder):
    """Predict sentiment for given text"""
    # Preprocess
    processed_text, cleaned_text = preprocess_text(text, tokenizer)

    # Predict
    prediction = model.predict(processed_text, verbose=0)

    # Get predicted class and confidence
    predicted_class_idx = np.argmax(prediction[0])
    confidence = prediction[0][predicted_class_idx]

    # Decode label (Indonesian)
    sentiment_indonesian = label_encoder.classes_[predicted_class_idx]

    # Translate to English
    sentiment = translate_sentiment(sentiment_indonesian)

    # Get all class probabilities (translated to English)
    all_probabilities = {
        translate_sentiment(label_encoder.classes_[i]): float(prediction[0][i])
        for i in range(len(label_encoder.classes_))
    }

    return sentiment, confidence, all_probabilities, cleaned_text


def display_metric_card(label, value, unit=""):
    """Display a metric in a styled card"""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-value">{value}{unit}</div>
            <div class="metric-label">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def display_confidence_gauge(label, value, color):
    """Display a confidence gauge with label"""
    st.markdown(
        f"""
        <div style="margin-bottom: 1rem;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                <span style="font-weight: 500;">{label}</span>
                <span>{value*100:.1f}%</span>
            </div>
            <div class="confidence-gauge">
                <div class="confidence-fill" style="width: {value*100}%; background-color: {color};"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    # Load model and artifacts
    with st.spinner(get_text("loading_model")):
        model, tokenizer, label_encoder = load_model_and_artifacts()

    if model is None:
        st.error(get_text("error_loading"))
        return

    # Get theme colors
    colors = get_theme_colors()

    # Sidebar with info
    with st.sidebar:
        # Language selector at the top
        st.markdown(
            f"""
            <div style="text-align: center; margin: 1rem 0 1rem 0;">
                <h2 style="font-size: 1.3rem; margin-bottom: 0; background: linear-gradient(135deg, {colors["primary_color"]}, {colors["secondary_color"]}); 
                           -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">
                    🌐 {get_text('language')}
                </h2>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Language selection buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button(
                "🇬🇧 English",
                use_container_width=True,
                type="primary" if st.session_state.language == "en" else "secondary",
            ):
                st.session_state.language = "en"
                st.rerun()
        with col2:
            if st.button(
                "🇮🇩 Indonesia",
                use_container_width=True,
                type="primary" if st.session_state.language == "id" else "secondary",
            ):
                st.session_state.language = "id"
                st.rerun()

        st.markdown("<div style='margin: 1.5rem 0;'></div>", unsafe_allow_html=True)

        # About section
        st.markdown(
            f"""
            <div style="text-align: center; margin: 1rem 0 1.5rem 0;">
                <h2 style="font-size: 1.5rem; margin-bottom: 0; background: linear-gradient(135deg, {colors["primary_color"]}, {colors["secondary_color"]}); 
                           -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">
                    📖 {get_text('about')}
                </h2>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # About section with modern styling
        st.markdown(
            f"""
            <div class="card">
                <h3 style="margin-top: 0; font-size: 1.25rem; margin-bottom: 1rem;">{get_text('about_title')}</h3>
                <p style="font-size: 0.9rem; line-height: 1.6; margin-bottom: 1rem;">
                    {get_text('about_text')}
                </p>
                <div style="margin: 1rem 0;">
                    <div style="display: flex; align-items: center; margin: 0.75rem 0;">
                        <span style="display: inline-block; width: 12px; height: 12px; border-radius: 50%; 
                                     background: {colors["positive_border"]}; margin-right: 0.75rem; 
                                     box-shadow: 0 0 10px {colors["positive_border"]}40;"></span>
                        <strong style="color: {colors["positive_border"]};">{get_text('positive')}</strong>
                    </div>
                    <div style="display: flex; align-items: center; margin: 0.75rem 0;">
                        <span style="display: inline-block; width: 12px; height: 12px; border-radius: 50%; 
                                     background: {colors["neutral_border"]}; margin-right: 0.75rem;
                                     box-shadow: 0 0 10px {colors["neutral_border"]}40;"></span>
                        <strong style="color: {colors["neutral_border"]};">{get_text('neutral')}</strong>
                    </div>
                    <div style="display: flex; align-items: center; margin: 0.75rem 0;">
                        <span style="display: inline-block; width: 12px; height: 12px; border-radius: 50%; 
                                     background: {colors["negative_border"]}; margin-right: 0.75rem;
                                     box-shadow: 0 0 10px {colors["negative_border"]}40;"></span>
                        <strong style="color: {colors["negative_border"]};">{get_text('negative')}</strong>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Model info with badges
        st.markdown(
            f"""
            <div class="card">
                <h3 style="margin-top: 0; font-size: 1.25rem; margin-bottom: 1rem;">🤖 {get_text('model_info')}</h3>
                <div style="margin: 1rem 0;">
                    <div style="margin-bottom: 1rem;">
                        <div style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; 
                                    opacity: 0.7; margin-bottom: 0.25rem;">{get_text('architecture')}</div>
                        <div style="background: linear-gradient(135deg, {colors["primary_color"]}20, {colors["secondary_color"]}20);
                                    border: 1px solid {colors["primary_color"]}40; border-radius: 8px; 
                                    padding: 0.5rem; font-weight: 600; font-size: 0.9rem;">
                            CNN + BiLSTM + Attention
                        </div>
                    </div>
                    <div style="margin-bottom: 1rem;">
                        <div style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; 
                                    opacity: 0.7; margin-bottom: 0.25rem;">{get_text('accuracy')}</div>
                        <div style="background: linear-gradient(135deg, {colors["secondary_color"]}20, {colors["accent_color"]}20);
                                    border: 1px solid {colors["secondary_color"]}40; border-radius: 8px; 
                                    padding: 0.5rem; font-weight: 600; font-size: 0.9rem;">
                            74.97% ⭐
                        </div>
                    </div>
                    <div>
                        <div style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; 
                                    opacity: 0.7; margin-bottom: 0.25rem;">{get_text('dataset_size')}</div>
                        <div style="background: linear-gradient(135deg, {colors["accent_color"]}20, {colors["primary_color"]}20);
                                    border: 1px solid {colors["accent_color"]}40; border-radius: 8px; 
                                    padding: 0.5rem; font-weight: 600; font-size: 0.9rem;">
                            26,852 {get_text('samples')}
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Example section
        st.markdown(
            f"""
            <div class="card">
                <h3 style="margin-top: 0; font-size: 1.25rem; margin-bottom: 0.75rem;">💡 {get_text('try_examples')}</h3>
                <p style="font-size: 0.85rem; opacity: 0.7; margin-bottom: 1rem;">
                    {get_text('input_subtitle').split('.')[0]}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("😊", use_container_width=True, key="positive_btn"):
                st.session_state.example_text = get_text("example_positive")
                st.rerun()
        with col2:
            if st.button("😐", use_container_width=True, key="neutral_btn"):
                st.session_state.example_text = get_text("example_neutral")
                st.rerun()
        with col3:
            if st.button("😞", use_container_width=True, key="negative_btn"):
                st.session_state.example_text = get_text("example_negative")
                st.rerun()

    # Main content area
    st.markdown(
        f"""
        <div style="text-align: center; margin: 2rem 0 3rem 0; position: relative;">
            <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); 
                        width: 400px; height: 400px; background: radial-gradient(circle, {colors["primary_color"]}15 0%, transparent 70%); 
                        filter: blur(60px); z-index: 0;"></div>
            <h1 style="font-size: 3rem; font-weight: 800; margin-bottom: 1rem; position: relative; z-index: 1;">
                <span style="background: linear-gradient(135deg, {colors["primary_color"]}, {colors["secondary_color"]}, {colors["accent_color"]}); 
                      -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                      background-clip: text; display: inline-block;">
                    {get_text('title')}
                </span>
            </h1>
            <p style="font-size: 1.125rem; margin-bottom: 0.5rem; color: {colors["text_color"]}; 
                      opacity: 0.9; font-weight: 500; position: relative; z-index: 1;">
                {get_text('subtitle')}
            </p>
            <p style="font-size: 0.95rem; color: {colors["text_color"]}; 
                      opacity: 0.6; max-width: 600px; margin: 0 auto; position: relative; z-index: 1;">
                {get_text('description')}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Main container
    with st.container():
        # Use example text if available
        default_text = st.session_state.get("example_text", "")

        # Input area with modern styling
        st.markdown(
            f"""
            <h3 style="margin-top: 0; font-size: 1.5rem; margin-bottom: 0.5rem; display: flex; align-items: center; gap: 0.5rem;">
                <span>{get_text('input_title')}</span>
            </h3>
            <p style="font-size: 0.95rem; opacity: 0.7; margin-bottom: 1rem;">
                {get_text('input_subtitle')}
            </p>
            """,
            unsafe_allow_html=True,
        )

        user_input = st.text_area(
            get_text("input_label"),
            value=default_text,
            height=200,
            placeholder=get_text("input_placeholder"),
            label_visibility="collapsed",
        )

        # Analyze button with modern styling
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            analyze_button = st.button(
                get_text("analyze_button"), use_container_width=True, type="primary"
            )

    # Perform analysis
    if analyze_button:
        # Clear example text from session state after button click
        if "example_text" in st.session_state:
            del st.session_state.example_text

        if not user_input.strip():
            st.warning(get_text("empty_warning"))
        else:
            with st.spinner(get_text("analyzing")):
                sentiment, confidence, all_probs, cleaned_text = predict_sentiment(
                    user_input, model, tokenizer, label_encoder
                )

            # Display results with modern design
            st.markdown(
                f"""
                <h2 style="margin-top: 2rem; font-size: 2rem; margin-bottom: 1.5rem;">
                    {get_text('results_title')}
                </h2>
                """,
                unsafe_allow_html=True,
            )

            # Get translated sentiment for comparison
            translated_positive = get_text("positive")
            translated_negative = get_text("negative")
            translated_neutral = get_text("neutral")

            # Sentiment box with color and better icons
            sentiment_class = (
                "sentiment-positive"
                if sentiment == translated_positive
                else (
                    "sentiment-negative"
                    if sentiment == translated_negative
                    else "sentiment-neutral"
                )
            )

            sentiment_icon = (
                "🎉"
                if sentiment == translated_positive
                else "😔" if sentiment == translated_negative else "🤔"
            )

            st.markdown(
                f"""
                <div class="{sentiment_class}" style="position: relative;">
                    <div style="display: flex; justify-content: space-between; align-items: center; position: relative; z-index: 1;">
                        <div style="display: flex; align-items: center; gap: 1rem;">
                            <span style="font-size: 3.5rem; line-height: 1;">{sentiment_icon}</span>
                            <div>
                                <div style="font-size: 0.875rem; text-transform: uppercase; letter-spacing: 0.1em; 
                                            opacity: 0.8; margin-bottom: 0.25rem;">{get_text('sentiment_label')}</div>
                                <h2 style="margin: 0; font-size: 2.5rem; font-weight: 800;">{sentiment}</h2>
                            </div>
                        </div>
                        <div style="text-align: right;">
                            <div style="font-size: 0.875rem; text-transform: uppercase; letter-spacing: 0.1em; 
                                        opacity: 0.8; margin-bottom: 0.25rem;">{get_text('confidence_label')}</div>
                            <div style="font-size: 2.5rem; font-weight: 800;">
                                {confidence*100:.1f}<span style="font-size: 1.5rem; opacity: 0.8;">%</span>
                            </div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Confidence breakdown with modern design
            st.markdown(
                f"""
                <h3 style="font-size: 1.5rem; margin-top: 2rem; margin-bottom: 1.25rem; display: flex; align-items: center; gap: 0.5rem;">
                    <span style="font-size: 1.5rem;">📈</span>
                    <span>{get_text('distribution_title')}</span>
                </h3>
                """,
                unsafe_allow_html=True,
            )

            # Sort probabilities
            sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)

            # Set colors and emojis for each sentiment (supporting both languages)
            sentiment_colors = {
                translated_positive: colors["positive_border"],
                translated_neutral: colors["neutral_border"],
                translated_negative: colors["negative_border"],
            }

            sentiment_emojis = {
                translated_positive: "😊",
                translated_neutral: "😐",
                translated_negative: "😞",
            }

            # Display confidence gauges with emojis
            for label, prob in sorted_probs:
                st.markdown(
                    f"""
                    <div style="margin-bottom: 1.25rem;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                            <div style="display: flex; align-items: center; gap: 0.5rem;">
                                <span style="font-size: 1.25rem;">{sentiment_emojis[label]}</span>
                                <span style="font-weight: 600; font-size: 1rem;">{label}</span>
                            </div>
                            <span style="font-weight: 700; font-size: 1.125rem; color: {sentiment_colors[label]};">
                                {prob*100:.1f}%
                            </span>
                        </div>
                        <div class="confidence-gauge">
                            <div class="confidence-fill" style="width: {prob*100}%; background: linear-gradient(90deg, {sentiment_colors[label]}, {sentiment_colors[label]}dd);"></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Text statistics - Simplified design
            st.markdown(
                f"""
                <h3 style="font-size: 1.5rem; margin-top: 2.5rem; margin-bottom: 1rem;">
                    📝 {get_text('statistics_title')}
                </h3>
                """,
                unsafe_allow_html=True,
            )

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-value">{len(user_input.split())}</div>
                        <div class="metric-label">{get_text('word_count')}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with col2:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-value">{len(user_input)}</div>
                        <div class="metric-label">{get_text('characters')}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with col3:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-value">{len(cleaned_text.split())}</div>
                        <div class="metric-label">{get_text('cleaned_words')}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Show cleaned text with better styling
            st.markdown(
                "<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True
            )
            with st.expander(get_text("view_preprocessed")):
                st.code(cleaned_text, language=None)

    # Footer with modern design
    st.markdown(
        f"""
        <div style="text-align: center; padding: 3rem 0 2rem 0; margin-top: 4rem; 
                    border-top: 1px solid {colors["card_border"]};">
            <div style="display: flex; justify-content: center; align-items: center; gap: 1rem; flex-wrap: wrap; margin-bottom: 1rem;">
                <span style="font-size: 1.25rem;">🚀</span>
                <span style="font-weight: 600; font-size: 1rem;">{get_text('powered_by')}</span>
                <span style="background: linear-gradient(135deg, {colors["primary_color"]}, {colors["secondary_color"]}); 
                             -webkit-background-clip: text; -webkit-text-fill-color: transparent; 
                             background-clip: text; font-weight: 700;">
                    TensorFlow
                </span>
                <span style="opacity: 0.5;">•</span>
                <span style="background: linear-gradient(135deg, {colors["secondary_color"]}, {colors["accent_color"]}); 
                             -webkit-background-clip: text; -webkit-text-fill-color: transparent; 
                             background-clip: text; font-weight: 700;">
                    Streamlit
                </span>
            </div>
            <p style="font-size: 0.875rem; opacity: 0.7; margin: 0.5rem 0;">
                {get_text('model_description')}
            </p>
            <p style="font-size: 0.75rem; opacity: 0.5; margin-top: 1rem;">
                {get_text('copyright')}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
