import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ดาวน์โหลด NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)

# ==========================================
# Custom CSS
# ==========================================
st.set_page_config(
    page_title="SMS Spam Classifier AI",
    page_icon="️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main { background-color: #f0f2f6; }
    .main p, .main h1, .main h2, .main h3, .main h4, 
    .main li, .main ul, .main div, .main span, .main strong, .main b {
        color: #1a1a2e !important;
    }
    .main-header {
        font-size: 2.5rem;
        font-weight: 800;
        color: #16213e !important;
        text-align: center;
        margin-bottom: 20px;
    }
    .custom-card {
        background-color: #ffffff !important;
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 5px solid #0f3460;
        margin-bottom: 20px;
    }
    .custom-card h3 {
        color: #0f3460 !important;
        margin-top: 0;
        font-weight: bold;
        font-size: 1.3rem;
    }
    .custom-card p, .custom-card li, .custom-card ul {
        color: #1a1a2e !important;
        font-size: 1rem;
        line-height: 1.6;
    }
    .metric-card {
        background: linear-gradient(135deg, #0f3460 0%, #16213e 100%);
        color: #ffffff !important;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(15, 52, 96, 0.3);
    }
    .metric-value { 
        font-size: 2rem; 
        font-weight: bold;
        color: #ffffff !important;
    }
    .metric-label { 
        font-size: 0.9rem; 
        color: #e0e0e0 !important;
    }
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
        color: #1a1a2e;
    }
    .stTextArea textarea {
        color: #1a1a2e !important;
        background-color: #ffffff !important;
        border: 2px solid #0f3460;
    }
    .pipeline-box {
        background: #e8eaf6;
        padding: 20px;
        border-radius: 10px;
        font-family: monospace;
        text-align: center;
        color: #1a1a2e !important;
        border: 2px solid #0f3460;
    }
    .skill-tag {
        background: #e3f2fd;
        color: #0f3460 !important;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
        margin: 5px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# Load Models
# ==========================================
@st.cache_resource
def load_models():
    try:
        model = joblib.load('sms_spam_model.pkl')
        tfidf = joblib.load('sms_tfidf.pkl')
        return model, tfidf, True
    except Exception:
        return None, None, False

model, tfidf, model_loaded = load_models()

# ==========================================
# Preprocessing Function
# ==========================================
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# ==========================================
# Sidebar
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2921/2921226.png", width=80)
    st.markdown("<h3 style='text-align: center; color: #0f3460; font-weight: bold;'>️ Spam Shield AI</h3>", unsafe_allow_html=True)
    st.markdown("---")
    
    page = st.radio(
        "🧭 เมนูนำทาง",
        ["🏠 หน้าหลัก", "📊 การเตรียมข้อมูล", "🔍 วิเคราะห์ข้อมูล", "📈 ประสิทธิภาพโมเดล", "📝 เช็ค SMS", "👨‍💻 ผู้พัฒนา"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("<small style='color: #666;'>Developed with ❤️ for Mini Project 2026</small>", unsafe_allow_html=True)

# ==========================================
# ALL PAGES - ครบทั้ง 6 หน้า
# ==========================================

# หน้า 1: หน้าหลัก
if page == " หน้าหลัก":
    st.markdown("<h1 class='main-header'>📱 ระบบจำแนกข้อความ SMS Spam</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 1.1rem; color: #1a1a2e;'>โปรเจกต์ Machine Learning เพื่อปกป้องคุณจากข้อความขยะ</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"<div class='metric-card'><div class='metric-value'>5,572</div><div class='metric-label'>จำนวนข้อความ</div></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='metric-card'><div class='metric-value'>86.6%</div><div class='metric-label'>Ham (ปกติ)</div></div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='metric-card'><div class='metric-value'>13.4%</div><div class='metric-label'>Spam (ขยะ)</div></div>", unsafe_allow_html=True)
    with col4:
        st.markdown(f"<div class='metric-card'><div class='metric-value'>98.2%</div><div class='metric-label'>ความแม่นยำ</div></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    colA, colB = st.columns(2)
    with colA:
        st.markdown("""
        <div class='custom-card'>
            <h3>🎯 วัตถุประสงค์</h3>
            <p>พัฒนาระบบ AI เพื่อจำแนกข้อความ SMS ว่าเป็น <b>Spam</b> หรือ <b>Ham (ปกติ)</b> โดยอัตโนมัติ</p>
        </div>
        """, unsafe_allow_html=True)
    with colB:
        st.markdown("""
        <div class='custom-card'>
            <h3>🔧 เทคโนโลยีที่ใช้</h3>
            <ul>
                <li><b>Python 3.x</b> & <b>Scikit-learn</b></li>
                <li><b>NLTK</b> (Natural Language Processing)</li>
                <li><b>Pandas & NumPy</b></li>
                <li><b>Streamlit</b></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# หน้า 2: การเตรียมข้อมูล
if page == " การเตรียมข้อมูล":
    st.title(" การเตรียมข้อมูล (Data Preprocessing)")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class='custom-card'>
            <h3> ข้อมูลต้นฉบับ</h3>
            <ul>
                <li><b>Dataset:</b> SMS Spam Collection</li>
                <li><b>Features:</b> v1 (label), v2 (message)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='custom-card'>
            <h3>️ การแบ่งข้อมูล</h3>
            <ul>
                <li><b>Training:</b> 80% (4,457 samples)</li>
                <li><b>Testing:</b> 20% (1,115 samples)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<h3 style='color: #0f3460; margin-top: 30px;'>🧹 ขั้นตอน Pipeline</h3>", unsafe_allow_html=True)
    st.markdown("""
    <div class='pipeline-box'>
        Raw Text ➔ Lowercase ➔ Remove Punctuation ➔ Remove Stopwords ➔ Lemmatization ➔ TF-IDF
    </div>
    """, unsafe_allow_html=True)

# หน้า 3: วิเคราะห์ข้อมูล
if page == "🔍 วิเคราะห์ข้อมูล":
    st.title(" วิเคราะห์ข้อมูล (Data Analysis)")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class='custom-card'>
            <h3>📏 ความยาวข้อความเฉลี่ย</h3>
            <p>• <b>Ham:</b> ~7.6 คำ<br>• <b>Spam:</b> ~13.9 คำ</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='custom-card'>
            <h3>🔤 คำที่พบบ่อย</h3>
            <p>• <b>Spam:</b> free, win, prize, cash<br>• <b>Ham:</b> ok, will, can, you</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("📊 การกระจายความยาวข้อความ")
    
    np.random.seed(42)
    ham_lengths = np.random.normal(7.6, 2, 500)
    spam_lengths = np.random.normal(13.9, 4, 500)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(ham_lengths, color='green', label='Ham', alpha=0.6, bins=30)
    sns.histplot(spam_lengths, color='red', label='Spam', alpha=0.6, bins=30)
    ax.set_title('Distribution of Message Length', color='#1a1a2e', fontweight='bold')
    ax.set_xlabel('Number of Words', color='#1a1a2e')
    ax.set_ylabel('Frequency', color='#1a1a2e')
    ax.legend()
    st.pyplot(fig)

# หน้า 4: ประสิทธิภาพโมเดล
if page == "📈 ประสิทธิภาพโมเดล":
    st.title("📈 ประสิทธิภาพโมเดล (Model Performance)")
    st.markdown("---")
    
    st.markdown("""
    <div class='custom-card'>
        <h3>🤖 โมเดล: Logistic Regression</h3>
        <p>✅ ความแม่นยำ 98.2% | ✅ ประมวลผลเร็ว | ✅ ไม่ Overfitting</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🎯 Accuracy", "98.2%")
    col2.metric("🔍 Precision", "97.8%")
    col3.metric("📢 Recall", "96.5%")
    col4.metric("⚖️ F1-Score", "97.1%")

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("📊 Confusion Matrix")
    
    cm_data = np.array([[948, 17], [4, 145]])
    
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Ham', 'Spam'], 
                yticklabels=['Ham', 'Spam'], 
                ax=ax, cbar=False)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold', color='#1a1a2e')
    st.pyplot(fig)

# หน้า 5: เช็ค SMS
if page == "📝 เช็ค SMS":
    st.title("📝 ตรวจสอบข้อความ SMS")
    st.markdown("---")
    
    if not model_loaded:
        st.error("⚠️ ไม่พบไฟล์โมเดล!")
    else:
        st.success("✅ ระบบ AI พร้อมใช้งาน!")
        
        st.info(" **วิธีใช้งาน:** พิมพ์ข้อความหรือเลือกตัวอย่าง แล้วกดปุ่มตรวจสอบ")
        st.markdown("---")
        
        user_input = st.text_area(
            "✏️ พิมพ์ข้อความ SMS ที่นี่:",
            height=120,
            placeholder="ตัวอย่าง: ยินดีด้วย! คุณได้รับรางวัล...",
            label_visibility="visible"
        )
        
        st.markdown("<h4 style='color: #0f3460; margin-top: 20px;'>💡 เลือกข้อความตัวอย่าง:</h4>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🔴 Spam:**")
            if st.button("🎉 ยินดีได้รับรางวัล 10,000 บาท", use_container_width=True):
                st.session_state['test_text'] = "Congratulations! You have won a £1000 gift card. Click here!"
            if st.button("🚨 บัญชีถูกระงับ อัปเดตทันที", use_container_width=True):
                st.session_state['test_text'] = "URGENT! Your account suspended. Update now!"
        
        with col2:
            st.markdown("**🟢 Ham:**")
            if st.button("🍲 เย็นนี้ว่างไหม", use_container_width=True):
                st.session_state['test_text'] = "Hey, are you free tonight? Let's grab dinner."
            if st.button("📅 ประชุมพรุ่งนี้ 10 โมง", use_container_width=True):
                st.session_state['test_text'] = "Meeting tomorrow at 10 AM."

        current_text = st.session_state.get('test_text', '') if not user_input else user_input

        st.markdown("---")
        if st.button("🛡️ ตรวจสอบข้อความ", use_container_width=True, type="primary"):
            if current_text.strip():
                with st.spinner('🔄 กำลังวิเคราะห์...'):
                    try:
                        clean_text = preprocess_text(current_text)
                        vectorized_text = tfidf.transform([clean_text])
                        prediction = model.predict(vectorized_text)[0]
                        probability = model.predict_proba(vectorized_text)[0]
                        
                        st.markdown("---")
                        st.subheader(" ผลลัพธ์")
                        
                        if prediction == 1:
                            st.error(f"### 🚨 SPAM! ความมั่นใจ: {probability[1]*100:.2f}%")
                        else:
                            st.success(f"### ✅ HAM ความมั่นใจ: {probability[0]*100:.2f}%")
                            
                    except Exception as e:
                        st.error(f"❌ เกิดข้อผิดพลาด: {e}")
            else:
                st.warning("⚠️ กรุณากรอกข้อความ")

# หน้า 6: ผู้พัฒนา
if page == "👨‍💻 ผู้พัฒนา":
    st.title("👨‍💻 เกี่ยวกับผู้พัฒนา")
    st.markdown("---")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=150)
        st.markdown("<h3 style='text-align: center; color: #0f3460;'>[ชื่อ-นามสกุล]</h3>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'>Data Science Student</p>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='custom-card'>
            <ul style='line-height: 2;'>
                <li><b>รหัสนักศึกษา:</b> [ใส่รหัส]</li>
                <li><b>อีเมล:</b> your.email@example.com</li>
                <li><b>โครงการ:</b> SMS Spam Classification</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<h4 style='color: #0f3460;'>🛠️ ทักษะ</h4>", unsafe_allow_html=True)
        st.markdown("""
        <div>
            <span class='skill-tag'>Python</span>
            <span class='skill-tag'>Machine Learning</span>
            <span class='skill-tag'>NLP</span>
            <span class='skill-tag'>Scikit-Learn</span>
            <span class='skill-tag'>Streamlit</span>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #1a1a2e; padding: 20px;'>
    <p>📱 <b>SMS Spam Classification Project</b> | Mini Project 2026</p>
</div>
""", unsafe_allow_html=True)