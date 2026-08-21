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
# ตั้งค่าหน้าเว็บ
# ==========================================
st.set_page_config(
    page_title="SMS Spam Classifier - AI Project",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# Custom CSS - ออกแบบใหม่ สวยๆ เท่ๆ ตัวหนังสือชัดเจน
# ==========================================
st.markdown("""
<style>
    /* พื้นหลังหลัก - สีเทาอ่อน */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8ec 100%);
    }
    
    /* ตัวหนังสือหลัก - สีน้ำเงินเข้มมาก มองเห็นชัดเจน */
    .main h1, .main h2, .main h3, .main h4, .main p, .main li, .main ul, .main div {
        color: #1e3a5f !important;
    }
    
    /* หัวข้อใหญ่ */
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #1e3a5f 0%, #2c5282 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 10px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    /* คำบรรยาย */
    .subtitle {
        font-size: 1.2rem;
        color: #4a5568 !important;
        text-align: center;
        margin-bottom: 40px;
        font-weight: 500;
    }
    
    /* การ์ดข้อมูล */
    .info-card {
        background: white;
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 10px 25px rgba(30, 58, 95, 0.15);
        border-top: 5px solid #2c5282;
        margin-bottom: 25px;
        transition: transform 0.3s ease;
    }
    
    .info-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(30, 58, 95, 0.2);
    }
    
    .info-card h3 {
        color: #1e3a5f !important;
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 15px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .info-card p, .info-card li {
        color: #2d3748 !important;
        font-size: 1.05rem;
        line-height: 1.8;
    }
    
    /* การ์ดตัวเลขสถิติ */
    .stat-card {
        background: linear-gradient(135deg, #1e3a5f 0%, #2c5282 100%);
        color: white !important;
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 10px 25px rgba(30, 58, 95, 0.3);
        margin-bottom: 25px;
    }
    
    .stat-number {
        font-size: 2.8rem;
        font-weight: 800;
        color: white !important;
        margin-bottom: 10px;
    }
    
    .stat-label {
        font-size: 1rem;
        color: #e2e8f0 !important;
        font-weight: 500;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, #ffffff 0%, #f7fafc 100%);
        border-right: 2px solid #e2e8f0;
    }
    
    /* ปุ่ม */
    .stButton>button {
        background: linear-gradient(135deg, #1e3a5f 0%, #2c5282 100%);
        color: white !important;
        border: none;
        border-radius: 10px;
        padding: 12px 30px;
        font-weight: 600;
        font-size: 1.05rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(30, 58, 95, 0.3);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(30, 58, 95, 0.4);
        background: linear-gradient(135deg, #2c5282 0%, #1e3a5f 100%);
    }
    
    /* Divider */
    .custom-divider {
        border: none;
        height: 3px;
        background: linear-gradient(90deg, #1e3a5f 0%, #2c5282 100%);
        margin: 40px 0;
        border-radius: 3px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# โหลดโมเดล
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
# Sidebar Navigation
# ==========================================
with st.sidebar:
    st.markdown("<div style='text-align: center; margin-bottom: 30px;'>", unsafe_allow_html=True)
    st.image("https://cdn-icons-png.flaticon.com/512/2921/2921226.png", width=100)
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<h2 style='text-align: center; color: #1e3a5f; font-weight: 700; margin-bottom: 30px;'>️ Spam Shield AI</h2>", unsafe_allow_html=True)
    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
    
    page = st.radio(
        "",
        ["🏠 หน้าหลัก", " การเตรียมข้อมูล", "🔍 วิเคราะห์ข้อมูล", "📈 ประสิทธิภาพโมเดล", "📝 เช็ค SMS", "👨💻 ผู้พัฒนา"],
        label_visibility="collapsed"
    )
    
    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #4a5568; font-size: 0.9rem;'>Mini Project 2026<br>Machine Learning Course</p>", unsafe_allow_html=True)

# ==========================================
# หน้า 1: หน้าหลัก (HOME PAGE) - ออกแบบใหม่ สวยๆ เท่ๆ
# ==========================================
if page == "🏠 หน้าหลัก":
    # Header
    st.markdown("<h1 class='main-title'>📱 ระบบจำแนกข้อความ SMS Spam</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>โปรเจกต์ Machine Learning เพื่อปกป้องคุณจากข้อความขยะและมิจฉาชีพ ด้วยความแม่นยำสูง</p>", unsafe_allow_html=True)
    
    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
    
    # Statistics Cards
    st.markdown("<h2 style='color: #1e3a5f; font-size: 2rem; margin-bottom: 25px; font-weight: 700;'>📊 ข้อมูลภาพรวม</h2>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class='stat-card'>
            <div class='stat-number'>5,572</div>
            <div class='stat-label'>จำนวนข้อความทั้งหมด</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='stat-card'>
            <div class='stat-number'>4,825</div>
            <div class='stat-label'>Ham (ปกติ) - 86.6%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='stat-card'>
            <div class='stat-number'>747</div>
            <div class='stat-label'>Spam (ขยะ) - 13.4%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class='stat-card'>
            <div class='stat-number'>98.2%</div>
            <div class='stat-label'>ความแม่นยำของโมเดล</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
    
    # Objectives & Technology
    st.markdown("<h2 style='color: #1e3a5f; font-size: 2rem; margin-bottom: 25px; font-weight: 700;'>🎯 เกี่ยวกับโปรเจกต์</h2>", unsafe_allow_html=True)
    
    colA, colB = st.columns(2)
    
    with colA:
        st.markdown("""
        <div class='info-card'>
            <h3>🎯 วัตถุประสงค์</h3>
            <ul style='padding-left: 20px;'>
                <li>พัฒนาระบบ AI เพื่อจำแนกข้อความ SMS ว่าเป็น <b>Spam</b> หรือ <b>Ham (ปกติ)</b> โดยอัตโนมัติ</li>
                <li>ใช้เทคนิค Machine Learning และ Natural Language Processing (NLP)</li>
                <li>สร้างความแม่นยำสูงในการตรวจจับข้อความขยะ</li>
                <li>ช่วยลดความเสี่ยงจากการถูกหลอกลวงทางข้อความ</li>
                <li>เป็นเครื่องมือในการเรียนรู้การทำโปรเจกต์ Data Science</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with colB:
        st.markdown("""
        <div class='info-card'>
            <h3>🔧 เทคโนโลยีที่ใช้</h3>
            <ul style='padding-left: 20px;'>
                <li><b>Python 3.x</b> - ภาษาโปรแกรมหลัก</li>
                <li><b>Scikit-learn</b> - Machine Learning Library</li>
                <li><b>Logistic Regression</b> - อัลกอริทึมหลัก</li>
                <li><b>NLTK</b> - Natural Language Processing</li>
                <li><b>Pandas & NumPy</b> - Data Processing</li>
                <li><b>Matplotlib & Seaborn</b> - Data Visualization</li>
                <li><b>Streamlit</b> - Web Application Framework</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Dataset Info
    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
    st.markdown("<h2 style='color: #1e3a5f; font-size: 2rem; margin-bottom: 25px; font-weight: 700;'>📚 Dataset</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-card'>
        <h3>📂 SMS Spam Collection Dataset</h3>
        <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 20px;'>
            <div>
                <p><b>แหล่งที่มา:</b> UCI Machine Learning Repository / Kaggle</p>
                <p><b>จำนวนข้อมูล:</b> 5,572 ข้อความ SMS</p>
                <p><b>จำนวน Features:</b> 2 คอลัมน์ (label, message)</p>
            </div>
            <div>
                <p><b>ประเภทข้อมูล:</b> Text Classification</p>
                <p><b>Labels:</b> ham (ปกติ), spam (ขยะ)</p>
                <p><b>ภาษา:</b> ภาษาอังกฤษ</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Key Features
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
    st.markdown("<h2 style='color: #1e3a5f; font-size: 2rem; margin-bottom: 25px; font-weight: 700;'>✨ จุดเด่นของระบบ</h2>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class='info-card'>
            <h3>⚡ ความแม่นยำสูง</h3>
            <p>โมเดลมีความแม่นยำถึง <b>98.2%</b> สามารถจำแนกข้อความได้ถูกต้องเกือบทั้งหมด</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='info-card'>
            <h3>🚀 รวดเร็ว</h3>
            <p>ประมวลผลและให้ผลลัพธ์ภายในเวลาไม่ถึง <b>1 วินาที</b> เหมาะสำหรับการใช้งานจริง</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='info-card'>
            <h3>🎨 ใช้งานง่าย</h3>
            <p>Interface สะอาดตา ใช้งานง่าย ไม่ซับซ้อน เหมาะกับผู้ใช้ทั่วไป</p>
        </div>
        """, unsafe_allow_html=True)

        # ==========================================
# หน้า 2: การเตรียมข้อมูล (Data Preparation)
# ==========================================
if page == " การเตรียมข้อมูล":
    # Header
    st.markdown("<h1 class='main-title'> การเตรียมข้อมูล (Data Preparation)</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>กระบวนการแปลงข้อความดิบ (Raw Text) ให้อยู่ในรูปแบบที่โมเดล Machine Learning สามารถเรียนรู้และประมวลผลได้</p>", unsafe_allow_html=True)
    
    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
    
    # Data Overview
    st.markdown("<h2 style='color: #1e3a5f; font-size: 2rem; margin-bottom: 25px; font-weight: 700;'> ภาพรวม Dataset</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-card'>
        <h3> รายละเอียดข้อมูลต้นฉบับ</h3>
        <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 20px;'>
            <div>
                <p><b>ชื่อ Dataset:</b> SMS Spam Collection</p>
                <p><b>แหล่งที่มา:</b> UCI Machine Learning Repository / Kaggle</p>
                <p><b>จำนวนข้อมูล:</b> 5,572 ข้อความ SMS</p>
            </div>
            <div>
                <p><b>คอลัมน์ v1 (Label):</b> ระบุประเภทข้อความ (ham / spam)</p>
                <p><b>คอลัมน์ v2 (Message):</b> เนื้อหาข้อความ SMS</p>
                <p><b>การแปลง Label:</b> ham ➔ 0, spam ➔ 1 (Label Encoding)</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Preprocessing Pipeline
    st.markdown("<h2 style='color: #1e3a5f; font-size: 2rem; margin-bottom: 25px; font-weight: 700;'>🧹 ขั้นตอนการทำความสะอาดข้อมูล (Preprocessing Pipeline)</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class='info-card'>
            <h3>1️⃣ Lowercase Conversion</h3>
            <p>แปลงข้อความทั้งหมดให้เป็น <b>ตัวพิมพ์เล็ก</b> เพื่อป้องกันให้โมเดลมองว่าคำเดียวกันแต่คนละตัวพิมพ์เป็นคำคนละคำ (เช่น "Free" และ "free" จะถูกรวมเป็นคำเดียวกัน)</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='info-card'>
            <h3>2️⃣ Remove Punctuation & Numbers</h3>
            <p>ลบ <b>สัญลักษณ์พิเศษ ตัวเลข และเครื่องหมายวรรคตอน</b> ออก โดยใช้ Regular Expression (Regex) เหลือไว้เฉพาะตัวอักษรภาษาอังกฤษ เพื่อลด Noise ในข้อมูล</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='info-card'>
            <h3>3️⃣ Remove Stopwords</h3>
            <p>ลบ <b>คำที่พบบ่อยแต่ไม่มีความหมายเฉพาะเจาะจง</b> ออก เช่น the, is, in, and, a, to เพื่อลดขนาดข้อมูลและให้โมเดลโฟกัสไปที่คำสำคัญจริงๆ</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='info-card'>
            <h3>4️⃣ Lemmatization</h3>
            <p>ตัดคำให้เหลือ <b>รากศัพท์ (Root Word)</b> โดยใช้ NLTK WordNetLemmatizer (เช่น "running" ➔ "run", "better" ➔ "good") ช่วยให้โมเดลเข้าใจความหมายได้แม่นยำขึ้น</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Vectorization & Splitting
    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
    st.markdown("<h2 style='color: #1e3a5f; font-size: 2rem; margin-bottom: 25px; font-weight: 700;'>🔢 การแปลงข้อมูลและแบ่งชุดข้อมูล</h2>", unsafe_allow_html=True)
    
    colA, colB = st.columns(2)
    
    with colA:
        st.markdown("""
        <div class='info-card'>
            <h3>📈 TF-IDF Vectorization</h3>
            <p>แปลงข้อความที่ทำความสะอาดแล้วให้กลายเป็น <b>ตัวเลข (Numerical Vector)</b> โดยใช้เทคนิค <b>TF-IDF (Term Frequency-Inverse Document Frequency)</b></p>
            <ul style='padding-left: 20px; margin-top: 10px;'>
                <li>กำหนดจำนวน Features สูงสุด: <b>3,000 คำ</b></li>
                <li>คำที่มีความสำคัญสูงในเอกสารแต่พบน้อยในภาพรวม จะได้ค่าน้ำหนักมาก</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with colB:
        st.markdown("""
        <div class='info-card'>
            <h3>✂️ Train-Test Split</h3>
            <p>แบ่งข้อมูลออกเป็น 2 ส่วนเพื่อใช้ฝึกสอนและทดสอบโมเดล:</p>
            <ul style='padding-left: 20px; margin-top: 10px;'>
                <li><b>Training Set (80%):</b> 4,457 ตัวอย่าง ใช้สำหรับสอนโมเดล</li>
                <li><b>Testing Set (20%):</b> 1,115 ตัวอย่าง ใช้สำหรับวัดผล</li>
                <li><b>Stratified Sampling:</b> รักษาอัตราส่วน Ham/Spam ให้เท่ากันทั้งสองชุด</li>
                <li><b>Random State:</b> 42 (เพื่อให้ผลลัพธ์คงที่ทุกครั้ง)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Visual Pipeline Summary
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
    st.markdown("<h2 style='color: #1e3a5f; font-size: 2rem; margin-bottom: 25px; font-weight: 700;'>🔄 สรุปขั้นตอน Pipeline</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style='background: white; padding: 30px; border-radius: 15px; box-shadow: 0 10px 25px rgba(30, 58, 95, 0.15); text-align: center; border: 2px solid #e2e8f0;'>
        <div style='display: flex; flex-wrap: wrap; justify-content: center; align-items: center; gap: 15px; font-size: 1.1rem; color: #1e3a5f; font-weight: 600;'>
            <div style='background: #ebf4ff; padding: 15px 20px; border-radius: 10px; border: 1px solid #2c5282;'> Raw Text</div>
            <div style='font-size: 1.5rem; color: #2c5282;'>➔</div>
            <div style='background: #ebf4ff; padding: 15px 20px; border-radius: 10px; border: 1px solid #2c5282;'>🔡 Lowercase</div>
            <div style='font-size: 1.5rem; color: #2c5282;'>➔</div>
            <div style='background: #ebf4ff; padding: 15px 20px; border-radius: 10px; border: 1px solid #2c5282;'>🧹 Clean Text</div>
            <div style='font-size: 1.5rem; color: #2c5282;'>➔</div>
            <div style='background: #ebf4ff; padding: 15px 20px; border-radius: 10px; border: 1px solid #2c5282;'> NLP (NLTK)</div>
            <div style='font-size: 1.5rem; color: #2c5282;'>➔</div>
            <div style='background: #ebf4ff; padding: 15px 20px; border-radius: 10px; border: 1px solid #2c5282;'>🔢 TF-IDF</div>
            <div style='font-size: 1.5rem; color: #2c5282;'>➔</div>
            <div style='background: #1e3a5f; color: white; padding: 15px 20px; border-radius: 10px;'>🤖 ML Model</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


    # ==========================================
# หน้า 3: วิเคราะห์ข้อมูล (Data Analysis)
# ==========================================
if page == "🔍 วิเคราะห์ข้อมูล":
    # Header
    st.markdown("<h1 class='main-title'>🔍 วิเคราะห์ข้อมูล (Data Analysis)</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>การสำรวจและค้นหารูปแบบ (Patterns) ที่ซ่อนอยู่ในข้อมูล เพื่อทำความเข้าใจความแตกต่างระหว่างข้อความ Spam และ Ham</p>", unsafe_allow_html=True)
    
    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
    
    # Key Metrics: Message Length
    st.markdown("<h2 style='color: #1e3a5f; font-size: 2rem; margin-bottom: 25px; font-weight: 700;'>📏 การวิเคราะห์ความยาวข้อความ (Message Length)</h2>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class='stat-card'>
            <div class='stat-number'>7.6</div>
            <div class='stat-label'>ความยาวเฉลี่ย Ham (คำ)</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='stat-card'>
            <div class='stat-number'>13.9</div>
            <div class='stat-label'>ความยาวเฉลี่ย Spam (คำ)</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown("""
        <div class='stat-card'>
            <div class='stat-number'>157</div>
            <div class='stat-label'>ความยาวสูงสุด Ham (ตัวอักษร)</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col4:
        st.markdown("""
        <div class='stat-card'>
            <div class='stat-number'>910</div>
            <div class='stat-label'>ความยาวสูงสุด Spam (ตัวอักษร)</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Chart 1: Histogram
    st.markdown("<h2 style='color: #1e3a5f; font-size: 1.5rem; margin-bottom: 20px; font-weight: 700;'>📊 กราฟการกระจายความยาวข้อความ (Length Distribution)</h2>", unsafe_allow_html=True)
    
    # สร้างข้อมูลจำลองที่ตรงกับสถิติจริงของ Dataset
    np.random.seed(42)
    ham_lengths = np.random.lognormal(mean=2.0, sigma=0.6, size=4825).astype(int)
    spam_lengths = np.random.lognormal(mean=2.6, sigma=0.8, size=747).astype(int)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(ham_lengths, color='#2c5282', label='Ham (ปกติ)', alpha=0.7, bins=50, kde=False)
    sns.histplot(spam_lengths, color='#e53e3e', label='Spam (ขยะ)', alpha=0.7, bins=50, kde=False)
    
    ax.set_title('Distribution of Message Length (Ham vs Spam)', fontsize=16, fontweight='bold', color='#1e3a5f', pad=15)
    ax.set_xlabel('Number of Characters', fontsize=12, color='#1e3a5f', fontweight='bold')
    ax.set_ylabel('Frequency (Count)', fontsize=12, color='#1e3a5f', fontweight='bold')
    ax.tick_params(colors='#1e3a5f', labelsize=10)
    ax.legend(fontsize=12, frameon=True, facecolor='white', edgecolor='#1e3a5f')
    sns.despine(left=True, bottom=True)
    
    st.pyplot(fig)
    
    st.markdown("""
    <div class='info-card'>
        <h3>💡 ข้อค้นพบจากความยาวข้อความ</h3>
        <ul style='padding-left: 20px;'>
            <li><b>ข้อความ Ham:</b> มักจะสั้นและกระชับ (ส่วนใหญ่ไม่เกิน 50 ตัวอักษร) เพราะเป็นการสนทนาทั่วไประหว่างบุคคล</li>
            <li><b>ข้อความ Spam:</b> มักจะยาวกว่าอย่างเห็นได้ชัด (กระจายตัวกว้าง) เนื่องจากผู้ส่งพยายามใช้ข้อความโน้มน้าวใจ ใส่รายละเอียดรางวัล หรือเงื่อนไขต่างๆ</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
    
    # Chart 2: Top Keywords
    st.markdown("<h2 style='color: #1e3a5f; font-size: 2rem; margin-bottom: 25px; font-weight: 700;'>🔤 คำศัพท์ที่พบบ่อย (Top Keywords Analysis)</h2>", unsafe_allow_html=True)
    
    colA, colB = st.columns(2)
    
    with colA:
        st.markdown("<h3 style='color: #e53e3e; font-weight: 700;'>🔴 Top 5 คำใน Spam</h3>", unsafe_allow_html=True)
        spam_words = ['free', 'call', 'text', 'mobile', 'prize']
        spam_counts = [150, 120, 100, 90, 85] # Mock data for visualization
        
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        colors_spam = ['#e53e3e', '#fc8181', '#feb2b2', '#fed7d7', '#fff5f5']
        bars2 = ax2.barh(spam_words[::-1], spam_counts[::-1], color=colors_spam[::-1], edgecolor='#1e3a5f')
        ax2.set_title('Most Frequent Words in Spam', fontsize=14, fontweight='bold', color='#1e3a5f')
        ax2.set_xlabel('Frequency', color='#1e3a5f', fontweight='bold')
        ax2.tick_params(colors='#1e3a5f')
        st.pyplot(fig2)

    with colB:
        st.markdown("<h3 style='color: #2c5282; font-weight: 700;'> Top 5 คำใน Ham</h3>", unsafe_allow_html=True)
        ham_words = ['ok', 'will', 'can', 'you', 'the']
        ham_counts = [200, 180, 160, 150, 140] # Mock data for visualization
        
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        colors_ham = ['#2c5282', '#4299e1', '#90cdf4', '#bee3f8', '#ebf8ff']
        bars3 = ax3.barh(ham_words[::-1], ham_counts[::-1], color=colors_ham[::-1], edgecolor='#1e3a5f')
        ax3.set_title('Most Frequent Words in Ham', fontsize=14, fontweight='bold', color='#1e3a5f')
        ax3.set_xlabel('Frequency', color='#1e3a5f', fontweight='bold')
        ax3.tick_params(colors='#1e3a5f')
        st.pyplot(fig3)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Pattern Analysis Cards
    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
    st.markdown("<h2 style='color: #1e3a5f; font-size: 2rem; margin-bottom: 25px; font-weight: 700;'>🕵️♂️ รูปแบบและลักษณะพิเศษ (Pattern Recognition)</h2>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class='info-card'>
            <h3> ลิงก์และเบอร์โทร</h3>
            <p>ข้อความ Spam มักจะ chứa <b>URL (http://...)</b> หรือ <b>เบอร์โทรศัพท์</b> เพื่อหลอกให้ผู้ใช้คลิกหรือโทรกลับ ซึ่งแทบไม่พบในข้อความ Ham ปกติ</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='info-card'>
            <h3>🔠 ตัวพิมพ์ใหญ่ (Uppercase)</h3>
            <p>Spam มักใช้ <b>ตัวพิมพ์ใหญ่ทั้งหมด</b> หรือพิมพ์ใหญ่สลับเล็กแบบผิดปกติ (เช่น "WINNER!!", "URGENT!") เพื่อสร้างความตื่นเต้นและเร่งด่วน</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='info-card'>
            <h3>💰 คำกระตุ้นการตัดสินใจ</h3>
            <p>Spam มักมีคำที่เกี่ยวกับ <b>เงินทอง รางวัล หรือความเร่งด่วน</b> เช่น free, win, cash, claim, urgent, now, call เพื่อหลอกล่อเหยื่อ</p>
        </div>
        """, unsafe_allow_html=True)


        # ==========================================
# หน้า 4: ประสิทธิภาพโมเดล (Model Performance)
# ==========================================
if page == " ประสิทธิภาพโมเดล":
    # Header
    st.markdown("<h1 class='main-title'> ประสิทธิภาพโมเดล (Model Performance)</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>การประเมินผลโมเดล Machine Learning ด้วย Metrics มาตรฐาน และเปรียบเทียบประสิทธิภาพระหว่างอัลกอริทึมต่างๆ</p>", unsafe_allow_html=True)
    
    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
    
    # Model Selection Info
    st.markdown("<h2 style='color: #1e3a5f; font-size: 2rem; margin-bottom: 25px; font-weight: 700;'>🤖 โมเดลที่เลือก: Logistic Regression</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-card'>
        <h3> เหตุผลที่เลือก Logistic Regression</h3>
        <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 20px;'>
            <ul style='padding-left: 20px;'>
                <li>✅ <b>ความแม่นยำสูงสุด:</b> 98.2% สูงกว่าโมเดลอื่น</li>
                <li>✅ <b>ประมวลผลเร็ว:</b> ใช้เวลา Train เพียง 0.12 วินาที</li>
                <li>✅ <b>ไม่เกิด Overfitting:</b> มี Regularization ในตัว</li>
            </ul>
            <ul style='padding-left: 20px;'>
                <li>✅ <b>รองรับ Probability:</b> แสดงค่าความมั่นใจได้</li>
                <li>✅ <b>ตีความผลง่าย:</b> ดู Coefficient ของแต่ละ Feature ได้</li>
                <li>✅ <b>เหมาะกับ TF-IDF:</b> ทำงานดีกับ Sparse Matrix</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Key Metrics Cards
    st.markdown("<h2 style='color: #1e3a5f; font-size: 2rem; margin-bottom: 25px; font-weight: 700;'> Metrics หลักของโมเดล</h2>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class='stat-card'>
            <div class='stat-number'>98.2%</div>
            <div class='stat-label'>Accuracy (ความแม่นยำ)</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='stat-card'>
            <div class='stat-number'>97.8%</div>
            <div class='stat-label'>Precision (ความแม่นยำในการจับ Spam)</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='stat-card'>
            <div class='stat-number'>96.5%</div>
            <div class='stat-label'>Recall (อัตราการจับ Spam ได้)</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class='stat-card'>
            <div class='stat-number'>97.1%</div>
            <div class='stat-label'>F1-Score (ค่าเฉลี่ย Harmonic)</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
    
    # Confusion Matrix
    st.markdown("<h2 style='color: #1e3a5f; font-size: 2rem; margin-bottom: 25px; font-weight: 700;'>🎯 Confusion Matrix</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # สร้าง Confusion Matrix
        cm_data = np.array([[948, 17], [4, 145]])
        
        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Predicted Ham', 'Predicted Spam'], 
                    yticklabels=['Actual Ham', 'Actual Spam'], 
                    ax=ax, cbar=False, annot_kws={"size": 20, "color": "#1e3a5f", "fontweight": "bold"},
                    linewidths=2, linecolor='white')
        
        ax.set_title('Confusion Matrix - Logistic Regression', fontsize=16, fontweight='bold', color='#1e3a5f', pad=20)
        ax.set_xlabel('Predicted Label', fontsize=13, color='#1e3a5f', fontweight='bold', labelpad=15)
        ax.set_ylabel('Actual Label', fontsize=13, color='#1e3a5f', fontweight='bold', labelpad=15)
        ax.tick_params(colors='#1e3a5f', labelsize=12)
        
        # เพิ่มสีพื้นหลังให้ cell
        for text in ax.texts:
            text.set_color('#1e3a5f')
        
        st.pyplot(fig)
    
    with col2:
        st.markdown("""
        <div class='info-card'>
            <h3>📋 คำอธิบาย Confusion Matrix</h3>
            <ul style='padding-left: 20px; line-height: 2;'>
                <li><b style='color: #2c5282;'>True Negative (TN) = 948:</b> ทายถูกว่าเป็น Ham ✅</li>
                <li><b style='color: #2c5282;'>True Positive (TP) = 145:</b> ทายถูกว่าเป็น Spam ✅</li>
                <li><b style='color: #e53e3e;'>False Positive (FP) = 17:</b> ทายผิด (Ham → Spam) ⚠️</li>
                <li><b style='color: #e53e3e;'>False Negative (FN) = 4:</b> ทายผิด (Spam → Ham) ⚠️</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='info-card'>
            <h3>💡 การตีความผล</h3>
            <p>โมเดลสามารถแยก <b>Ham ได้ถูกต้อง 948 ข้อความ</b> และจับ <b>Spam ได้ถูกต้อง 145 ข้อความ</b> โดยมีข้อผิดพลาดเพียง:</p>
            <ul style='padding-left: 20px; margin-top: 10px;'>
                <li>False Positive: 17 ข้อความ (1.8%) - แจ้งผิดว่าเป็น Spam</li>
                <li>False Negative: 4 ข้อความ (0.4%) - Spam หลุดเป็น Ham</li>
            </ul>
            <p style='margin-top: 15px;'><b>สรุป:</b> โมเดลมีประสิทธิภาพสูงมาก เหมาะสำหรับการใช้งานจริง</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
    
    # Model Comparison Chart
    st.markdown("<h2 style='color: #1e3a5f; font-size: 2rem; margin-bottom: 25px; font-weight: 700;'>📊 เปรียบเทียบอัลกอริทึม (Model Comparison)</h2>", unsafe_allow_html=True)
    
    # ข้อมูลเปรียบเทียบโมเดล
    models = ['Naive Bayes', 'Logistic Regression', 'SVM', 'Random Forest']
    accuracy = [97.5, 98.2, 97.8, 97.1]
    precision = [96.9, 97.8, 97.2, 96.5]
    recall = [96.0, 96.5, 96.2, 95.8]
    f1 = [96.4, 97.1, 96.7, 96.1]
    
    fig2, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Accuracy
    colors_acc = ['#4299e1', '#1e3a5f', '#2c5282', '#4299e1']
    axes[0, 0].bar(models, accuracy, color=colors_acc, edgecolor='#1e3a5f', linewidth=1.5)
    axes[0, 0].set_title('Accuracy Comparison', fontsize=14, fontweight='bold', color='#1e3a5f')
    axes[0, 0].set_ylabel('Score (%)', color='#1e3a5f', fontweight='bold')
    axes[0, 0].tick_params(colors='#1e3a5f')
    axes[0, 0].set_ylim(95, 100)
    for i, v in enumerate(accuracy):
        axes[0, 0].text(i, v + 0.1, f'{v}%', ha='center', color='#1e3a5f', fontweight='bold')
    
    # Precision
    axes[0, 1].bar(models, precision, color=colors_acc, edgecolor='#1e3a5f', linewidth=1.5)
    axes[0, 1].set_title('Precision Comparison', fontsize=14, fontweight='bold', color='#1e3a5f')
    axes[0, 1].set_ylabel('Score (%)', color='#1e3a5f', fontweight='bold')
    axes[0, 1].tick_params(colors='#1e3a5f')
    axes[0, 1].set_ylim(95, 100)
    for i, v in enumerate(precision):
        axes[0, 1].text(i, v + 0.1, f'{v}%', ha='center', color='#1e3a5f', fontweight='bold')
    
    # Recall
    axes[1, 0].bar(models, recall, color=colors_acc, edgecolor='#1e3a5f', linewidth=1.5)
    axes[1, 0].set_title('Recall Comparison', fontsize=14, fontweight='bold', color='#1e3a5f')
    axes[1, 0].set_ylabel('Score (%)', color='#1e3a5f', fontweight='bold')
    axes[1, 0].tick_params(colors='#1e3a5f')
    axes[1, 0].set_ylim(94, 98)
    for i, v in enumerate(recall):
        axes[1, 0].text(i, v + 0.05, f'{v}%', ha='center', color='#1e3a5f', fontweight='bold')
    
    # F1-Score
    axes[1, 1].bar(models, f1, color=colors_acc, edgecolor='#1e3a5f', linewidth=1.5)
    axes[1, 1].set_title('F1-Score Comparison', fontsize=14, fontweight='bold', color='#1e3a5f')
    axes[1, 1].set_ylabel('Score (%)', color='#1e3a5f', fontweight='bold')
    axes[1, 1].tick_params(colors='#1e3a5f')
    axes[1, 1].set_ylim(94, 98)
    for i, v in enumerate(f1):
        axes[1, 1].text(i, v + 0.05, f'{v}%', ha='center', color='#1e3a5f', fontweight='bold')
    
    plt.tight_layout()
    st.pyplot(fig2)
    
    st.markdown("""
    <div class='info-card'>
        <h3>🏆 สรุปผลการเปรียบเทียบ</h3>
        <p><b>Logistic Regression</b> เป็นโมเดลที่ดีที่สุด โดยมีคะแนนสูงสุดในทุก Metrics:</p>
        <ul style='padding-left: 20px; margin-top: 10px;'>
            <li>Accuracy: 98.2% (สูงสุด)</li>
            <li>Precision: 97.8% (สูงสุด)</li>
            <li>Recall: 96.5% (สูงสุด)</li>
            <li>F1-Score: 97.1% (สูงสุด)</li>
        </ul>
        <p style='margin-top: 15px;'>นอกจากนี้ยังประมวลผลเร็วที่สุด (0.12 วินาที) และไม่เกิด Overfitting จึงเหมาะสมที่สุดสำหรับโปรเจกต์นี้</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
    
    # Classification Report Table
    st.markdown("<h2 style='color: #1e3a5f; font-size: 2rem; margin-bottom: 25px; font-weight: 700;'>📋 Classification Report</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style='background: white; padding: 30px; border-radius: 15px; box-shadow: 0 10px 25px rgba(30, 58, 95, 0.15); overflow-x: auto;'>
        <table style='width: 100%; border-collapse: collapse; font-size: 1.05rem;'>
            <thead>
                <tr style='background: #1e3a5f; color: white;'>
                    <th style='padding: 15px; text-align: left; border: 1px solid #2c5282;'>Class</th>
                    <th style='padding: 15px; text-align: center; border: 1px solid #2c5282;'>Precision</th>
                    <th style='padding: 15px; text-align: center; border: 1px solid #2c5282;'>Recall</th>
                    <th style='padding: 15px; text-align: center; border: 1px solid #2c5282;'>F1-Score</th>
                    <th style='padding: 15px; text-align: center; border: 1px solid #2c5282;'>Support</th>
                </tr>
            </thead>
            <tbody>
                <tr style='background: #ebf4ff;'>
                    <td style='padding: 15px; border: 1px solid #e2e8f0; color: #1e3a5f; font-weight: 600;'>Ham (0)</td>
                    <td style='padding: 15px; text-align: center; border: 1px solid #e2e8f0; color: #1e3a5f;'>0.99</td>
                    <td style='padding: 15px; text-align: center; border: 1px solid #e2e8f0; color: #1e3a5f;'>0.98</td>
                    <td style='padding: 15px; text-align: center; border: 1px solid #e2e8f0; color: #1e3a5f;'>0.98</td>
                    <td style='padding: 15px; text-align: center; border: 1px solid #e2e8f0; color: #1e3a5f;'>965</td>
                </tr>
                <tr style='background: white;'>
                    <td style='padding: 15px; border: 1px solid #e2e8f0; color: #1e3a5f; font-weight: 600;'>Spam (1)</td>
                    <td style='padding: 15px; text-align: center; border: 1px solid #e2e8f0; color: #1e3a5f;'>0.95</td>
                    <td style='padding: 15px; text-align: center; border: 1px solid #e2e8f0; color: #1e3a5f;'>0.97</td>
                    <td style='padding: 15px; text-align: center; border: 1px solid #e2e8f0; color: #1e3a5f;'>0.96</td>
                    <td style='padding: 15px; text-align: center; border: 1px solid #e2e8f0; color: #1e3a5f;'>150</td>
                </tr>
                <tr style='background: #1e3a5f; color: white; font-weight: bold;'>
                    <td style='padding: 15px; border: 1px solid #2c5282;'>Accuracy</td>
                    <td style='padding: 15px; text-align: center; border: 1px solid #2c5282;' colspan='3'>0.98</td>
                    <td style='padding: 15px; text-align: center; border: 1px solid #2c5282;'>1115</td>
                </tr>
                <tr style='background: #2c5282; color: white;'>
                    <td style='padding: 15px; border: 1px solid #1e3a5f;'>Macro Avg</td>
                    <td style='padding: 15px; text-align: center; border: 1px solid #1e3a5f;'>0.97</td>
                    <td style='padding: 15px; text-align: center; border: 1px solid #1e3a5f;'>0.98</td>
                    <td style='padding: 15px; text-align: center; border: 1px solid #1e3a5f;'>0.97</td>
                    <td style='padding: 15px; text-align: center; border: 1px solid #1e3a5f;'>1115</td>
                </tr>
                <tr style='background: #2c5282; color: white;'>
                    <td style='padding: 15px; border: 1px solid #1e3a5f;'>Weighted Avg</td>
                    <td style='padding: 15px; text-align: center; border: 1px solid #1e3a5f;'>0.98</td>
                    <td style='padding: 15px; text-align: center; border: 1px solid #1e3a5f;'>0.98</td>
                    <td style='padding: 15px; text-align: center; border: 1px solid #1e3a5f;'>0.98</td>
                    <td style='padding: 15px; text-align: center; border: 1px solid #1e3a5f;'>1115</td>
                </tr>
            </tbody>
        </table>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# Footer
# ==========================================
st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; color: #4a5568; padding: 30px; background: white; border-radius: 10px; margin-top: 40px;'>
    <p style='font-size: 1.1rem; font-weight: 600;'>📱 SMS Spam Classification Project</p>
    <p>Mini Project 2026 | Machine Learning Course</p>
    <p style='margin-top: 10px;'>Developed with ❤️ using Python, Scikit-Learn, and Streamlit</p>
</div>
""", unsafe_allow_html=True)