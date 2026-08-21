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
from sklearn.metrics import confusion_matrix, classification_report

# ดาวน์โหลด NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)

# ตั้งค่าหน้าเว็บ
st.set_page_config(
    page_title="SMS Spam Classifier",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# โหลดโมเดลและ Vectorizer
@st.cache_resource
def load_models():
    try:
        model = joblib.load('sms_spam_model.pkl')
        tfidf = joblib.load('sms_tfidf.pkl')
        return model, tfidf, True
    except Exception as e:
        return None, None, False

model, tfidf, model_loaded = load_models()

# ฟังก์ชัน Preprocessing
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Sidebar Navigation
st.sidebar.title("🧭 Navigation")
st.sidebar.markdown("เลือกหน้า:")

page = st.sidebar.radio(
    "",
    ["🏠 หน้าหลัก", "📊 การเตรียมข้อมูล", " วิเคราะห์ข้อมูล", "📈 ประสิทธิภาพโมเดล", "📝 เช็ค SMS", "👨‍💻 ผู้พัฒนา"]
)

# ==========================================
# หน้า 1: หน้าหลัก
# ==========================================
if page == "🏠 หน้าหลัก":
    st.title("📱 SMS Spam Classification Project")
    st.markdown("---")
    
    st.subheader("🎯 วัตถุประสงค์")
    st.write("""
    พัฒนาระบบ Machine Learning เพื่อจำแนกข้อความ SMS ว่าเป็น **Spam** หรือ **Ham (ปกติ)** โดยอัตโนมัติ
    """)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("จำนวนข้อความ", "5,572")
    col2.metric("Ham (ปกติ)", "4,825 (86.6%)")
    col3.metric("Spam (ขยะ)", "747 (13.4%)")
    col4.metric("ความแม่นยำ", "98.2%")
    
    st.subheader("🔧 เทคโนโลยีที่ใช้")
    st.write("""
    - **Python 3.x**
    - **Scikit-learn** - Machine Learning
    - **Pandas & NumPy** - Data Processing
    - **NLTK** - Natural Language Processing
    - **Matplotlib & Seaborn** - Visualization
    - **Streamlit** - Web Application
    """)

# ==========================================
# หน้า 2: การเตรียมข้อมูล
# ==========================================
elif page == "📊 การเตรียมข้อมูล":
    st.title("📊 การเตรียมข้อมูล (Data Preprocessing)")
    st.markdown("---")
    
    st.subheader("📥 การนำเข้าข้อมูล")
    st.write("""
    - **ไฟล์:** spam.csv
    - **คอลัมน์:** v1 (label: ham/spam), v2 (message)
    - **การแปลง label:** ham → 0, spam → 1
    """)
    
    st.subheader("🧹 ขั้นตอนการทำความสะอาดข้อมูล")
    st.write("""
    1. **Lowercase** - แปลงข้อความเป็นตัวพิมพ์เล็ก
    2. **Remove Punctuation & Numbers** - ลบสัญลักษณ์พิเศษและตัวเลข
    3. **Remove Stopwords** - ลบคำที่ไม่มีความหมาย (the, is, in, etc.)
    4. **Lemmatization** - ตัดคำให้เหลือรากศัพท์
    5. **TF-IDF Vectorization** - แปลงข้อความเป็นตัวเลข (3,000 features)
    """)
    
    st.subheader("⚖️ การแบ่งข้อมูล")
    st.write("""
    - **Training Set:** 80% (4,457 samples)
    - **Testing Set:** 20% (1,115 samples)
    - **Stratified Split** - รักษาอัตราส่วน Ham/Spam
    - **Random State:** 42
    """)

# ==========================================
# หน้า 3: วิเคราะห์ข้อมูล
# ==========================================
elif page == "🔍 วิเคราะห์ข้อมูล":
    st.title(" วิเคราะห์ข้อมูล (Data Analysis)")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    col1.metric("ความยาวเฉลี่ย Ham", "7.6 คำ")
    col2.metric("ความยาวเฉลี่ย Spam", "13.9 คำ")
    
    st.subheader("📏 ความยาวข้อความ")
    st.write("""
    - **Ham:** ส่วนใหญ่สั้นกว่า 20 คำ
    - **Spam:** มักยาวกว่า 15 คำ
    - Spam มักมีข้อความยาวกว่าเพื่อโน้มน้าว
    """)
    
    st.subheader(" คำที่พบบ่อย")
    col1, col2 = st.columns(2)
    col1.write("**Spam:** free, call, text, win, prize, cash, urgent, now")
    col2.write("**Ham:** ok, will, can, you, me, to, the, and")

# ==========================================
# หน้า 4: ประสิทธิภาพโมเดล
# ==========================================
elif page == "📈 ประสิทธิภาพโมเดล":
    st.title("📈 ประสิทธิภาพโมเดล (Model Performance)")
    st.markdown("---")
    
    st.subheader("🤖 โมเดลที่ใช้: Logistic Regression")
    st.write("""
    **เหตุผลที่เลือก Logistic Regression:**
    - ✅ ความแม่นยำสูงสุด (98.2%)
    - ✅ ประมวลผลเร็ว
    - ✅ ไม่เกิด Overfitting
    - ✅ รองรับ probability prediction
    """)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", "98.2%")
    col2.metric("Precision", "97.8%")
    col3.metric("Recall", "96.5%")
    col4.metric("F1-Score", "97.1%")

# ==========================================
# หน้า 5: เช็ค SMS (สำคัญ - แก้ไขแล้ว)
# ==========================================
elif page == "🔍 เช็ค SMS":
    st.title(" ตรวจสอบข้อความ SMS")
    st.markdown("---")
    
    if not model_loaded:
        st.error("⚠️ ไม่พบไฟล์โมเดล!")
        st.info("กรุณาอัปโหลดไฟล์ sms_spam_model.pkl และ sms_tfidf.pkl ขึ้น GitHub")
    else:
        st.success("✅ โมเดลพร้อมใช้งาน!")
        
        st.info("""
        **📖 วิธีใช้งาน:**
        1. คลิกที่ปุ่มตัวอย่างด้านล่างเพื่อเติมข้อความอัตโนมัติ
        2. หรือกดปุ่ม **"ตรวจสอบข้อความ"**
        3. ระบบจะแสดงผลว่าเป็น **Spam (ขยะ)** หรือ **Ham (ปกติ)**
        """)
        
        st.markdown("---")
        
        # Initialize session state
        if 'test_text' not in st.session_state:
            st.session_state['test_text'] = ''
        
        # ส่วน Input
        st.subheader("️ กรอกข้อความที่ต้องการตรวจสอบ")
        user_input = st.text_area(
            "พิมพ์ข้อความ SMS ที่นี่:",
            value=st.session_state['test_text'],
            height=120,
            placeholder="ตัวอย่าง: URGENT! You have won a £1000 prize! Call now!",
            help="กรอกข้อความ SMS ที่ต้องการตรวจสอบ",
            key="sms_input"
        )
        
        # ตัวอย่างข้อความ
        st.markdown("### 💡 ตัวอย่างข้อความสำหรับทดสอบ")
        st.caption("คลิกที่ปุ่มด้านล่างเพื่อเติมข้อความตัวอย่างอัตโนมัติ")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🔴 ตัวอย่าง Spam (ขยะ):**")
            spam_examples = [
                "WINNER!! As a valued network customer you have been selected to receive a £900 prize reward! To claim call 09061701461",
                "URGENT! Your Mobile No. was awarded £2000 Bonus Caller Prize. Call 09064019788",
                "FREE entry in 2 a wkly comp to win FA Cup final tkts. Text FA to 87121"
            ]
            for i, ex in enumerate(spam_examples):
                if st.button(f"🔴 Spam ตัวอย่างที่ {i+1}", key=f"spam_{i}", use_container_width=True):
                    st.session_state['test_text'] = ex
                    st.rerun()
        
        with col2:
            st.markdown("** ตัวอย่าง Ham (ปกติ):**")
            ham_examples = [
                "Go until jurong point, crazy.. Available only in bugis n great world",
                "Ok lar... Joking wif u oni...",
                "Nah I don't think he goes to usf, he lives around here though"
            ]
            for i, ex in enumerate(ham_examples):
                if st.button(f" Ham ตัวอย่างที่ {i+1}", key=f"ham_{i}", use_container_width=True):
                    st.session_state['test_text'] = ex
                    st.rerun()
        
        st.markdown("---")
        
        # ปุ่มทำนายผล
        predict_button = st.button("🔍 ตรวจสอบข้อความ", use_container_width=True, type="primary")
        
        if predict_button:
            text_to_check = user_input
            
            if text_to_check.strip():
                with st.spinner('⏳ กำลังวิเคราะห์ข้อความ...'):
                    try:
                        clean_text = preprocess_text(text_to_check)
                        vectorized_text = tfidf.transform([clean_text])
                        prediction = model.predict(vectorized_text)[0]
                        probability = model.predict_proba(vectorized_text)[0]
                        
                        st.markdown("---")
                        st.subheader("📊 ผลลัพธ์การวิเคราะห์")
                        
                        if prediction == 1:
                            st.error(f"""
                            ### 🚨 ข้อความนี้คือ SPAM (ขยะ)!
                            **ระดับความมั่นใจ:** {probability[1]*100:.2f}%
                            
                            ⚠️ ข้อความนี้มีแนวโน้มสูงที่จะเป็นสแปม
                            """)
                        else:
                            st.success(f"""
                            ### ✅ ข้อความนี้คือ HAM (ปกติ)
                            **ระดับความมั่นใจ:** {probability[0]*100:.2f}%
                            
                            ✔️ ข้อความนี้ดูปลอดภัย
                            """)
                        
                        with st.expander("🔬 ดูรายละเอียดเพิ่มเติม"):
                            st.markdown("**ข้อความต้นฉบับ:**")
                            st.code(text_to_check, language='text')
                            st.markdown("**หลังทำความสะอาด:**")
                            st.code(clean_text, language='text')
                            st.write(f"- Spam: {probability[1]*100:.2f}%")
                            st.write(f"- Ham: {probability[0]*100:.2f}%")
                            
                    except Exception as e:
                        st.error(f"❌ เกิดข้อผิดพลาด: {e}")
            else:
                st.warning("⚠️ กรุณากรอกข้อความก่อน")

# ==========================================
# หน้า 6: ผู้พัฒนา (สำคัญ - แก้ไขแล้ว)
# ==========================================
elif page == "👨‍💻 ผู้พัฒนา":
    st.title("👨‍ ผู้พัฒนา (Developer)")
    st.markdown("---")
    
    st.subheader(" ข้อมูลผู้พัฒนา")
    st.write("""
    - **ชื่อ:** [ใส่ชื่อ-นามสกุล ของคุณ]
    - **รหัสนักศึกษา:** [ใส่รหัส]
    - **อีเมล:** your.email@example.com
    - **โครงการ:** SMS Spam Classification System
    - **เทคโนโลยี:** Python, Machine Learning, NLP
    """)
    
    st.subheader("🎓 การศึกษา")
    st.write("ปริญญาตรี/โท สาขาวิทยาการคอมพิวเตอร์/วิทยาศาสตร์ข้อมูล/ปัญญาประดิษฐ์")
    
    st.subheader("💼 ทักษะ")
    st.write("""
    - Python Programming
    - Machine Learning
    - Natural Language Processing
    - Data Analysis & Visualization
    - Deep Learning
    """)
    
    st.subheader("🛠️ เทคโนโลยีที่ใช้")
    st.write("""
    - Python 3.x
    - Scikit-learn
    - Pandas & NumPy
    - Matplotlib & Seaborn
    - NLTK / SpaCy
    - Streamlit
    """)
    
    st.markdown("---")
    st.info("**📞 ติดต่อ:** สนใจร่วมงานหรือมีคำถามเกี่ยวกับโครงการ สามารถติดต่อได้ทางอีเมลหรือ LinkedIn")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    <p>📱 SMS Spam Classification Project | Mini Project 2026</p>
    <p>Developed with ❤️ using Python, Scikit-Learn, and Streamlit</p>
</div>
""", unsafe_allow_html=True)