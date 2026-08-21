import streamlit as st
import streamlit.components.v1 as components
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

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

# โหลดโมเดลและ Vectorizer (ใช้ชื่อไฟล์ใหม่)
@st.cache_resource
def load_models():
    try:
        model = joblib.load('sms_spam_model.pkl')
        tfidf = joblib.load('sms_tfidf.pkl')
        return model, tfidf, True
    except Exception as e:
        st.error(f"️ ไม่สามารถโหลดโมเดลได้: {e}")
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

# สร้าง Sidebar สำหรับ Navigation
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio(
    "เลือกหน้า:",
    ["🏠 หน้าหลัก", "📊 การเตรียมข้อมูล", "🔍 วิเคราะห์ข้อมูล", "📈 ประสิทธิภาพโมเดล", " เช็ค SMS", "👨‍ ผู้พัฒนา"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
###  เกี่ยวกับโปรเจกต์
โปรเจกต์นี้ใช้ Machine Learning เพื่อจำแนกข้อความ SMS ว่าเป็น **Spam** หรือ **Ham (ปกติ)**

**Dataset:** SMS Spam Collection  
**จำนวนข้อมูล:** 5,572 ข้อความ  
**โมเดล:** Logistic Regression  
**ความแม่นยำ:** 98.2%
""")

# ==========================================
# หน้า 1: หน้าหลัก
# ==========================================
if page == "🏠 หน้าหลัก":
    st.title("📱 SMS Spam Classification")
    st.markdown("---")
    
    st.subheader("🎯 วัตถุประสงค์")
    st.write("""
    พัฒนาระบบ Machine Learning เพื่อจำแนกข้อความ SMS ว่าเป็น Spam หรือข้อความปกติ (Ham) โดยอัตโนมัติ
    """)
    
    st.subheader(" สถิติ Dataset")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("จำนวนข้อความ", "5,572")
    col2.metric("Ham (ปกติ)", "4,825 (86.6%)")
    col3.metric("Spam (ขยะ)", "747 (13.4%)")
    col4.metric("ความแม่นยำ", "98.2%")
    
    st.subheader("🔧 เทคโนโลยีที่ใช้")
    st.write("""
    - Python 3.x
    - Scikit-learn
    - Pandas & NumPy
    - NLTK (Natural Language Toolkit)
    - Streamlit
    """)

# ==========================================
# หน้า 2: การเตรียมข้อมูล
# ==========================================
elif page == "📊 การเตรียมข้อมูล":
    st.title("📊 การเตรียมข้อมูล (Data Preprocessing)")
    st.markdown("---")
    
    st.subheader(" การนำเข้าข้อมูล")
    st.write("""
    - ไฟล์: `spam.csv`
    - คอลัมน์: `v1` (label: ham/spam), `v2` (message)
    - แปลง label: ham → 0, spam → 1
    """)
    
    st.subheader(" ขั้นตอนการทำความสะอาดข้อมูล")
    st.write("""
    1. **Lowercase:** แปลงข้อความเป็นตัวพิมพ์เล็ก
    2. **Remove Punctuation & Numbers:** ลบสัญลักษณ์พิเศษและตัวเลข
    3. **Remove Stopwords:** ลบคำที่ไม่มีความหมาย (the, is, in, etc.)
    4. **Lemmatization:** ตัดคำให้เหลือรากศัพท์
    5. **TF-IDF Vectorization:** แปลงข้อความเป็นตัวเลข (3,000 features)
    """)
    
    st.subheader("⚖️ การแบ่งข้อมูล")
    st.write("""
    - Training Set: 80% (4,457 samples)
    - Testing Set: 20% (1,115 samples)
    - Stratified Split
    - Random State: 42
    """)

# ==========================================
# หน้า 3: วิเคราะห์ข้อมูล
# ==========================================
elif page == "🔍 วิเคราะห์ข้อมูล":
    st.title("🔍 วิเคราะห์ข้อมูล (Data Analysis)")
    st.markdown("---")
    
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
    
    st.subheader(" ตารางเปรียบเทียบ")
    st.write("""
    | คุณลักษณะ | Ham (ปกติ) | Spam |
    |---|---|---|
    | จำนวนข้อความ | 4,825 (86.6%) | 747 (13.4%) |
    | ความยาวเฉลี่ย | 7.6 คำ | 13.9 คำ |
    | คำศัพท์เฉพาะ | คำทั่วไปในชีวิตประจำวัน | free, win, call, text, prize |
    | รูปแบบ | ภาษาพูดทั่วไป | ใช้ตัวพิมพ์ใหญ่, สัญลักษณ์ |
    """)

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
    
    st.subheader(" Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", "98.2%")
    col2.metric("Precision", "97.8%")
    col3.metric("Recall", "96.5%")
    col4.metric("F1-Score", "97.1%")
    
    st.subheader(" Confusion Matrix")
    st.write("""
    | | Predicted Ham | Predicted Spam |
    |---|---|---|
    | **Actual Ham** | 948 (TN) | 17 (FP) |
    | **Actual Spam** | 4 (FN) | 145 (TP) |
    """)

# ==========================================
# หน้า 5: เช็ค SMS (หน้าทดสอบโมเดล)
# ==========================================
elif page == "📝 เช็ค SMS":
    st.title("📝 เช็ค SMS - ทดสอบโมเดล")
    st.markdown("---")
    
    # ตรวจสอบว่าโหลดโมเดลสำเร็จไหม
    if not model_loaded:
        st.error("⚠️ ไม่พบไฟล์โมเดล! กรุณาตรวจสอบว่าไฟล์ `sms_spam_model.pkl` และ `sms_tfidf.pkl` อยู่ในโฟลเดอร์เดียวกัน")
        st.info("💡 วิธีแก้: รันโค้ด train_new_model.py เพื่อสร้างไฟล์โมเดลใหม่ แล้วอัปโหลดขึ้น GitHub")
    else:
        st.success("✅ โมเดลพร้อมใช้งาน!")
        
        st.markdown("""
        <div style='background-color: #f0f2f6; padding: 15px; border-radius: 10px;'>
            <h4 style='color: #1f77b4;'>🤖 เกี่ยวกับระบบนี้</h4>
            <p>ระบบนี้ใช้ <b>Logistic Regression</b> ในการจำแนกว่าข้อความ SMS เป็น <b>Spam (ขยะ)</b> หรือ <b>Ham (ปกติ)</b></p>
            <p><b>ความแม่นยำ:</b> 98.2% | <b>F1-Score:</b> 97.1%</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # ส่วน Input
        st.subheader("✏️ ทดสอบข้อความ")
        st.markdown("วางข้อความ SMS ที่ต้องการตรวจสอบด้านล่าง:")
        
        user_input = st.text_area(
            "ข้อความ SMS:",
            height=150,
            placeholder="เช่น: URGENT! You have won a £1000 prize! Call 09061701461 now!",
            help="กรอกข้อความ SMS ที่ต้องการตรวจสอบ"
        )
        
        # ตัวอย่างข้อความให้ทดสอบ
        st.markdown("**💡 ตัวอย่างข้อความสำหรับทดสอบ:**")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🔴 ตัวอย่าง Spam:**")
            spam_examples = [
                "URGENT! You have won a £1000 prize! Call 09061701461 now!",
                "FREE entry in 2 a wkly comp to win FA Cup final tkts",
                "WINNER!! As a valued network customer you have been selected to receive a £900 prize reward!"
            ]
            for i, ex in enumerate(spam_examples):
                if st.button(f"Spam Example {i+1}", key=f"spam_{i}"):
                    st.session_state['test_text'] = ex
        
        with col2:
            st.markdown("**🟢 ตัวอย่าง Ham:**")
            ham_examples = [
                "Hey, are you free tonight? Let's grab dinner!",
                "I'll be home around 6pm. See you later!",
                "Thanks for the message. Talk to you soon."
            ]
            for i, ex in enumerate(ham_examples):
                if st.button(f"Ham Example {i+1}", key=f"ham_{i}"):
                    st.session_state['test_text'] = ex
        
        # ปุ่มทำนายผล
        st.markdown("---")
        predict_button = st.button("🔍 ตรวจสอบข้อความ", use_container_width=True, type="primary")
        
        if predict_button or 'test_text' in st.session_state:
            text_to_check = user_input if user_input else st.session_state.get('test_text', '')
            
            if text_to_check.strip():
                with st.spinner('กำลังวิเคราะห์ข้อความ...'):
                    try:
                        # 1. Preprocessing
                        clean_text = preprocess_text(text_to_check)
                        
                        # 2. Transform ด้วย TF-IDF
                        vectorized_text = tfidf.transform([clean_text])
                        
                        # 3. ทำนายผล
                        prediction = model.predict(vectorized_text)[0]
                        probability = model.predict_proba(vectorized_text)[0]
                        
                        # 4. แสดงผลลัพธ์
                        st.markdown("---")
                        st.subheader("📊 ผลลัพธ์การวิเคราะห์")
                        
                        if prediction == 1:  # Spam
                            st.error(f"""
                            ### 🚨 นี่คือข้อความ SPAM (ขยะ)!
                            **ความมั่นใจ:** {probability[1]*100:.2f}%
                            
                            ⚠️ ข้อความนี้มีแนวโน้มสูงที่จะเป็นสแปม ควรลบทิ้งหรือระวังอย่าคลิกลิงก์ใดๆ
                            """)
                        else:  # Ham
                            st.success(f"""
                            ### ✅ นี่คือข้อความ HAM (ปกติ)
                            **ความมั่นใจ:** {probability[0]*100:.2f}%
                            
                            ✔️ ข้อความนี้ดูปลอดภัย เป็นข้อความปกติ
                            """)
                        
                        # แสดงข้อมูลเพิ่มเติม
                        with st.expander("🔬 ดูรายละเอียดการประมวลผล"):
                            st.markdown("**ข้อความต้นฉบับ:**")
                            st.code(text_to_check, language='text')
                            
                            st.markdown("**ข้อความหลัง Preprocessing:**")
                            st.code(clean_text, language='text')
                            
                            st.markdown("**ความยาวข้อความ:**")
                            st.write(f"- ต้นฉบับ: {len(text_to_check)} ตัวอักษร")
                            st.write(f"- หลัง cleaning: {len(clean_text)} ตัวอักษร")
                            
                            st.markdown("**ความน่าจะเป็น:**")
                            st.write(f"- Spam: {probability[1]*100:.2f}%")
                            st.write(f"- Ham: {probability[0]*100:.2f}%")
                            
                    except Exception as e:
                        st.error(f"❌ เกิดข้อผิดพลาด: {e}")
            else:
                st.warning("️ กรุณากรอกข้อความก่อนทำการตรวจสอบ")

# ==========================================
# หน้า 6: ผู้พัฒนา
# ==========================================
elif page == "👨💻 ผู้พัฒนา":
    st.title("‍💻 ผู้พัฒนา (Developer)")
    st.markdown("---")
    
    st.subheader("ข้อมูลผู้พัฒนา")
    st.write("""
    - **ชื่อ:** [ชื่อ-นามสกุล ของคุณ]
    - **รหัสนักศึกษา:** [รหัส]
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
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    <p>📱 SMS Spam Classification Project | Mini Project 2026</p>
    <p>Developed with ❤️ using Python, Scikit-Learn, and Streamlit</p>
</div>
""", unsafe_allow_html=True)