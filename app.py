import streamlit as st
import re
from pythainlp.tokenize import word_tokenize
from pythainlp.corpus.common import thai_stopwords
from pythainlp.tag import pos_tag, NER
import pandas as pd
from collections import Counter

# 1. ตั้งค่าหน้าเพจแบบกว้าง
st.set_page_config(page_title="NLP Dashboard", page_icon="✨", layout="wide")

# 2. แทรก CSS สไตล์ Modern & Minimal
st.markdown("""
    <style>
    /* ปรับแต่งปุ่มและพื้นหลัง */
    .stButton>button {
        border-radius: 8px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    /* กล่องข้อความสไตล์การ์ดโมเดิร์น */
    .modern-card { 
        background-color: #ffffff; 
        padding: 20px; 
        border-radius: 12px; 
        border-left: 6px solid #4F46E5; /* สี Indigo Accent */
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        margin-bottom: 20px;
        color: #333333;
    }
    /* ตกแต่ง Tag หมวดหมู่ */
    .topic-tag {
        display: inline-block;
        background: linear-gradient(135deg, #4F46E5, #7C3AED);
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 600;
        margin-right: 10px;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_ner():
    return NER("thainer")

ner_model = load_ner()

# --- Header Section ---
st.title("✨ NLP Review Analytics")
st.markdown("**แพลตฟอร์มวิเคราะห์และสกัดข้อมูลเชิงลึกจากข้อความรีวิวแบบอัตโนมัติ**")
st.divider()

# --- Input Section ---
with st.container():
    raw_text = st.text_area(
        "💬 ระบุข้อความรีวิวที่ต้องการวิเคราะห์:", 
        "ร้านนี้อร่อยมากครับ บรรยากาศดีแอร์เย็น พนักงานบริการดีเยี่ยม อยู่ที่สยามพารากอน โทร 081-234-5678 ดูรีวิวเพิ่มเติมที่ https://example.com",
        height=100
    )
    submit_btn = st.button("🚀 เริ่มการประมวลผล AI", type="primary", use_container_width=True)

if submit_btn:
    with st.spinner("กำลังวิเคราะห์ข้อมูล..."):
        
        # --- กระบวนการ NLP (Backend) ---
        clean_text = re.sub(r'0\d{1,2}-\d{3}-\d{4}', '[เซ็นเซอร์เบอร์โทร]', raw_text)
        clean_text = re.sub(r'https?://\S+|www\.\S+', '[เซ็นเซอร์ลิงก์]', clean_text)
        clean_text = re.sub(r'(.)\1{2,}', r'\1', clean_text)
        
        tokens = word_tokenize(clean_text, engine="newmm")
        stopwords = list(thai_stopwords())
        clean_tokens = [w.strip() for w in tokens if w.strip() and w not in stopwords]
        
        pos_result = pos_tag(clean_tokens, corpus="pud")
        pos_df = pd.DataFrame(pos_result, columns=["คำศัพท์", "ชนิดคำ (POS)"])
        
        ner_result = ner_model.tag(clean_text)
        entities = []
        for item in ner_result:
            if len(item) == 2: word, tag = item
            elif len(item) == 3: word, _, tag = item
            else: continue
            if tag != 'O': entities.append({"คำศัพท์": word, "ประเภท (Tag)": tag})
        ner_df = pd.DataFrame(entities)

        topics_found = {}
        food_kw = ["อร่อย", "รสชาติ", "เค็ม", "หวาน", "อาหาร", "เมนู", "ผัดไทย"]
        service_kw = ["พนักงาน", "บริการ", "ช้า", "เร็ว", "รอ", "สุภาพ"]
        place_kw = ["บรรยากาศ", "ร้าน", "แอร์", "ที่จอดรถ", "สยาม"]
        
        text_concat = "".join(clean_tokens)
        if food_match := [k for k in food_kw if k in text_concat]: topics_found["🍲 รสชาติ/อาหาร"] = food_match
        if service_match := [k for k in service_kw if k in text_concat]: topics_found["🤵 การบริการ"] = service_match
        if place_match := [k for k in place_kw if k in text_concat]: topics_found["🏪 บรรยากาศ/สถานที่"] = place_match

        # --- Dashboard UI (Frontend) ---
        st.markdown("<br>", unsafe_allow_html=True)
        
        # 1. แถบสรุปสถิติ (Metrics)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("จำนวนคำ (ดั้งเดิม)", f"{len(tokens)} คำ")
        col2.metric("จำนวนคำ (หลังคลีน)", f"{len(clean_tokens)} คำ")
        col3.metric("ชนิดคำที่พบ (POS)", f"{len(pos_df)} จุด")
        col4.metric("เอนทิตีสำคัญ (NER)", f"{len(entities)} จุด")
        
        # 2. กล่องแสดงหมวดหมู่ (Modern Card)
        st.markdown("### 🎯 หมวดหมู่รีวิวที่ประเมินได้")
        if topics_found:
            tags_html = "".join([f"<span class='topic-tag'>{topic}</span>" for topic in topics_found.keys()])
            st.markdown(f"<div class='modern-card'>{tags_html}</div>", unsafe_allow_html=True)
        else:
            st.warning("ไม่สามารถระบุหมวดหมู่ได้ชัดเจน")

        # 3. จัดกลุ่มการแสดงผลข้อมูลแบบละเอียดด้วย Tabs
        st.markdown("### 📊 ข้อมูลเชิงลึก (Deep Dive)")
        tab1, tab2, tab3 = st.tabs(["📝 ข้อมูลที่ถูกคลีน", "🏷️ ตาราง POS & NER", "📈 กราฟความถี่คำ"])
        
        with tab1:
            st.markdown(f"<div class='modern-card'><strong>ข้อความพร้อมใช้งาน:</strong><br>{clean_text}</div>", unsafe_allow_html=True)
            with st.expander("ดูรายการคำที่ถูกตัด (Tokens & Stopwords removed)"):
                st.write(clean_tokens)

        with tab2:
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**รายการชนิดคำทั้งหมด (Part of Speech)**")
                st.dataframe(pos_df, use_container_width=True, height=300, hide_index=True)
            with col_b:
                st.markdown("**เอนทิตีที่พบ (Named Entity Recognition)**")
                if not ner_df.empty:
                    st.dataframe(ner_df, use_container_width=True, height=300, hide_index=True)
                else:
                    st.info("ไม่พบเอนทิตีเฉพาะเจาะจงในข้อความนี้")

        with tab3:
            st.markdown("**กราฟแสดงคำที่พบบ่อยที่สุด 10 อันดับแรก**")
            if clean_tokens:
                word_counts = Counter(clean_tokens)
                top_words = pd.DataFrame(word_counts.most_common(10), columns=['คำศัพท์', 'ความถี่']).set_index('คำศัพท์')
                st.bar_chart(top_words, color="#4F46E5")
            else:
                st.write("ไม่มีข้อมูลเพียงพอสำหรับสร้างกราฟ")