import streamlit as st
import re
from pythainlp.tokenize import word_tokenize
from pythainlp.corpus.common import thai_stopwords
from pythainlp.tag import pos_tag, NER
import pandas as pd
from collections import Counter

# 1. ตั้งค่าหน้าเพจแบบกว้าง
st.set_page_config(page_title="Restaurant Review NLP", page_icon="🍽️", layout="wide")

# เพิ่ม CSS ปรับแต่งความสวยงามเล็กน้อย
st.markdown("""
    <style>
    .highlight-box { background-color: #f0f2f6; padding: 15px; border-radius: 10px; border-left: 5px solid #ff4b4b; }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_ner():
    return NER("thainer")

ner_model = load_ner()

# Header
st.title("🍽️ ระบบวิเคราะห์และคัดกรองรีวิวร้านอาหาร (NLP Dashboard)")
st.markdown("วิเคราะห์ข้อมูลเชิงลึก สกัดคำสำคัญ และจัดหมวดหมู่โดยอัตโนมัติ")
st.divider()

# รับข้อมูล
with st.container(border=True):
    raw_text = st.text_area(
        "📝 พิมพ์ข้อความรีวิวที่ต้องการวิเคราะห์:", 
        "ร้านนี้อร่อยมากครับ บรรยากาศดีแอร์เย็น พนักงานบริการดีเยี่ยม อยู่ที่สยามพารากอน โทร 081-234-5678 ดูรีวิวเพิ่มเติมที่ https://example.com",
        height=120
    )
    submit_btn = st.button("🚀 เริ่มการประมวลผลข้อความ", type="primary", use_container_width=True)

if submit_btn:
    with st.spinner("กำลังวิเคราะห์ข้อมูลด้วย AI..."):
        
        # --- ประมวลผลเบื้องหลัง ---
        # 1. Cleansing
        clean_text = re.sub(r'0\d{1,2}-\d{3}-\d{4}', '[เซ็นเซอร์เบอร์โทร]', raw_text)
        clean_text = re.sub(r'https?://\S+|www\.\S+', '[เซ็นเซอร์ลิงก์]', clean_text)
        clean_text = re.sub(r'(.)\1{2,}', r'\1', clean_text)
        
        # 2. Tokenize & Stopwords
        tokens = word_tokenize(clean_text, engine="newmm")
        stopwords = list(thai_stopwords())
        clean_tokens = [w.strip() for w in tokens if w.strip() and w not in stopwords]
        
        # 3. POS Tagging
        pos_result = pos_tag(clean_tokens, corpus="pud")
        pos_df = pd.DataFrame(pos_result, columns=["คำศัพท์", "ชนิดคำ (POS)"])
        
        # 4. NER Tagging
        ner_result = ner_model.tag(clean_text)
        entities = []
        for item in ner_result:
            if len(item) == 2:
                word, tag = item
            elif len(item) == 3:
                word, _, tag = item
            else:
                continue
            if tag != 'O':
                entities.append({"คำศัพท์": word, "ประเภท (Tag)": tag})
        ner_df = pd.DataFrame(entities)

        # 5. Topic Identification (จับคู่คีย์เวิร์ดแบบละเอียด)
        topics_found = {}
        food_kw = ["อร่อย", "รสชาติ", "เค็ม", "หวาน", "อาหาร", "เมนู", "ผัดไทย"]
        service_kw = ["พนักงาน", "บริการ", "ช้า", "เร็ว", "รอ", "สุภาพ"]
        place_kw = ["บรรยากาศ", "ร้าน", "แอร์", "ที่จอดรถ", "สยาม"]
        
        text_concat = "".join(clean_tokens)
        
        food_match = [k for k in food_kw if k in text_concat]
        if food_match: topics_found["🍲 รสชาติ/อาหาร"] = food_match
            
        service_match = [k for k in service_kw if k in text_concat]
        if service_match: topics_found["🤵 การบริการ"] = service_match
            
        place_match = [k for k in place_kw if k in text_concat]
        if place_match: topics_found["🏪 บรรยากาศ/สถานที่"] = place_match

        # --- ส่วนแสดงผล UI (สวยงามและครบถ้วน) ---
        st.success("✨ ประมวลผลเสร็จสมบูรณ์!")
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 สรุปภาพรวม (Overview)", 
            "✂️ การตัดคำ (Tokenization)", 
            "🏷️ ชนิดคำและเอนทิตี (POS & NER)", 
            "🎯 เกณฑ์การจัดหัวข้อ"
        ])
        
        with tab1:
            st.markdown(f"<div class='highlight-box'><strong>ข้อความที่วิเคราะห์:</strong><br>{raw_text}</div><br>", unsafe_allow_html=True)
            
            c1, c2, c3 = st.columns(3)
            c1.metric("จำนวนคำทั้งหมด (ก่อนกรอง)", f"{len(tokens)} คำ")
            c2.metric("จำนวนคำ (หลังลบ Stopwords)", f"{len(clean_tokens)} คำ")
            c3.metric("เอนทิตีที่พบ (NER)", f"{len(entities)} จุด")
            
            st.markdown("### 📌 หมวดหมู่รีวิวที่พบ")
            if topics_found:
                for topic, kws in topics_found.items():
                    st.info(f"**{topic}** (วิเคราะห์จากคำว่า: {', '.join(kws)})")
            else:
                st.warning("ไม่สามารถจัดหมวดหมู่ได้ (ไม่พบคีย์เวิร์ดที่ตั้งไว้)")

        with tab2:
            st.markdown("### 🧹 ข้อความ## 🛠️ ยินดีให้บริการครับ!")