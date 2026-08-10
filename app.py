import streamlit as st
import re
from pythainlp.tokenize import word_tokenize
from pythainlp.corpus.common import thai_stopwords
from pythainlp.tag import pos_tag, NER
import pandas as pd

# 1. ตั้งค่าหน้าเพจให้กว้างขึ้น
st.set_page_config(page_title="Restaurant Review NLP", page_icon="🍽️", layout="wide")

@st.cache_resource
def load_ner():
    return NER("thainer")

ner_model = load_ner()

# ส่วนหัวของเว็บ
st.title("🍽️ ระบบวิเคราะห์และคัดกรองรีวิวร้านอาหาร")
st.markdown("**แอปพลิเคชันสำหรับสกัดข้อมูลสำคัญและจัดกลุ่มรีวิวร้านอาหารด้วยเทคนิค NLP**")
st.divider() # เส้นคั่น

# 2. จัดกลุ่มกล่องรับข้อความให้อยู่ใน Container
with st.container(border=True):
    raw_text = st.text_area(
        "📝 พิมพ์ข้อความรีวิวที่ต้องการวิเคราะห์:", 
        "ร้านนี้อร่อยมากครับ บรรยากาศดีแอร์เย็น พนักงานบริการดีเยี่ยม อยู่ที่สยามพารากอน โทร 081-234-5678 ดูรีวิวเพิ่มเติมที่ https://example.com",
        height=150
    )
    
    # ทำให้ปุ่มกดใหญ่และเป็นสีหลัก (Primary)
    submit_btn = st.button("🚀 เริ่มการประมวลผลข้อความ", type="primary", use_container_width=True)

if submit_btn:
    # 3. ใส่ Spinner ระหว่างรอประมวลผล
    with st.spinner("กำลังวิเคราะห์ข้อมูลด้วย AI..."):
        
        # --- ประมวลผลเบื้องหลัง ---
        clean_text = re.sub(r'0\d{1,2}-\d{3}-\d{4}', '[เซ็นเซอร์เบอร์โทร]', raw_text)
        clean_text = re.sub(r'https?://\S+|www\.\S+', '[เซ็นเซอร์ลิงก์]', clean_text)
        clean_text = re.sub(r'(.)\1{2,}', r'\1', clean_text)
        
        tokens = word_tokenize(clean_text, engine="newmm")
        stopwords = list(thai_stopwords())
        clean_tokens = [w.strip() for w in tokens if w.strip() and w not in stopwords]
        
        pos_result = pos_tag(clean_tokens, corpus="pud")
        nouns = [word for word, pos in pos_result if pos == 'NOUN']
        adjectives = [word for word, pos in pos_result if pos == 'ADJ']
        
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

        topics = []
        food_kw = ["อร่อย", "รสชาติ", "เค็ม", "หวาน", "อาหาร", "เมนู", "ผัดไทย"]
        service_kw = ["พนักงาน", "บริการ", "ช้า", "เร็ว", "รอ", "สุภาพ"]
        place_kw = ["บรรยากาศ", "ร้าน", "แอร์", "ที่จอดรถ", "สยาม"]
        text_concat = "".join(clean_tokens)
        if any(kw in text_concat for kw in food_kw): topics.append("🍲 รสชาติ/อาหาร")
        if any(kw in text_concat for kw in service_kw): topics.append("🤵 การบริการ")
        if any(kw in text_concat for kw in place_kw): topics.append("🏪 บรรยากาศ/สถานที่")

        # 4. แบ่งการแสดงผลออกเป็น Tabs
        st.success("✨ ประมวลผลเสร็จสมบูรณ์!")
        
        tab1, tab2, tab3, tab4 = st.tabs(["🧹 1. Regex & Cleansing", "✂️ 2. Tokenization", "🏷️ 3. POS & NER", "💡 4. Topic ID"])
        
        with tab1:
            st.subheader("การทำความสะอาดข้อมูล")
            st.info(f"**ข้อความดั้งเดิม:** {raw_text}")
            st.success(f"**ข้อความหลัง Clean:** {clean_text}")
            st.caption("ระบบได้ทำการเซ็นเซอร์เบอร์โทรศัพท์ ลิงก์ และลดคำลากเสียงเรียบร้อยแล้ว")
            
        with tab2:
            st.subheader("ตัดคำและลบ Stopwords")
            st.write(clean_tokens)
            st.metric(label="จำนวนคำศัพท์ที่ใช้ได้", value=f"{len(clean_tokens)} คำ")
            
        with tab3:
            st.subheader("การสกัดเอนทิตีและชนิดคำ")
            col_pos, col_ner = st.columns(2)
            with col_pos:
                st.markdown("**คำนาม (Nouns):**")
                st.write(nouns if nouns else "ไม่พบ")
                st.markdown("**คำคุณศัพท์ (Adjectives):**")
                st.write(adjectives if adjectives else "ไม่พบ")
            with col_ner:
                st.markdown("**เอนทิตีที่พบ (Named Entities):**")
                if entities:
                    st.dataframe(pd.DataFrame(entities), use_container_width=True, hide_index=True)
                else:
                    st.write("ไม่พบเอนทิตี")
                    
        with tab4:
            st.subheader("การจัดกลุ่มหัวข้อ (Topic Identification)")
            if topics:
                for t in topics:
                    st.toast(f"พบหัวข้อ: {t}") # ขึ้นแจ้งเตือนเล็กๆ มุมขวาล่าง
                st.success(f"**หัวข้อหลักของรีวิวนี้คือ:** {', '.join(topics)}")
            else:
                st.warning("ไม่สามารถระบุหัวข้อที่ชัดเจนได้จากคีย์เวิร์ดที่มี")