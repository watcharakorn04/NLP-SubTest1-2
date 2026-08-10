import streamlit as st
import re
from pythainlp.tokenize import word_tokenize
from pythainlp.corpus.common import thai_stopwords
from pythainlp.tag import pos_tag, NER
import pandas as pd

# โหลด NER Engine (ใช้ thainer)
@st.cache_resource
def load_ner():
    return NER("thainer")

ner = load_ner()
stopwords = list(thai_stopwords())

st.title("🍽️ ระบบคัดกรองรีวิวร้านอาหาร")
st.write("แอปพลิเคชันนี้ใช้เทคนิค NLP ในการสกัดข้อมูลและวิเคราะห์รีวิวร้านอาหาร")

# รับข้อความจากผู้ใช้
raw_text = st.text_area("📝 พิมพ์ข้อความรีวิวที่นี่:", 
                        "ร้านนี้อร่อยมากครับ บรรยากาศดีแอร์เย็น พนักงานบริการดีเยี่ยม อยู่ที่สยามพารากอน โทร 081-234-5678 ดูรีวิวเพิ่มเติมที่ https://example.com")

if st.button("ประมวลผลข้อความ"):
    st.markdown("---")
    
    # 1. Regex & Cleansing: ลบเบอร์โทรศัพท์และลิงก์
    st.subheader("1. Regex & Cleansing (การทำความสะอาดข้อมูล)")
    clean_text = re.sub(r'0\d{1,2}-\d{3}-\d{4}', '[เบอร์โทรศัพท์ถูกซ่อน]', raw_text)
    clean_text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '[ลิงก์ถูกซ่อน]', clean_text)
    st.write("**ข้อความหลังทำความสะอาด:**", clean_text)

    # 2. Tokenization & Normalization: ตัดคำและลบ Stopwords
    st.subheader("2. Tokenization & Normalization")
    tokens = word_tokenize(clean_text, engine="newmm")
    clean_tokens = [word for word in tokens if word not in stopwords and word.strip() != ""]
    st.write("**คำที่ได้จากการตัดและลบ Stopwords:**")
    st.write(clean_tokens)

    # 3. POS & NER: สกัด Named Entities และ Part-of-Speech
    st.subheader("3. POS & NER (การสกัดเอนทิตีและชนิดคำ)")
    
    # POS Tagging (ดึงเฉพาะคำนามและคำคุณศัพท์มาดู Keyword)
    pos_result = pos_tag(clean_tokens, corpus="pud")
    keywords = [word for word, pos in pos_result if pos in ['NOUN', 'ADJ']]
    
    # NER Tagging
    ner_result = ner.tag(clean_text)
    entities = []
    
    # แก้ปัญหา Error จากภาพ โดยดักจับความยาวของ Tuple ให้รองรับทั้ง 2 และ 3 ค่า
    for item in ner_result:
        if len(item) == 3:
            word, pos, tag = item
        elif len(item) == 2:
            word, tag = item
        else:
            continue
            
        if tag != 'O': # กรองเฉพาะคำที่เป็น Entity (ไม่เอา 'O' ที่แปลว่า Other)
            entities.append(f"{word} ({tag})")
            
    col1, col2 = st.columns(2)
    with col1:
        st.write("**คำสำคัญ (Nouns/Adjectives):**")
        st.write(keywords[:10]) # แสดงแค่ 10 คำแรก
    with col2:
        st.write("**Named Entities (สถานที่, องค์กร ฯลฯ):**")
        st.write(entities if entities else "ไม่พบเอนทิตี")

    # 4. Topic Identification: จัดกลุ่มหัวข้อหลักของข้อความ
    st.subheader("4. Topic Identification (การจัดกลุ่มหัวข้อ)")
    topics = []
    food_keywords = ["อร่อย", "รสชาติ", "เค็ม", "หวาน", "เปรี้ยว", "เผ็ด", "อาหาร", "เมนู"]
    service_keywords = ["พนักงาน", "บริการ", "ช้า", "เร็ว", "รอ", "สุภาพ"]
    place_keywords = ["บรรยากาศ", "ร้าน", "แอร์", "ที่จอดรถ", "สะอาด", "กว้าง"]

    text_to_check = "".join(clean_tokens)
    if any(kw in text_to_check for kw in food_keywords):
        topics.append("🍲 รสชาติ/อาหาร")
    if any(kw in text_to_check for kw in service_keywords):
        topics.append("🤵 การบริการ")
    if any(kw in text_to_check for kw in place_keywords):
        topics.append("🏪 บรรยากาศ/สถานที่")

    if topics:
        st.success(f"หัวข้อที่พบในรีวิวนี้: {', '.join(topics)}")
    else:
        st.info("ไม่สามารถระบุหัวข้อที่ชัดเจนได้จากรีวิวนี้")