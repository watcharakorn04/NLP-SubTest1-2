import streamlit as st
import re
from pythainlp.tokenize import word_tokenize
from pythainlp.corpus.common import thai_stopwords
from pythainlp.tag import pos_tag, NER
import pandas as pd

st.set_page_config(page_title="ระบบวิเคราะห์รีวิวร้านอาหาร", page_icon="🍜")

# โหลดโมเดล NER ของภาษาไทยเตรียมไว้
@st.cache_resource
def load_ner():
    return NER("thainer")

ner_model = load_ner()

st.title("🍜 ระบบวิเคราะห์รีวิวร้านอาหาร")
st.write("แอปพลิเคชันสำหรับสกัดข้อมูลสำคัญและจัดกลุ่มรีวิวร้านอาหารด้วย NLP")

# รับข้อความจากผู้ใช้
raw_text = st.text_area(
    "📝 พิมพ์ข้อความรีวิวที่นี่:", 
    "ร้านนี้อร่อยมากครับ บรรยากาศดีแอร์เย็น พนักงานบริการดีเยี่ยม อยู่ที่สยามพารากอน โทร 081-234-5678 ดูรีวิวเพิ่มเติมที่ https://example.com"
)

if st.button("ประมวลผลข้อความ"):
    st.markdown("---")
    
    # 1. Regex & Cleansing
    st.subheader("1. Regex & Cleansing (การทำความสะอาดข้อมูล)")
    clean_text = re.sub(r'0\d{1,2}-\d{3}-\d{4}', '[เซ็นเซอร์เบอร์โทร]', raw_text)
    clean_text = re.sub(r'https?://\S+|www\.\S+', '[เซ็นเซอร์ลิงก์]', clean_text)
    clean_text = re.sub(r'(.)\1{2,}', r'\1', clean_text)  # ลดคำลากเสียง
    st.info(f"**ข้อความหลัง Clean:** {clean_text}")

    # 2. Tokenization & Normalization
    st.subheader("2. Tokenization & Normalization")
    tokens = word_tokenize(clean_text, engine="newmm")
    stopwords = list(thai_stopwords())
    clean_tokens = [w.strip() for w in tokens if w.strip() and w not in stopwords]
    st.write("**คำที่ได้จากการตัดและลบ Stopwords:**")
    st.write(clean_tokens)

    # 3. POS Tagging & NER
    st.subheader("3. POS Tagging & NER (การสกัดเอนทิตีและชนิดคำ)")
    
    # POS Tagging
    pos_result = pos_tag(clean_tokens, corpus="pud")
    nouns = [word for word, pos in pos_result if pos == 'NOUN']
    adjectives = [word for word, pos in pos_result if pos == 'ADJ']
    
    # NER Tagging (แก้ไขการ Unpack ให้ปลอดภัย)
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

    col1, col2 = st.columns(2)
    with col1:
        st.write("**คำนาม (Nouns):**", nouns if nouns else "ไม่พบ")
        st.write("**คำคุณศัพท์ (Adjectives):**", adjectives if adjectives else "ไม่พบ")
    with col2:
        st.write("**เอนทิตีที่พบ (NER):**")
        if entities:
            st.dataframe(pd.DataFrame(entities), use_container_width=True)
        else:
            st.write("ไม่พบเอนทิตี")

    # 4. Topic Identification
    st.subheader("4. Topic Identification (การจัดกลุ่มหัวข้อ)")
    topics = []
    food_kw = ["อร่อย", "รสชาติ", "เค็ม", "หวาน", "อาหาร", "เมนู", "ผัดไทย"]
    service_kw = ["พนักงาน", "บริการ", "ช้า", "เร็ว", "รอ", "สุภาพ"]
    place_kw = ["บรรยากาศ", "ร้าน", "แอร์", "ที่จอดรถ", "สยาม"]

    text_concat = "".join(clean_tokens)
    if any(kw in text_concat for kw in food_kw):
        topics.append("🍲 รสชาติ/อาหาร")
    if any(kw in text_concat for kw in service_kw):
        topics.append("🤵 การบริการ")
    if any(kw in text_concat for kw in place_kw):
        topics.append("🏪 บรรยากาศ/สถานที่")

    if topics:
        st.success(f"**หัวข้อที่พบในรีวิว:** {', '.join(topics)}")
    else:
        st.warning("ไม่สามารถระบุหัวข้อที่ชัดเจนได้")