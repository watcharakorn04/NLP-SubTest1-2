import streamlit as st
import re
import pandas as pd
from pythainlp.tokenize import word_tokenize
from pythainlp.corpus import thai_stopwords
from pythainlp.util import normalize
from pythainlp.tag import NER

# โหลดโมเดล NER ของภาษาไทยเตรียมไว้
ner_model = NER("thainer")

st.title("🍜 ระบบวิเคราะห์รีวิวร้านอาหาร")
st.write("แอปพลิเคชันสำหรับดึงข้อมูลสำคัญและจัดกลุ่มรีวิวร้านอาหาร")

# สร้างกล่องรับข้อความจากผู้ใช้ พร้อมข้อความจำลองสำหรับทดสอบ (Test Data)
default_text = "ไปกินข้าวที่ร้านอาหารเชียงใหม่มาเมื่อวาน เมนูผัดไทยกุ้งสดอร่อยมากกกก บริการดีเยี่ยม สนใจจองโต๊ะโทร 081-234-5678 แนะนำเลยยยย"
user_input = st.text_area("ใส่ข้อความรีวิวที่นี่:", default_text, height=150)

if st.button("เริ่มวิเคราะห์ข้อมูล"):
    
    # ---------------------------------------------------------
    # 1. Regex & Cleansing (ทำความสะอาดข้อความ)
    # ---------------------------------------------------------
    st.subheader("1. การทำความสะอาดข้อมูล (Cleansing)")
    
    # ลบเบอร์โทรศัพท์ (เซ็นเซอร์ข้อมูลส่วนตัว)
    text_no_phone = re.sub(r'\d{2,3}-\d{3,4}-\d{4}', '[เซ็นเซอร์เบอร์โทร]', user_input)
    
    # ลดตัวอักษรที่พิมพ์ลากเสียงยาว (เช่น มากกกก -> มาก)
    cleaned_text = re.sub(r'(.)\1{2,}', r'\1', text_no_phone)
    
    st.info(f"ข้อความหลัง Clean: {cleaned_text}")

    # ---------------------------------------------------------
    # 2. Tokenization & Normalization (ตัดคำ & ลบคำฟุ่มเฟือย)
    # ---------------------------------------------------------
    st.subheader("2. การตัดคำ (Tokenization)")
    
    # ปรับรูปสระให้ถูกต้อง
    norm_text = normalize(cleaned_text)
    
    # ตัดคำ
    tokens = word_tokenize(norm_text, engine='newmm')
    
    # ลบ Stopwords และช่องว่าง
    stopwords = thai_stopwords()
    filtered_tokens = []
    for word in tokens:
        word = word.strip()
        if word != "" and word not in stopwords:
            filtered_tokens.append(word)
            
    st.write(f"**คำที่สกัดได้:** {', '.join(filtered_tokens)}")

    # ---------------------------------------------------------
    # 3. POS & NER (สกัด Named Entities)
    # ---------------------------------------------------------
    st.subheader("3. การสกัดเอนทิตี (NER)")
    
    # ใช้ PyThaiNLP ดึงข้อมูล คน, สถานที่, องค์กร
    ner_result = ner_model.tag(cleaned_text)
    
    # กรองเอาเฉพาะคำที่เป็น Entity (ไม่เอาแท็ก 'O')
    entities = []
    for word, pos, tag in ner_result:
        if tag != 'O':
            entities.append({"คำศัพท์": word, "ประเภท (Tag)": tag})
            
    if len(entities) > 0:
        st.table(pd.DataFrame(entities))
    else:
        st.write("ไม่พบชื่อคน สถานที่ หรือองค์กรในข้อความนี้")

    # ---------------------------------------------------------
    # 4. Topic Identification (จัดกลุ่มหัวข้อ)
    # ---------------------------------------------------------
    st.subheader("4. การจำแนกหัวข้อ (Topic Identification)")
    
    # ใช้วิธีนับคีย์เวิร์ดแบบง่ายๆ (Rule-based) เพื่อจัดกลุ่ม
    food_keywords = ["อร่อย", "เมนู", "รสชาติ", "อาหาร", "ผัดไทย", "กุ้งสด", "หวาน", "เค็ม"]
    service_keywords = ["บริการ", "พนักงาน", "รอ", "ช้า", "เร็ว", "บรรยากาศ", "ที่จอดรถ"]
    
    food_score = sum([1 for word in filtered_tokens if word in food_keywords])
    service_score = sum([1 for word in filtered_tokens if word in service_keywords])
    
    if food_score > service_score:
        topic = "🍽️ รีวิวเน้นรสชาติอาหาร"
    elif service_score > food_score:
        topic = "👨‍🍳 รีวิวเน้นการบริการ/บรรยากาศ"
    elif food_score > 0 and service_score > 0:
        topic = "⭐ รีวิวผสมผสาน (อาหารและการบริการ)"
    else:
        topic = "📝 รีวิวทั่วไป"
        
    st.success(f"ระบบจัดกลุ่มรีวิวนี้อยู่ในหมวดหมู่: **{topic}**")