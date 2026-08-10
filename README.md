# NLP-SubTest1-2
SubTest1-2

# Restaurant Review Analyzer 🍜
โปรเจกต์นี้เป็น Web Application ที่พัฒนาด้วย Python และ Streamlit สำหรับวิเคราะห์ข้อความรีวิวร้านอาหาร โดยใช้เทคนิค Natural Language Processing (NLP) เบื้องต้น

## เทคนิคที่นำมาประยุกต์ใช้
1. **Regex & Cleansing:** เซ็นเซอร์เบอร์โทรศัพท์และลดรูปคำลากเสียง (เช่น อร่อยยยย -> อร่อย)
2. **Tokenization & Normalization:** ตัดคำภาษาไทยและลบ Stopwords ด้วย PyThaiNLP
3. **NER (Named Entity Recognition):** สกัดชื่อสถานที่และองค์กรออกจากรีวิว
4. **Topic Identification:** จำแนกหัวข้อรีวิว (รสชาติอาหาร vs การบริการ) ด้วยการนับ Keyword

## วิธีการติดตั้งและรันโปรแกรม
1. Clone repository นี้ลงเครื่องของคุณ
2. ติดตั้งไลบรารีที่จำเป็นโดยใช้คำสั่ง `pip install -r requirements.txt`
3. รันแอปพลิเคชันด้วยคำสั่ง `streamlit run app.py`

## Prompt AI ที่ใช้เป็นผู้ช่วยในงานนี้
> "ช่วยเขียนโครงร่างแอป Streamlit สำหรับวิเคราะห์ข้อความรีวิวร้านอาหาร โดยใช้ PyThaiNLP สกัดคำและทำ NER พร้อมทั้งใช้วิธีจัดกลุ่มหัวข้อแบบง่ายๆ ด้วย if-else ให้โค้ดดูเป็นผู้เริ่มต้นอ่านเข้าใจง่าย"