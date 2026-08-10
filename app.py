import streamlit as st
import re
import pandas as pd

from pythainlp.tokenize import word_tokenize
from pythainlp.corpus.common import thai_stopwords
from pythainlp.tag import pos_tag, NER


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Restaurant Review NLP",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

    /* ---------- Global ---------- */
    .main {
        background-color: #f8fafc;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    /* ---------- Header ---------- */
    .hero {
        padding: 2rem;
        border-radius: 20px;
        background: linear-gradient(
            135deg,
            #ff6b35 0%,
            #ff8c42 50%,
            #ffb347 100%
        );
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 30px rgba(255, 107, 53, 0.20);
    }

    .hero h1 {
        color: white;
        margin-bottom: 0.4rem;
        font-size: 2.4rem;
    }

    .hero p {
        color: rgba(255,255,255,0.90);
        font-size: 1.05rem;
        margin-bottom: 0;
    }

    /* ---------- Cards ---------- */
    .info-card {
        background: white;
        padding: 1.2rem;
        border-radius: 16px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 4px 15px rgba(0,0,0,0.04);
        height: 100%;
    }

    .info-card-title {
        font-size: 0.9rem;
        color: #64748b;
        margin-bottom: 0.3rem;
    }

    .info-card-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1e293b;
    }

    /* ---------- Section ---------- */
    .section-title {
        font-size: 1.35rem;
        font-weight: 700;
        color: #1e293b;
        margin-top: 0.5rem;
        margin-bottom: 0.8rem;
    }

    /* ---------- Result Box ---------- */
    .result-box {
        padding: 1rem 1.2rem;
        border-radius: 14px;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        margin-bottom: 0.8rem;
    }

    /* ---------- Topic Badge ---------- */
    .topic-badge {
        display: inline-block;
        padding: 0.45rem 0.8rem;
        margin: 0.25rem;
        border-radius: 999px;
        background: #fff7ed;
        color: #c2410c;
        border: 1px solid #fed7aa;
        font-weight: 600;
    }

    /* ---------- Footer ---------- */
    .footer {
        text-align: center;
        color: #94a3b8;
        padding-top: 2rem;
        font-size: 0.85rem;
    }

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_ner():
    return NER("thainer")


@st.cache_data
def load_stopwords():
    return set(thai_stopwords())


ner_model = load_ner()
stopwords = load_stopwords()


# =========================================================
# FUNCTIONS
# =========================================================

def clean_text(text):
    """
    ทำความสะอาดข้อความ
    """

    # Mask เบอร์โทรศัพท์
    cleaned = re.sub(
        r'0\d{1,2}[-\s]?\d{3}[-\s]?\d{4}',
        '[เซ็นเซอร์เบอร์โทร]',
        text
    )

    # Mask URL
    cleaned = re.sub(
        r'https?://\S+|www\.\S+',
        '[เซ็นเซอร์ลิงก์]',
        cleaned
    )

    # ลดการลากเสียง เช่น อร่อยยยยย -> อร่อย
    cleaned = re.sub(
        r'(.)\1{2,}',
        r'\1',
        cleaned
    )

    # ลดช่องว่างซ้ำ
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    return cleaned


def tokenize_text(text):
    """
    ตัดคำ + ลบ Stopwords
    """

    tokens = word_tokenize(
        text,
        engine="newmm"
    )

    clean_tokens = [
        word.strip()
        for word in tokens
        if word.strip()
        and word not in stopwords
    ]

    return clean_tokens


def analyze_pos(tokens):
    """
    วิเคราะห์ Part-of-Speech
    """

    pos_result = pos_tag(
        tokens,
        corpus="pud"
    )

    nouns = [
        word
        for word, pos in pos_result
        if pos == "NOUN"
    ]

    adjectives = [
        word
        for word, pos in pos_result
        if pos == "ADJ"
    ]

    verbs = [
        word
        for word, pos in pos_result
        if pos == "VERB"
    ]

    return pos_result, nouns, adjectives, verbs


def analyze_ner(text):
    """
    วิเคราะห์ Named Entity Recognition
    """

    ner_result = ner_model.tag(text)

    entities = []

    for item in ner_result:

        if len(item) == 2:
            word, tag = item

        elif len(item) == 3:
            word, _, tag = item

        else:
            continue

        if tag != "O":
            entities.append({
                "คำศัพท์": word,
                "ประเภท": tag
            })

    return entities


def analyze_topics(tokens):
    """
    วิเคราะห์หัวข้อจาก Keyword
    """

    food_keywords = [
        "อร่อย",
        "รสชาติ",
        "เค็ม",
        "หวาน",
        "เผ็ด",
        "อาหาร",
        "เมนู",
        "ผัดไทย",
        "ก๋วยเตี๋ยว",
        "ข้าว",
        "หมู",
        "ไก่",
        "ปลา"
    ]

    service_keywords = [
        "พนักงาน",
        "บริการ",
        "ช้า",
        "เร็ว",
        "รอ",
        "สุภาพ",
        "บริการดี",
        "บริการแย่"
    ]

    place_keywords = [
        "บรรยากาศ",
        "ร้าน",
        "แอร์",
        "ที่จอดรถ",
        "สยาม",
        "ทำเล",
        "สถานที่"
    ]

    price_keywords = [
        "ราคา",
        "แพง",
        "ถูก",
        "คุ้ม",
        "เงิน",
        "บาท"
    ]

    text = " ".join(tokens)

    topics = []

    if any(keyword in text for keyword in food_keywords):
        topics.append("🍲 รสชาติ / อาหาร")

    if any(keyword in text for keyword in service_keywords):
        topics.append("🤵 การบริการ")

    if any(keyword in text for keyword in place_keywords):
        topics.append("🏪 บรรยากาศ / สถานที่")

    if any(keyword in text for keyword in price_keywords):
        topics.append("💰 ราคา / ความคุ้มค่า")

    return topics


def analyze_sentiment(text):
    """
    วิเคราะห์ Sentiment แบบ Keyword-based
    """

    positive_words = [
        "อร่อย",
        "ดี",
        "ดีมาก",
        "เยี่ยม",
        "ประทับใจ",
        "คุ้ม",
        "สด",
        "อร่อยมาก",
        "บริการดี",
        "ชอบ"
    ]

    negative_words = [
        "แย่",
        "เค็ม",
        "หวานเกิน",
        "แพง",
        "ช้า",
        "ไม่อร่อย",
        "ผิดหวัง",
        "สกปรก",
        "แย่มาก",
        "รอนาน"
    ]

    positive_score = sum(
        1 for word in positive_words
        if word in text
    )

    negative_score = sum(
        1 for word in negative_words
        if word in text
    )

    if positive_score > negative_score:
        return "😊 เชิงบวก", positive_score, negative_score

    elif negative_score > positive_score:
        return "😞 เชิงลบ", positive_score, negative_score

    else:
        return "😐 เป็นกลาง", positive_score, negative_score


# =========================================================
# HEADER
# =========================================================

st.markdown("""
<div class="hero">

    <h1>🍽️ Restaurant Review NLP</h1>

    <p>
        ระบบวิเคราะห์และคัดกรองรีวิวร้านอาหารด้วย
        Natural Language Processing (NLP)
    </p>

</div>
""", unsafe_allow_html=True)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("⚙️ การตั้งค่า")

    st.markdown("""
    **Pipeline**

    🧹 Text Cleaning  
    ↓  
    ✂️ Tokenization  
    ↓  
    🏷️ POS Tagging  
    ↓  
    🔎 Named Entity Recognition  
    ↓  
    💡 Topic Identification  
    ↓  
    😊 Sentiment Analysis
    """)

    st.divider()

    st.caption(
        "ระบบนี้ใช้ PyThaiNLP สำหรับการประมวลผลภาษาไทย"
    )


# =========================================================
# INPUT
# =========================================================

st.markdown(
    '<div class="section-title">📝 ข้อความรีวิว</div>',
    unsafe_allow_html=True
)

with st.container(border=True):

    raw_text = st.text_area(
        "พิมพ์ข้อความรีวิวที่ต้องการวิเคราะห์",
        value=(
            "ร้านนี้อร่อยมากครับ บรรยากาศดี แอร์เย็น "
            "พนักงานบริการดีเยี่ยม อยู่ที่สยามพารากอน "
            "โทร 081-234-5678 "
            "ดูรีวิวเพิ่มเติมที่ https://example.com"
        ),
        height=160,
        label_visibility="collapsed"
    )

    submit_btn = st.button(
        "🚀 เริ่มการวิเคราะห์",
        type="primary",
        use_container_width=True
    )


# =========================================================
# PROCESSING
# =========================================================

if submit_btn:

    if not raw_text.strip():

        st.warning("⚠️ กรุณาพิมพ์ข้อความรีวิวก่อนเริ่มการวิเคราะห์")

        st.stop()

    with st.spinner("🔄 กำลังประมวลผลข้อความ..."):

        # -----------------------------
        # Cleaning
        # -----------------------------

        clean = clean_text(raw_text)

        # -----------------------------
        # Tokenization
        # -----------------------------

        tokens = tokenize_text(clean)

        # -----------------------------
        # POS
        # -----------------------------

        pos_result, nouns, adjectives, verbs = analyze_pos(tokens)

        # -----------------------------
        # NER
        # -----------------------------

        entities = analyze_ner(clean)

        # -----------------------------
        # Topics
        # -----------------------------

        topics = analyze_topics(tokens)

        # -----------------------------
        # Sentiment
        # -----------------------------

        sentiment, positive_score, negative_score = analyze_sentiment(
            clean
        )

    # =====================================================
    # SUCCESS
    # =====================================================

    st.success("✨ วิเคราะห์ข้อความเสร็จสมบูรณ์!")

    # =====================================================
    # SUMMARY METRICS
    # =====================================================

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            "🔤 จำนวนคำ",
            len(tokens)
        )

    with col2:
        st.metric(
            "🏷️ NER",
            len(entities)
        )

    with col3:
        st.metric(
            "📚 คำนาม",
            len(nouns)
        )

    with col4:
        st.metric(
            "✨ คำคุณศัพท์",
            len(adjectives)
        )

    with col5:
        st.metric(
            "💡 หัวข้อ",
            len(topics)
        )

    st.divider()

    # =====================================================
    # MAIN TABS
    # =====================================================

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🧹 Text Cleaning",
        "✂️ Tokenization",
        "🏷️ POS & NER",
        "💡 Topic",
        "😊 Sentiment"
    ])

    # =====================================================
    # TAB 1
    # =====================================================

    with tab1:

        st.subheader("🧹 การทำความสะอาดข้อความ")

        col1, col2 = st.columns(2)

        with col1:

            st.markdown("### 📄 ข้อความต้นฉบับ")

            st.info(raw_text)

        with col2:

            st.markdown("### ✨ ข้อความหลัง Clean")

            st.success(clean)

        st.markdown("### 🔐 ข้อมูลที่ถูก Mask")

        masked_items = []

        if "[เซ็นเซอร์เบอร์โทร]" in clean:
            masked_items.append("📱 เบอร์โทรศัพท์")

        if "[เซ็นเซอร์ลิงก์]" in clean:
            masked_items.append("🔗 URL / Link")

        if masked_items:

            for item in masked_items:
                st.markdown(
                    f'<span class="topic-badge">{item}</span>',
                    unsafe_allow_html=True
                )

        else:

            st.caption("ไม่พบข้อมูลที่ต้อง Mask")


    # =====================================================
    # TAB 2
    # =====================================================

    with tab2:

        st.subheader("✂️ Tokenization")

        st.write(
            "ระบบแบ่งข้อความออกเป็นคำ และลบ Stopwords "
            "ภาษาไทยที่ไม่จำเป็นต่อการวิเคราะห์"
        )

        if tokens:

            token_df = pd.DataFrame({
                "ลำดับ": range(1, len(tokens) + 1),
                "Token": tokens
            })

            st.dataframe(
                token_df,
                use_container_width=True,
                hide_index=True
            )

        else:

            st.warning("ไม่พบ Token")


    # =====================================================
    # TAB 3
    # =====================================================

    with tab3:

        st.subheader("🏷️ Part-of-Speech & Named Entity")

        col_pos, col_ner = st.columns(2)

        # -------------------------
        # POS
        # -------------------------

        with col_pos:

            st.markdown("### 🔤 Part-of-Speech")

            st.markdown("**คำนาม (NOUN)**")

            if nouns:
                st.write(" • ".join(nouns))
            else:
                st.caption("ไม่พบ")

            st.markdown("**คำคุณศัพท์ (ADJ)**")

            if adjectives:
                st.write(" • ".join(adjectives))
            else:
                st.caption("ไม่พบ")

            st.markdown("**คำกริยา (VERB)**")

            if verbs:
                st.write(" • ".join(verbs))
            else:
                st.caption("ไม่พบ")

        # -------------------------
        # NER
        # -------------------------

        with col_ner:

            st.markdown("### 🔎 Named Entities")

            if entities:

                entity_df = pd.DataFrame(entities)

                st.dataframe(
                    entity_df,
                    use_container_width=True,
                    hide_index=True
                )

            else:

                st.info("ไม่พบ Named Entity")


    # =====================================================
    # TAB 4
    # =====================================================

    with tab4:

        st.subheader("💡 Topic Identification")

        if topics:

            st.markdown("### หัวข้อที่พบ")

            for topic in topics:

                st.markdown(
                    f"""
                    <span class="topic-badge">
                        {topic}
                    </span>
                    """,
                    unsafe_allow_html=True
                )

            st.divider()

            st.success(
                f"พบหัวข้อทั้งหมด {len(topics)} หัวข้อ"
            )

        else:

            st.warning(
                "ไม่สามารถระบุหัวข้อที่ชัดเจนจากข้อความนี้ได้"
            )


    # =====================================================
    # TAB 5
    # =====================================================

    with tab5:

        st.subheader("😊 Sentiment Analysis")

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "ผลการวิเคราะห์",
                sentiment
            )

        with col2:

            st.metric(
                "😊 Positive Keywords",
                positive_score
            )

        with col3:

            st.metric(
                "😞 Negative Keywords",
                negative_score
            )

        st.divider()

        if "เชิงบวก" in sentiment:

            st.success(
                "รีวิวนี้มีแนวโน้มเป็นความคิดเห็นเชิงบวก 👍"
            )

        elif "เชิงลบ" in sentiment:

            st.error(
                "รีวิวนี้มีแนวโน้มเป็นความคิดเห็นเชิงลบ 👎"
            )

        else:

            st.info(
                "รีวิวนี้มีแนวโน้มเป็นความคิดเห็นที่เป็นกลาง 😐"
            )


# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<div class="footer">

    🍽️ Restaurant Review NLP Dashboard  
    <br>
    Powered by Python • Streamlit • PyThaiNLP

</div>
""", unsafe_allow_html=True)