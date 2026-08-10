import re
import json
import html
import pandas as pd
import streamlit as st

from pythainlp.tokenize import word_tokenize
from pythainlp.corpus.common import thai_stopwords
from pythainlp.tag import pos_tag, NER


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Smart Restaurant Review Analyzer",
    page_icon="🍽️",
    layout="wide"
)


# =========================================================
# CSS
# =========================================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(180deg, #fff7ed 0%, #ffffff 45%, #f8fafc 100%);
}

.block-container {
    max-width: 1400px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Hero */
.hero {
    background: linear-gradient(135deg, #ea580c, #f97316, #fb923c);
    padding: 2.5rem;
    border-radius: 24px;
    color: white;
    margin-bottom: 1.5rem;
    box-shadow: 0 15px 40px rgba(234, 88, 12, 0.2);
}

.hero h1 {
    color: white;
    font-size: 2.5rem;
    font-weight: 800;
    margin: 0;
}

.hero p {
    color: rgba(255,255,255,0.92);
    font-size: 1.05rem;
    margin-top: 0.5rem;
}

/* Cards */
.info-card {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 18px;
    padding: 1.2rem;
    min-height: 125px;
    box-shadow: 0 5px 20px rgba(15,23,42,0.05);
}

.info-icon {
    font-size: 1.6rem;
}

.info-title {
    color: #64748b;
    font-size: 0.85rem;
    margin-top: 0.3rem;
}

.info-value {
    color: #0f172a;
    font-size: 1.05rem;
    font-weight: 700;
    margin-top: 0.3rem;
}

/* Badges */
.badge {
    display: inline-block;
    padding: 0.45rem 0.8rem;
    margin: 0.2rem;
    border-radius: 999px;
    background: #fff7ed;
    color: #c2410c;
    border: 1px solid #fed7aa;
    font-weight: 600;
}

.good-badge {
    display: inline-block;
    padding: 0.45rem 0.8rem;
    margin: 0.2rem;
    border-radius: 999px;
    background: #ecfdf5;
    color: #047857;
    border: 1px solid #a7f3d0;
    font-weight: 600;
}

.bad-badge {
    display: inline-block;
    padding: 0.45rem 0.8rem;
    margin: 0.2rem;
    border-radius: 999px;
    background: #fef2f2;
    color: #b91c1c;
    border: 1px solid #fecaca;
    font-weight: 600;
}

.footer {
    text-align: center;
    color: #94a3b8;
    margin-top: 3rem;
    padding: 1rem;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD NLP
# =========================================================

@st.cache_resource
def load_ner():
    try:
        return NER("thainer")
    except Exception:
        return None


@st.cache_data
def load_stopwords():
    return set(thai_stopwords())


ner_model = load_ner()
thai_stopwords_set = load_stopwords()


# =========================================================
# KEYWORDS
# =========================================================

POSITIVE_WORDS = [
    "อร่อย", "อร่อยมาก", "ดี", "ดีมาก", "เยี่ยม",
    "ยอดเยี่ยม", "ประทับใจ", "ชอบ", "คุ้ม", "สด",
    "หอม", "นุ่ม", "กรอบ", "บริการดี", "พนักงานดี",
    "บรรยากาศดี", "ถูกใจ",
    "delicious", "great", "good", "excellent",
    "amazing", "friendly", "fresh", "tasty",
    "love", "perfect", "worth"
]

NEGATIVE_WORDS = [
    "ไม่อร่อย", "แย่", "แย่มาก", "ผิดหวัง",
    "เค็ม", "หวานเกิน", "เผ็ดเกิน", "แพง",
    "ช้า", "รอนาน", "สกปรก", "เหม็น",
    "บริการแย่", "พนักงานแย่", "ไม่คุ้ม",
    "bad", "terrible", "awful", "disappointing",
    "expensive", "slow", "dirty", "salty",
    "bland", "worst"
]

FOOD_KEYWORDS = [
    "อาหาร", "เมนู", "รสชาติ", "อร่อย", "เค็ม",
    "หวาน", "เผ็ด", "ผัดไทย", "กะเพรา", "ต้มยำ",
    "ก๋วยเตี๋ยว", "ข้าว", "หมู", "ไก่", "ปลา",
    "ของหวาน", "กาแฟ", "เค้ก", "ไอศกรีม",
    "food", "menu", "taste", "delicious",
    "noodle", "rice", "chicken", "pork",
    "dessert", "coffee", "cake"
]

SERVICE_KEYWORDS = [
    "พนักงาน", "บริการ", "รอ", "ช้า", "เร็ว",
    "สุภาพ", "บริการดี", "บริการแย่",
    "staff", "service", "wait", "slow", "friendly"
]

PLACE_KEYWORDS = [
    "ร้าน", "บรรยากาศ", "ที่จอดรถ", "ทำเล",
    "สถานที่", "สยาม", "ห้าง", "ใกล้",
    "restaurant", "location", "parking",
    "atmosphere", "place"
]

PRICE_KEYWORDS = [
    "ราคา", "แพง", "ถูก", "คุ้ม", "บาท",
    "price", "expensive", "cheap", "worth"
]

MENU_KEYWORDS = [
    "ผัดไทย",
    "ผัดกะเพรา",
    "กะเพรา",
    "ต้มยำ",
    "ต้มยำกุ้ง",
    "ก๋วยเตี๋ยว",
    "ข้าวมันไก่",
    "ข้าวผัด",
    "หมูกรอบ",
    "ส้มตำ",
    "ไก่ทอด",
    "พิซซ่า",
    "เบอร์เกอร์",
    "สเต๊ก",
    "ชาไทย",
    "กาแฟ",
    "เค้ก",
    "ไอศกรีม",
    "pad thai",
    "tom yum",
    "fried rice",
    "burger",
    "pizza",
    "steak",
    "coffee",
    "cake",
    "ice cream"
]


# =========================================================
# CLEANING
# =========================================================

def clean_text(text):

    cleaned = text

    # Mask phone number
    cleaned = re.sub(
        r"0\d{1,2}[-\s]?\d{3}[-\s]?\d{4}",
        "[PHONE]",
        cleaned
    )

    # Mask URL
    cleaned = re.sub(
        r"https?://\S+|www\.\S+",
        "[URL]",
        cleaned
    )

    # Remove markdown link
    cleaned = re.sub(
        r"\[([^\]]+)\]\([^)]+\)",
        r"\1",
        cleaned
    )

    # Reduce repeated characters
    cleaned = re.sub(
        r"(.)\1{2,}",
        r"\1",
        cleaned
    )

    # Remove repeated spaces
    cleaned = re.sub(
        r"\s+",
        " ",
        cleaned
    ).strip()

    return cleaned


# =========================================================
# TOKENIZATION
# =========================================================

def tokenize_text(text):

    tokens = word_tokenize(
        text,
        engine="newmm"
    )

    english_stopwords = {
        "the", "a", "an", "is", "are", "was", "were",
        "and", "or", "of", "to", "in", "on", "at",
        "for", "this", "that", "very", "really", "it"
    }

    result = []

    for token in tokens:

        token = token.strip()

        if not token:
            continue

        if token in thai_stopwords_set:
            continue

        if token.lower() in english_stopwords:
            continue

        if re.fullmatch(r"[\W_]+", token):
            continue

        result.append(token)

    return result


# =========================================================
# POS TAGGING
# =========================================================

def analyze_pos(tokens):

    try:
        result = pos_tag(
            tokens,
            corpus="pud"
        )
    except Exception:
        result = []

    nouns = []
    adjectives = []
    verbs = []

    for item in result:

        if len(item) < 2:
            continue

        word = item[0]
        tag = item[1]

        if tag == "NOUN":
            nouns.append(word)

        elif tag == "ADJ":
            adjectives.append(word)

        elif tag == "VERB":
            verbs.append(word)

    return result, nouns, adjectives, verbs


# =========================================================
# NER
# =========================================================

def analyze_ner(text):

    entities = []

    if ner_model is None:
        return entities

    try:
        result = ner_model.tag(text)
    except Exception:
        return entities

    for item in result:

        if len(item) == 2:
            word, tag = item

        elif len(item) == 3:
            word, _, tag = item

        else:
            continue

        if tag != "O":
            entities.append({
                "คำ": word,
                "ประเภท": tag
            })

    return entities


# =========================================================
# EXTRACT RESTAURANT
# =========================================================

def extract_restaurant(text, entities):

    restaurants = []

    # จาก NER
    for entity in entities:

        tag = entity["ประเภท"].upper()

        if "ORG" in tag:
            restaurants.append(entity["คำ"])

    # จากคำว่า "ร้าน..."
    patterns = [
        r"ร้าน\s*([ก-๙A-Za-z0-9][^,.!?]{1,40})",
        r"ที่ร้าน\s*([ก-๙A-Za-z0-9][^,.!?]{1,40})"
    ]

    for pattern in patterns:

        matches = re.findall(
            pattern,
            text
        )

        for match in matches:

            match = re.split(
                r"\s+(?:แถว|อยู่|มี|บรรยากาศ|บริการ|สั่ง|ราคา|อร่อย|ดี)",
                match
            )[0].strip()

            if match:
                restaurants.append(match)

    return list(dict.fromkeys(restaurants))


# =========================================================
# EXTRACT LOCATION
# =========================================================

def extract_location(text, entities):

    locations = []

    for entity in entities:

        tag = entity["ประเภท"].upper()

        if "LOC" in tag or "LOCATION" in tag:
            locations.append(entity["คำ"])

    patterns = [
        r"อยู่ที่\s*([^,.!?]+)",
        r"แถว\s*([^,.!?]+)",
        r"บริเวณ\s*([^,.!?]+)",
        r"ใกล้\s*([^,.!?]+)"
    ]

    for pattern in patterns:

        matches = re.findall(
            pattern,
            text
        )

        for match in matches:

            value = match.strip()

            value = re.split(
                r"\s+(?:โทร|มี|และ|ร้าน|บรรยากาศ|พนักงาน)",
                value
            )[0].strip()

            if value:
                locations.append(value)

    return list(dict.fromkeys(locations))


# =========================================================
# EXTRACT MENU
# =========================================================

def extract_menu(text):

    found = []

    lower_text = text.lower()

    for menu in MENU_KEYWORDS:

        if menu.lower() in lower_text:
            found.append(menu)

    return list(dict.fromkeys(found))


# =========================================================
# SENTIMENT
# =========================================================

def analyze_sentiment(text):

    lower_text = text.lower()

    positive = [
        word for word in POSITIVE_WORDS
        if word.lower() in lower_text
    ]

    negative = [
        word for word in NEGATIVE_WORDS
        if word.lower() in lower_text
    ]

    positive_score = len(positive)
    negative_score = len(negative)

    if positive_score > negative_score:
        label = "😊 Positive"

    elif negative_score > positive_score:
        label = "😞 Negative"

    else:
        label = "😐 Neutral"

    return {
        "label": label,
        "positive_score": positive_score,
        "negative_score": negative_score,
        "positive_words": positive,
        "negative_words": negative
    }


# =========================================================
# TOPIC IDENTIFICATION
# =========================================================

def identify_topics(text):

    lower_text = text.lower()

    topics = []

    if any(
        keyword.lower() in lower_text
        for keyword in FOOD_KEYWORDS
    ):
        topics.append("🍲 อาหาร / รสชาติ")

    if any(
        keyword.lower() in lower_text
        for keyword in SERVICE_KEYWORDS
    ):
        topics.append("🤵 การบริการ")

    if any(
        keyword.lower() in lower_text
        for keyword in PLACE_KEYWORDS
    ):
        topics.append("🏪 สถานที่ / บรรยากาศ")

    if any(
        keyword.lower() in lower_text
        for keyword in PRICE_KEYWORDS
    ):
        topics.append("💰 ราคา / ความคุ้มค่า")

    return topics


# =========================================================
# COMPLETE ANALYSIS
# =========================================================

def analyze_review(raw_text):

    cleaned = clean_text(raw_text)

    tokens = tokenize_text(cleaned)

    pos_result, nouns, adjectives, verbs = analyze_pos(tokens)

    entities = analyze_ner(cleaned)

    restaurant = extract_restaurant(
        cleaned,
        entities
    )

    locations = extract_location(
        cleaned,
        entities
    )

    menus = extract_menu(cleaned)

    sentiment = analyze_sentiment(cleaned)

    topics = identify_topics(cleaned)

    return {
        "raw_text": raw_text,
        "clean_text": cleaned,
        "tokens": tokens,
        "pos": pos_result,
        "nouns": nouns,
        "adjectives": adjectives,
        "verbs": verbs,
        "entities": entities,
        "restaurant": restaurant,
        "locations": locations,
        "menus": menus,
        "sentiment": sentiment,
        "topics": topics
    }


# =========================================================
# HERO
# =========================================================

st.markdown("""
<div class="hero">

    <h1>🍽️ Smart Restaurant Review Analyzer</h1>

    <p>
        ระบบวิเคราะห์และสกัดข้อมูลจากรีวิวร้านอาหาร
        ด้วย Natural Language Processing (NLP)
        รองรับภาษาไทยและภาษาอังกฤษ
    </p>

</div>
""", unsafe_allow_html=True)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("🍽️ Smart Review NLP")

    st.markdown("""
### 🔎 NLP Pipeline

**1. Regex & Cleansing**  
🔐 Mask เบอร์โทร / URL

**2. Tokenization**  
✂️ ตัดคำ + Stopwords

**3. POS Tagging**  
🏷️ คำนาม / คำคุณศัพท์ / คำกริยา

**4. NER**  
📍 สถานที่ / องค์กร / บุคคล

**5. Information Extraction**  
🏪 ร้าน / 📍 สถานที่ / 🍜 เมนู

**6. Topic Identification**  
💡 อาหาร / บริการ / สถานที่ / ราคา

**7. Sentiment Analysis**  
😊 Positive / 😐 Neutral / 😞 Negative
""")

    st.divider()

    st.info(
        "โปรเจกต์นี้พัฒนาสำหรับการเรียนรู้ "
        "และสาธิตการประยุกต์ใช้ NLP"
    )


# =========================================================
# INPUT
# =========================================================

st.subheader("📝 ข้อความรีวิว")

demo_options = {
    "🇹🇭 รีวิวภาษาไทย": (
        "ร้านครัวบ้านสวนอร่อยมากครับ "
        "บรรยากาศดี แอร์เย็น พนักงานบริการดีเยี่ยม "
        "ผมสั่งผัดไทยกับต้มยำกุ้ง "
        "รสชาติอร่อยมาก แต่ต้มยำค่อนข้างเค็มนิดหน่อย "
        "ร้านอยู่ที่สยามพารากอน "
        "โทร 081-234-5678 "
        "ดูรีวิวเพิ่มเติมที่ https://example.com"
    ),
    "🇬🇧 English Review": (
        "The food at ABC Restaurant was delicious. "
        "The staff were friendly and the atmosphere was great. "
        "I ordered pad thai and fried rice. "
        "The price was a little expensive but worth it. "
        "The restaurant is near Siam."
    ),
    "✏️ เขียนข้อความเอง": ""
}

selected_demo = st.selectbox(
    "เลือกตัวอย่าง",
    list(demo_options.keys())
)

raw_text = st.text_area(
    "พิมพ์ข้อความรีวิว",
    value=demo_options[selected_demo],
    height=180
)

analyze_button = st.button(
    "🚀 วิเคราะห์รีวิว",
    type="primary",
    use_container_width=True
)


# =========================================================
# ANALYSIS
# =========================================================

if analyze_button:

    if not raw_text.strip():

        st.warning(
            "⚠️ กรุณากรอกข้อความรีวิวก่อน"
        )

        st.stop()

    with st.spinner(
        "🔄 กำลังวิเคราะห์ข้อความ..."
    ):

        result = analyze_review(raw_text)

    st.success(
        "✨ วิเคราะห์ข้อความเรียบร้อยแล้ว"
    )

    # =====================================================
    # EXTRACTION
    # =====================================================

    st.subheader("📌 ข้อมูลสำคัญที่สกัดได้")

    restaurant_text = (
        ", ".join(result["restaurant"])
        if result["restaurant"]
        else "ไม่พบ"
    )

    location_text = (
        ", ".join(result["locations"])
        if result["locations"]
        else "ไม่พบ"
    )

    menu_text = (
        ", ".join(result["menus"])
        if result["menus"]
        else "ไม่พบ"
    )

    sentiment_text = result["sentiment"]["label"]

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.markdown(
            f"""
            <div class="info-card">
                <div class="info-icon">🏪</div>
                <div class="info-title">ชื่อร้าน / แบรนด์</div>
                <div class="info-value">
                    {html.escape(restaurant_text)}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            f"""
            <div class="info-card">
                <div class="info-icon">📍</div>
                <div class="info-title">สถานที่</div>
                <div class="info-value">
                    {html.escape(location_text)}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            f"""
            <div class="info-card">
                <div class="info-icon">🍜</div>
                <div class="info-title">เมนู</div>
                <div class="info-value">
                    {html.escape(menu_text)}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c4:
        st.markdown(
            f"""
            <div class="info-card">
                <div class="info-icon">😊</div>
                <div class="info-title">Sentiment</div>
                <div class="info-value">
                    {html.escape(sentiment_text)}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c5:
        st.markdown(
            f"""
            <div class="info-card">
                <div class="info-icon">💡</div>
                <div class="info-title">Topics</div>
                <div class="info-value">
                    {len(result["topics"])} หัวข้อ
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.divider()

    # =====================================================
    # PRAISE / CRITICISM
    # =====================================================

    good_col, bad_col = st.columns(2)

    with good_col:

        st.subheader("👍 คำชม")

        words = result["sentiment"]["positive_words"]

        if words:

            for word in words:

                st.markdown(
                    f"""
                    <span class="good-badge">
                        👍 {html.escape(word)}
                    </span>
                    """,
                    unsafe_allow_html=True
                )

        else:
            st.caption("ไม่พบคำชม")

    with bad_col:

        st.subheader("👎 คำติ")

        words = result["sentiment"]["negative_words"]

        if words:

            for word in words:

                st.markdown(
                    f"""
                    <span class="bad-badge">
                        👎 {html.escape(word)}
                    </span>
                    """,
                    unsafe_allow_html=True
                )

        else:
            st.caption("ไม่พบคำติ")

    st.divider()

    # =====================================================
    # METRICS
    # =====================================================

    st.subheader("📊 NLP Summary")

    m1, m2, m3, m4, m5 = st.columns(5)

    m1.metric(
        "🔤 Tokens",
        len(result["tokens"])
    )

    m2.metric(
        "🏷️ NER",
        len(result["entities"])
    )

    m3.metric(
        "📚 Nouns",
        len(result["nouns"])
    )

    m4.metric(
        "✨ Adjectives",
        len(result["adjectives"])
    )

    m5.metric(
        "💡 Topics",
        len(result["topics"])
    )

    st.divider()

    # =====================================================
    # TABS
    # =====================================================

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🧹 Cleaning",
        "✂️ Tokenization",
        "🏷️ POS & NER",
        "💡 Topic",
        "😊 Sentiment",
        "📦 JSON"
    ])

    # =====================================================
    # CLEANING
    # =====================================================

    with tab1:

        st.subheader(
            "🧹 Regex & Text Cleansing"
        )

        col_a, col_b = st.columns(2)

        with col_a:

            st.markdown("#### 📄 ข้อความต้นฉบับ")

            st.info(
                result["raw_text"]
            )

        with col_b:

            st.markdown("#### ✨ ข้อความหลัง Clean")

            st.success(
                result["clean_text"]
            )

        st.markdown("""
### เทคนิคที่ใช้

- 📱 Mask เบอร์โทรศัพท์
- 🔗 Mask URL
- 🔤 ลดคำลากเสียง เช่น `อร่อยยยยย`
- 🧹 ลดช่องว่างซ้ำ
        """)

    # =====================================================
    # TOKENIZATION
    # =====================================================

    with tab2:

        st.subheader(
            "✂️ Tokenization & Normalization"
        )

        token_df = pd.DataFrame({
            "ลำดับ": range(
                1,
                len(result["tokens"]) + 1
            ),
            "Token": result["tokens"]
        })

        st.dataframe(
            token_df,
            use_container_width=True,
            hide_index=True
        )

    # =====================================================
    # POS & NER
    # =====================================================

    with tab3:

        left, right = st.columns(2)

        with left:

            st.subheader(
                "🏷️ Part-of-Speech"
            )

            st.markdown("**คำนาม (NOUN)**")

            st.write(
                ", ".join(result["nouns"])
                if result["nouns"]
                else "ไม่พบ"
            )

            st.markdown("**คำคุณศัพท์ (ADJ)**")

            st.write(
                ", ".join(result["adjectives"])
                if result["adjectives"]
                else "ไม่พบ"
            )

            st.markdown("**คำกริยา (VERB)**")

            st.write(
                ", ".join(result["verbs"])
                if result["verbs"]
                else "ไม่พบ"
            )

        with right:

            st.subheader(
                "🔎 Named Entity Recognition"
            )

            if result["entities"]:

                entity_df = pd.DataFrame(
                    result["entities"]
                )

                st.dataframe(
                    entity_df,
                    use_container_width=True,
                    hide_index=True
                )

            else:

                st.info(
                    "ไม่พบ Named Entity"
                )

    # =====================================================
    # TOPIC
    # =====================================================

    with tab4:

        st.subheader(
            "💡 Topic Identification"
        )

        if result["topics"]:

            for topic in result["topics"]:

                st.markdown(
                    f"""
                    <span class="badge">
                        {html.escape(topic)}
                    </span>
                    """,
                    unsafe_allow_html=True
                )

        else:

            st.warning(
                "ไม่สามารถระบุ Topic ได้"
            )

        st.markdown("### 📖 Topic ที่ระบบรองรับ")

        topic_df = pd.DataFrame({
            "หัวข้อ": [
                "🍲 อาหาร / รสชาติ",
                "🤵 การบริการ",
                "🏪 สถานที่ / บรรยากาศ",
                "💰 ราคา / ความคุ้มค่า"
            ],
            "ตัวอย่างคำ": [
                "อร่อย, เมนู, รสชาติ",
                "พนักงาน, บริการ, รอ",
                "ร้าน, ทำเล, บรรยากาศ",
                "ราคา, แพง, คุ้ม"
            ]
        })

        st.dataframe(
            topic_df,
            use_container_width=True,
            hide_index=True
        )

    # =====================================================
    # SENTIMENT
    # =====================================================

    with tab5:

        st.subheader(
            "😊 Sentiment Analysis"
        )

        s1, s2, s3 = st.columns(3)

        s1.metric(
            "ผลลัพธ์",
            result["sentiment"]["label"]
        )

        s2.metric(
            "👍 Positive",
            result["sentiment"]["positive_score"]
        )

        s3.metric(
            "👎 Negative",
            result["sentiment"]["negative_score"]
        )

        st.divider()

        if (
            result["sentiment"]["positive_score"]
            >
            result["sentiment"]["negative_score"]
        ):

            st.success(
                "😊 รีวิวนี้มีแนวโน้มเป็นความคิดเห็นเชิงบวก"
            )

        elif (
            result["sentiment"]["negative_score"]
            >
            result["sentiment"]["positive_score"]
        ):

            st.error(
                "😞 รีวิวนี้มีแนวโน้มเป็นความคิดเห็นเชิงลบ"
            )

        else:

            st.info(
                "😐 รีวิวนี้มีแนวโน้มเป็นความคิดเห็นที่เป็นกลาง"
            )

    # =====================================================
    # JSON
    # =====================================================

    with tab6:

        st.subheader(
            "📦 Structured Output"
        )

        export_data = {
            "restaurant": result["restaurant"],
            "locations": result["locations"],
            "menus": result["menus"],
            "positive_words": result["sentiment"]["positive_words"],
            "negative_words": result["sentiment"]["negative_words"],
            "sentiment": result["sentiment"]["label"],
            "topics": result["topics"],
            "tokens": result["tokens"],
            "entities": result["entities"]
        }

        st.json(export_data)

        json_data = json.dumps(
            export_data,
            ensure_ascii=False,
            indent=2
        )

        st.download_button(
            "⬇️ ดาวน์โหลดผลลัพธ์ JSON",
            data=json_data,
            file_name="restaurant_review_result.json",
            mime="application/json",
            use_container_width=True
        )


# =========================================================
# TEST DATA
# =========================================================

st.divider()

st.subheader("🧪 Test Dataset")

st.caption(
    "ตัวอย่างข้อมูลทดสอบที่อยู่ในไฟล์ test_data.txt"
)

try:

    with open(
        "test_data.txt",
        "r",
        encoding="utf-8"
    ) as file:

        test_lines = [
            line.strip()
            for line in file.readlines()
            if line.strip()
        ]

    st.write(
        f"📄 พบข้อมูลทดสอบทั้งหมด {len(test_lines)} รายการ"
    )

    with st.expander("ดูข้อมูลทดสอบ"):

        for i, line in enumerate(
            test_lines,
            start=1
        ):

            st.markdown(
                f"**{i}.** {line}"
            )

except FileNotFoundError:

    st.warning(
        "ไม่พบไฟล์ test_data.txt"
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<div class="footer">

🍽️ <b>Smart Restaurant Review Analyzer</b>
<br>
Natural Language Processing • Python • Streamlit • PyThaiNLP

</div>
""", unsafe_allow_html=True)