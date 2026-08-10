import re
import json
import html
import pandas as pd
import streamlit as st

from pythainlp.tokenize import word_tokenize
from pythainlp.corpus.common import thai_stopwords
from pythainlp.tag import pos_tag, NER


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="DineSense AI",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# MODERN UI
# ============================================================

st.markdown("""
<style>

/* ---------- GLOBAL ---------- */

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(
            circle at 90% 0%,
            rgba(251, 146, 60, 0.10),
            transparent 28%
        ),
        radial-gradient(
            circle at 0% 20%,
            rgba(249, 115, 22, 0.06),
            transparent 25%
        ),
        #f8fafc;
}

.block-container {
    max-width: 1380px;
    padding-top: 1.5rem;
    padding-bottom: 4rem;
}


/* ---------- HIDE DEFAULT ---------- */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    background: transparent !important;
}


/* ---------- NAV ---------- */

.navbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.6rem 0 1.5rem 0;
}

.brand {
    display: flex;
    align-items: center;
    gap: 10px;
}

.brand-icon {
    width: 42px;
    height: 42px;
    border-radius: 13px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: linear-gradient(
        135deg,
        #f97316,
        #ea580c
    );
    box-shadow: 0 8px 20px rgba(234,88,12,.22);
    font-size: 21px;
}

.brand-name {
    font-weight: 800;
    font-size: 18px;
    color: #0f172a;
}

.brand-sub {
    font-size: 11px;
    color: #94a3b8;
}

.status {
    display: flex;
    align-items: center;
    gap: 7px;
    background: #ffffff;
    border: 1px solid #e2e8f0;
    padding: 7px 12px;
    border-radius: 999px;
    color: #475569;
    font-size: 12px;
    font-weight: 600;
}

.status-dot {
    width: 7px;
    height: 7px;
    background: #22c55e;
    border-radius: 50%;
    box-shadow: 0 0 0 4px #dcfce7;
}


/* ---------- HERO ---------- */

.hero {
    position: relative;
    overflow: hidden;
    border-radius: 28px;
    padding: 42px 45px;
    background:
        radial-gradient(
            circle at 85% 20%,
            rgba(255,255,255,.20),
            transparent 24%
        ),
        linear-gradient(
            135deg,
            #c2410c 0%,
            #ea580c 45%,
            #f97316 100%
        );
    box-shadow:
        0 25px 60px rgba(194,65,12,.18);
    margin-bottom: 25px;
}

.hero:after {
    content: "";
    position: absolute;
    width: 250px;
    height: 250px;
    right: -90px;
    bottom: -120px;
    border-radius: 50%;
    border: 45px solid rgba(255,255,255,.07);
}

.hero-kicker {
    display: inline-block;
    padding: 6px 11px;
    border-radius: 999px;
    background: rgba(255,255,255,.15);
    border: 1px solid rgba(255,255,255,.20);
    color: white;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: .4px;
    margin-bottom: 14px;
}

.hero h1 {
    color: white;
    font-size: 40px;
    line-height: 1.1;
    font-weight: 800;
    margin: 0;
    letter-spacing: -1.2px;
}

.hero p {
    color: rgba(255,255,255,.88);
    font-size: 15px;
    max-width: 680px;
    line-height: 1.7;
    margin-top: 14px;
    margin-bottom: 0;
}


/* ---------- SECTION ---------- */

.section-title {
    font-size: 20px;
    font-weight: 800;
    color: #0f172a;
    margin-top: 28px;
    margin-bottom: 5px;
}

.section-sub {
    font-size: 13px;
    color: #64748b;
    margin-bottom: 15px;
}


/* ---------- INPUT CARD ---------- */

.input-card {
    background: rgba(255,255,255,.88);
    border: 1px solid #e2e8f0;
    border-radius: 22px;
    padding: 22px;
    box-shadow: 0 12px 35px rgba(15,23,42,.05);
}


/* ---------- METRIC ---------- */

.metric-card {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 18px;
    padding: 19px;
    min-height: 105px;
    box-shadow: 0 8px 25px rgba(15,23,42,.045);
    transition: all .2s ease;
}

.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 14px 35px rgba(15,23,42,.08);
}

.metric-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.metric-icon {
    width: 35px;
    height: 35px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 11px;
    background: #fff7ed;
    font-size: 17px;
}

.metric-label {
    color: #64748b;
    font-size: 11px;
    font-weight: 600;
    margin-top: 13px;
}

.metric-value {
    color: #0f172a;
    font-size: 15px;
    font-weight: 800;
    margin-top: 4px;
}


/* ---------- RESULT CARD ---------- */

.result-card {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 20px;
    padding: 21px;
    min-height: 155px;
    box-shadow: 0 8px 28px rgba(15,23,42,.045);
}

.result-icon {
    font-size: 22px;
}

.result-title {
    color: #64748b;
    font-size: 12px;
    margin-top: 11px;
}

.result-value {
    color: #0f172a;
    font-size: 18px;
    font-weight: 800;
    line-height: 1.4;
    margin-top: 5px;
}


/* ---------- TAG ---------- */

.tag {
    display: inline-block;
    padding: 7px 11px;
    border-radius: 999px;
    margin: 3px;
    background: #fff7ed;
    color: #c2410c;
    border: 1px solid #fed7aa;
    font-size: 12px;
    font-weight: 700;
}

.tag-green {
    display: inline-block;
    padding: 7px 11px;
    border-radius: 999px;
    margin: 3px;
    background: #ecfdf5;
    color: #047857;
    border: 1px solid #a7f3d0;
    font-size: 12px;
    font-weight: 700;
}

.tag-red {
    display: inline-block;
    padding: 7px 11px;
    border-radius: 999px;
    margin: 3px;
    background: #fef2f2;
    color: #b91c1c;
    border: 1px solid #fecaca;
    font-size: 12px;
    font-weight: 700;
}


/* ---------- PROCESS ---------- */

.pipeline {
    display: flex;
    align-items: center;
    gap: 8px;
    margin: 18px 0 25px;
    overflow-x: auto;
}

.pipeline-item {
    white-space: nowrap;
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 9px 12px;
    font-size: 11px;
    font-weight: 700;
    color: #475569;
}

.pipeline-arrow {
    color: #cbd5e1;
}


/* ---------- FOOTER ---------- */

.modern-footer {
    margin-top: 60px;
    padding-top: 25px;
    border-top: 1px solid #e2e8f0;
    text-align: center;
    color: #94a3b8;
    font-size: 11px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# NLP MODEL
# ============================================================

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
stopwords = load_stopwords()


# ============================================================
# KEYWORDS
# ============================================================

POSITIVE = [
    "อร่อย", "ดี", "ดีมาก", "เยี่ยม",
    "ยอดเยี่ยม", "ประทับใจ", "ชอบ",
    "คุ้ม", "สด", "หอม", "นุ่ม",
    "กรอบ", "บริการดี", "บรรยากาศดี",
    "delicious", "great", "good",
    "excellent", "amazing", "friendly",
    "fresh", "tasty", "love", "perfect",
    "worth"
]

NEGATIVE = [
    "ไม่อร่อย", "แย่", "แย่มาก",
    "ผิดหวัง", "เค็ม", "หวานเกิน",
    "เผ็ดเกิน", "แพง", "ช้า",
    "รอนาน", "สกปรก", "เหม็น",
    "บริการแย่", "ไม่คุ้ม",
    "bad", "terrible", "awful",
    "disappointing", "expensive",
    "slow", "dirty", "worst"
]

FOOD = [
    "อาหาร", "เมนู", "รสชาติ", "อร่อย",
    "เค็ม", "หวาน", "เผ็ด",
    "ผัดไทย", "กะเพรา", "ต้มยำ",
    "ก๋วยเตี๋ยว", "ข้าว", "หมู", "ไก่",
    "ปลา", "ของหวาน", "กาแฟ",
    "เค้ก", "ไอศกรีม",
    "food", "menu", "taste",
    "delicious", "noodle", "rice",
    "chicken", "pork", "dessert",
    "coffee", "cake"
]

SERVICE = [
    "พนักงาน", "บริการ", "รอ",
    "ช้า", "เร็ว", "สุภาพ",
    "บริการดี", "บริการแย่",
    "staff", "service",
    "wait", "slow", "friendly"
]

PLACE = [
    "ร้าน", "บรรยากาศ",
    "ที่จอดรถ", "ทำเล",
    "สถานที่", "สยาม",
    "ห้าง", "ใกล้",
    "restaurant", "location",
    "parking", "atmosphere",
    "place"
]

PRICE = [
    "ราคา", "แพง", "ถูก",
    "คุ้ม", "บาท",
    "price", "expensive",
    "cheap", "worth"
]

MENUS = [
    "ผัดไทย", "ผัดกะเพรา",
    "กะเพรา", "ต้มยำ",
    "ต้มยำกุ้ง", "ก๋วยเตี๋ยว",
    "ข้าวมันไก่", "ข้าวผัด",
    "หมูกรอบ", "ส้มตำ",
    "ไก่ทอด", "พิซซ่า",
    "เบอร์เกอร์", "สเต๊ก",
    "ชาไทย", "กาแฟ",
    "เค้ก", "ไอศกรีม",
    "pad thai", "tom yum",
    "fried rice", "burger",
    "pizza", "steak",
    "coffee", "cake"
]


# ============================================================
# FUNCTIONS
# ============================================================

def clean_text(text):

    text = re.sub(
        r"0\d{1,2}[-\s]?\d{3}[-\s]?\d{4}",
        "[PHONE]",
        text
    )

    text = re.sub(
        r"https?://\S+|www\.\S+",
        "[URL]",
        text
    )

    text = re.sub(
        r"\[([^\]]+)\]\([^)]+\)",
        r"\1",
        text
    )

    text = re.sub(
        r"(.)\1{2,}",
        r"\1",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


def tokenize_text(text):

    tokens = word_tokenize(
        text,
        engine="newmm"
    )

    english_stopwords = {
        "the", "a", "an", "is", "are",
        "was", "were", "and", "or",
        "of", "to", "in", "on",
        "at", "for", "this",
        "that", "very", "really"
    }

    result = []

    for token in tokens:

        token = token.strip()

        if not token:
            continue

        if token in stopwords:
            continue

        if token.lower() in english_stopwords:
            continue

        if re.fullmatch(r"[\W_]+", token):
            continue

        result.append(token)

    return result


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


def extract_restaurant(text, entities):

    found = []

    for entity in entities:

        if "ORG" in entity["ประเภท"].upper():
            found.append(entity["คำ"])

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

            value = re.split(
                r"\s+(?:แถว|อยู่|มี|บรรยากาศ|บริการ|สั่ง|ราคา|อร่อย|ดี)",
                match
            )[0].strip()

            if value:
                found.append(value)

    return list(dict.fromkeys(found))


def extract_location(text, entities):

    found = []

    for entity in entities:

        tag = entity["ประเภท"].upper()

        if "LOC" in tag or "LOCATION" in tag:
            found.append(entity["คำ"])

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

            value = re.split(
                r"\s+(?:โทร|มี|และ|ร้าน|บรรยากาศ|พนักงาน)",
                match.strip()
            )[0].strip()

            if value:
                found.append(value)

    return list(dict.fromkeys(found))


def extract_menu(text):

    result = []

    for menu in MENUS:

        if menu.lower() in text.lower():
            result.append(menu)

    return list(dict.fromkeys(result))


def sentiment(text):

    lower = text.lower()

    positive = [
        x for x in POSITIVE
        if x.lower() in lower
    ]

    negative = [
        x for x in NEGATIVE
        if x.lower() in lower
    ]

    if len(positive) > len(negative):
        label = "😊 Positive"

    elif len(negative) > len(positive):
        label = "😞 Negative"

    else:
        label = "😐 Neutral"

    return {
        "label": label,
        "positive": positive,
        "negative": negative
    }


def topics(text):

    lower = text.lower()

    result = []

    if any(x.lower() in lower for x in FOOD):
        result.append("🍲 อาหาร")

    if any(x.lower() in lower for x in SERVICE):
        result.append("🤝 บริการ")

    if any(x.lower() in lower for x in PLACE):
        result.append("📍 สถานที่")

    if any(x.lower() in lower for x in PRICE):
        result.append("💰 ราคา")

    return result


def analyze(text):

    cleaned = clean_text(text)

    tokens = tokenize_text(cleaned)

    pos_result, nouns, adjectives, verbs = analyze_pos(
        tokens
    )

    entities = analyze_ner(cleaned)

    return {
        "raw": text,
        "cleaned": cleaned,
        "tokens": tokens,
        "nouns": nouns,
        "adjectives": adjectives,
        "verbs": verbs,
        "entities": entities,
        "restaurant": extract_restaurant(
            cleaned,
            entities
        ),
        "location": extract_location(
            cleaned,
            entities
        ),
        "menus": extract_menu(cleaned),
        "sentiment": sentiment(cleaned),
        "topics": topics(cleaned)
    }


# ============================================================
# NAVIGATION
# ============================================================

st.markdown("""
<div class="navbar">

    <div class="brand">

        <div class="brand-icon">
            🍽️
        </div>

        <div>
            <div class="brand-name">
                DineSense AI
            </div>

            <div class="brand-sub">
                Restaurant Review Intelligence
            </div>
        </div>

    </div>

    <div class="status">
        <span class="status-dot"></span>
        NLP Engine Online
    </div>

</div>
""", unsafe_allow_html=True)


# ============================================================
# HERO
# ============================================================

st.markdown("""
<div class="hero">

    <div class="hero-kicker">
        ✦ AI-POWERED NLP ANALYTICS
    </div>

    <h1>
        Understand every<br>
        restaurant review.
    </h1>

    <p>
        วิเคราะห์รีวิวร้านอาหารด้วย Natural Language Processing
        พร้อมสกัดชื่อร้าน สถานที่ เมนู คำชม คำติ
        Topic และ Sentiment จากข้อความภาษาไทยและภาษาอังกฤษ
    </p>

</div>
""", unsafe_allow_html=True)


# ============================================================
# PIPELINE
# ============================================================

st.markdown("""
<div class="pipeline">

    <div class="pipeline-item">01 · Clean</div>
    <div class="pipeline-arrow">→</div>

    <div class="pipeline-item">02 · Tokenize</div>
    <div class="pipeline-arrow">→</div>

    <div class="pipeline-item">03 · POS / NER</div>
    <div class="pipeline-arrow">→</div>

    <div class="pipeline-item">04 · Extract</div>
    <div class="pipeline-arrow">→</div>

    <div class="pipeline-item">05 · Topic</div>
    <div class="pipeline-arrow">→</div>

    <div class="pipeline-item">06 · Sentiment</div>

</div>
""", unsafe_allow_html=True)


# ============================================================
# INPUT
# ============================================================

st.markdown(
    '<div class="section-title">Analyze a review</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-sub">วางข้อความรีวิวเพื่อเริ่มการวิเคราะห์</div>',
    unsafe_allow_html=True
)

demo = {
    "🇹🇭 Thai example":
        "ร้านครัวบ้านสวนอร่อยมากครับ "
        "บรรยากาศดี พนักงานบริการดีเยี่ยม "
        "ผมสั่งผัดไทยกับต้มยำกุ้ง "
        "รสชาติอร่อยมาก แต่ต้มยำค่อนข้างเค็ม "
        "ร้านอยู่ที่สยามพารากอน "
        "โทร 081-234-5678 "
        "https://example.com",

    "🇬🇧 English example":
        "The food at ABC Restaurant was delicious. "
        "The staff were friendly and the atmosphere was great. "
        "I ordered pad thai and fried rice. "
        "The price was expensive but worth it.",

    "✍️ Write my own review": ""
}

selected = st.selectbox(
    "Example",
    list(demo.keys()),
    label_visibility="collapsed"
)

review = st.text_area(
    "Review",
    value=demo[selected],
    height=170,
    placeholder="เช่น ร้านนี้อาหารอร่อยมาก พนักงานบริการดี...",
    label_visibility="collapsed"
)

analyze_button = st.button(
    "✨ Analyze Review",
    type="primary",
    use_container_width=True
)


# ============================================================
# RESULT
# ============================================================

if analyze_button:

    if not review.strip():

        st.warning(
            "กรุณากรอกข้อความรีวิวก่อนเริ่มวิเคราะห์"
        )

        st.stop()

    with st.spinner("Analyzing your review..."):

        result = analyze(review)

    st.success(
        "Analysis completed successfully"
    )

    # --------------------------------------------------------
    # KEY RESULTS
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Key insights</div>',
        unsafe_allow_html=True
    )

    restaurant = (
        ", ".join(result["restaurant"])
        if result["restaurant"]
        else "ไม่พบ"
    )

    location = (
        ", ".join(result["location"])
        if result["location"]
        else "ไม่พบ"
    )

    menus = (
        ", ".join(result["menus"])
        if result["menus"]
        else "ไม่พบ"
    )

    sentiment_label = result["sentiment"]["label"]

    cards = [
        ("🏪", "Restaurant", restaurant),
        ("📍", "Location", location),
        ("🍜", "Menu", menus),
        ("💬", "Sentiment", sentiment_label),
    ]

    cols = st.columns(4)

    for col, card in zip(cols, cards):

        icon, title, value = card

        with col:

            st.markdown(
                f"""
                <div class="result-card">

                    <div class="result-icon">
                        {icon}
                    </div>

                    <div class="result-title">
                        {title}
                    </div>

                    <div class="result-value">
                        {html.escape(value)}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">NLP metrics</div>',
        unsafe_allow_html=True
    )

    metric_data = [
        ("🔤", "Tokens", len(result["tokens"])),
        ("🏷️", "Entities", len(result["entities"])),
        ("📚", "Nouns", len(result["nouns"])),
        ("✨", "Adjectives", len(result["adjectives"])),
        ("💡", "Topics", len(result["topics"])),
    ]

    cols = st.columns(5)

    for col, data in zip(cols, metric_data):

        icon, label, value = data

        with col:

            st.markdown(
                f"""
                <div class="metric-card">

                    <div class="metric-top">

                        <div class="metric-icon">
                            {icon}
                        </div>

                    </div>

                    <div class="metric-label">
                        {label}
                    </div>

                    <div class="metric-value">
                        {value}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

    st.divider()

    # --------------------------------------------------------
    # TABS
    # --------------------------------------------------------

    tabs = st.tabs([
        "✨ Insights",
        "🧹 Cleaning",
        "✂️ Tokens",
        "🏷️ POS & NER",
        "📦 JSON"
    ])

    # ========================================================
    # INSIGHTS
    # ========================================================

    with tabs[0]:

        left, right = st.columns(
            [1.2, 1]
        )

        with left:

            st.markdown(
                '<div class="section-title">💡 Topics detected</div>',
                unsafe_allow_html=True
            )

            if result["topics"]:

                for topic in result["topics"]:

                    st.markdown(
                        f"""
                        <span class="tag">
                            {html.escape(topic)}
                        </span>
                        """,
                        unsafe_allow_html=True
                    )

            else:

                st.caption(
                    "ไม่พบหัวข้อที่ชัดเจน"
                )

        with right:

            st.markdown(
                '<div class="section-title">😊 Sentiment</div>',
                unsafe_allow_html=True
            )

            if (
                len(result["sentiment"]["positive"])
                >
                len(result["sentiment"]["negative"])
            ):

                st.success(
                    "ความคิดเห็นมีแนวโน้มเป็น Positive"
                )

            elif (
                len(result["sentiment"]["negative"])
                >
                len(result["sentiment"]["positive"])
            ):

                st.error(
                    "ความคิดเห็นมีแนวโน้มเป็น Negative"
                )

            else:

                st.info(
                    "ความคิดเห็นมีแนวโน้มเป็น Neutral"
                )

        st.markdown(
            '<div class="section-title">👍 Positive words</div>',
            unsafe_allow_html=True
        )

        if result["sentiment"]["positive"]:

            for word in result["sentiment"]["positive"]:

                st.markdown(
                    f"""
                    <span class="tag-green">
                        {html.escape(word)}
                    </span>
                    """,
                    unsafe_allow_html=True
                )

        else:

            st.caption("ไม่พบ")

        st.markdown(
            '<div class="section-title">👎 Negative words</div>',
            unsafe_allow_html=True
        )

        if result["sentiment"]["negative"]:

            for word in result["sentiment"]["negative"]:

                st.markdown(
                    f"""
                    <span class="tag-red">
                        {html.escape(word)}
                    </span>
                    """,
                    unsafe_allow_html=True
                )

        else:

            st.caption("ไม่พบ")

    # ========================================================
    # CLEANING
    # ========================================================

    with tabs[1]:

        st.markdown(
            '<div class="section-title">🧹 Regex & Cleansing</div>',
            unsafe_allow_html=True
        )

        c1, c2 = st.columns(2)

        with c1:

            st.markdown("**Original text**")

            st.info(
                result["raw"]
            )

        with c2:

            st.markdown("**Cleaned text**")

            st.success(
                result["cleaned"]
            )

        st.markdown("""
**Processing applied**

- 📱 Phone number masking
- 🔗 URL masking
- ✨ Repeated-character normalization
- 🧹 Whitespace normalization
        """)

    # ========================================================
    # TOKENS
    # ========================================================

    with tabs[2]:

        st.markdown(
            '<div class="section-title">✂️ Tokenization</div>',
            unsafe_allow_html=True
        )

        token_df = pd.DataFrame({
            "Index":
                range(
                    1,
                    len(result["tokens"]) + 1
                ),
            "Token":
                result["tokens"]
        })

        st.dataframe(
            token_df,
            use_container_width=True,
            hide_index=True
        )

    # ========================================================
    # POS / NER
    # ========================================================

    with tabs[3]:

        c1, c2 = st.columns(2)

        with c1:

            st.markdown(
                '<div class="section-title">🏷️ POS Tagging</div>',
                unsafe_allow_html=True
            )

            st.markdown("**NOUN**")

            st.write(
                ", ".join(result["nouns"])
                if result["nouns"]
                else "ไม่พบ"
            )

            st.markdown("**ADJ**")

            st.write(
                ", ".join(result["adjectives"])
                if result["adjectives"]
                else "ไม่พบ"
            )

            st.markdown("**VERB**")

            st.write(
                ", ".join(result["verbs"])
                if result["verbs"]
                else "ไม่พบ"
            )

        with c2:

            st.markdown(
                '<div class="section-title">🔎 Named Entities</div>',
                unsafe_allow_html=True
            )

            if result["entities"]:

                df = pd.DataFrame(
                    result["entities"]
                )

                st.dataframe(
                    df,
                    use_container_width=True,
                    hide_index=True
                )

            else:

                st.info(
                    "ไม่พบ Named Entity"
                )

    # ========================================================
    # JSON
    # ========================================================

    with tabs[4]:

        export = {
            "restaurant":
                result["restaurant"],

            "location":
                result["location"],

            "menus":
                result["menus"],

            "positive_words":
                result["sentiment"]["positive"],

            "negative_words":
                result["sentiment"]["negative"],

            "sentiment":
                result["sentiment"]["label"],

            "topics":
                result["topics"],

            "tokens":
                result["tokens"],

            "entities":
                result["entities"]
        }

        st.json(export)

        json_data = json.dumps(
            export,
            ensure_ascii=False,
            indent=2
        )

        st.download_button(
            "⬇️ Download JSON",
            data=json_data,
            file_name="restaurant_analysis.json",
            mime="application/json",
            use_container_width=True
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="modern-footer">

    <b>DineSense AI</b>
    · Restaurant Review Intelligence

    <br><br>

    Built with Python · Streamlit · PyThaiNLP · NLP

</div>
""", unsafe_allow_html=True)