from pathlib import Path

import pandas as pd
import streamlit as st
import tensorflow as tf


st.set_page_config(
    page_title="ReviewSense | Sentiment Intelligence",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    .stApp { background: radial-gradient(ellipse at 15% 0%, #20294b 0, #11162a 42%, #0b1020 100%); color: #edf1ff; }
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; color: #f4f6ff !important; }
    p, label, .stMarkdown, [data-testid="stMetricLabel"] { color: #c4cbe0; }
    [data-testid="stSidebar"] { background: #0a0f1f; border-right: 1px solid #252d49; }
    [data-testid="stMetric"] { background: linear-gradient(145deg,#1c2541,#151c33); border: 1px solid #313c5d; padding: 18px 20px; border-radius: 16px; }
    [data-testid="stMetricValue"] { color: #f5f7ff; }
    div.stButton > button { border: 0; border-radius: 10px; background: linear-gradient(90deg,#8b5cf6,#5865f2); color: white; font-weight: 700; padding: .6rem 1.1rem; }
    div.stButton > button:hover { color: white; border: 1px solid #b7a2ff; }
    .hero { padding: 30px 34px; border: 1px solid #343c61; border-radius: 24px; background: linear-gradient(120deg,rgba(124,88,245,.23),rgba(40,190,198,.10)),#171e36; margin: 4px 0 22px; }
    .eyebrow { text-transform: uppercase; letter-spacing: .16em; color: #9daaff; font-size: .76rem; font-weight: 700; }
    .hero-title { font: 700 2.25rem 'Space Grotesk',sans-serif; color: #f8f8ff; margin: 8px 0; }
    .hero-copy { color: #c5cce2; font-size: 1.04rem; max-width: 780px; }
    .glass { background: #171e36; border: 1px solid #303958; border-radius: 18px; padding: 20px 22px; height: 100%; }
    .tag { display:inline-block; padding:5px 10px; border-radius:99px; background:#2b3152; color:#c8c1ff; font-size:.78rem; margin:3px 5px 3px 0; }
    .result-positive { border:1px solid #1ea876; background:rgba(18,139,99,.16); border-radius:16px; padding:18px 20px; }
    .result-negative { border:1px solid #e16a7a; background:rgba(173,53,73,.16); border-radius:16px; padding:18px 20px; }
    .small-muted { color:#9da8c4; font-size:.9rem; }
    hr { border-color:#303958; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    return 


@st.cache_data
def load_reviews():
    return pd.read_csv("data/reviews_prepared.csv")


def hero(eyebrow, title, copy):
    st.markdown(
        f'<div class="hero"><div class="eyebrow">{eyebrow}</div>'
        f'<div class="hero-title">{title}</div><div class="hero-copy">{copy}</div></div>',
        unsafe_allow_html=True,
    )


st.sidebar.markdown(
    "<div style='font:700 1.35rem Space Grotesk;color:#f4f6ff'>✦ ReviewSense</div>"
    "<div style='color:#8793b6;font-size:.85rem;margin:4px 0 18px'>Sentiment intelligence</div>",
    unsafe_allow_html=True,
)
page = st.sidebar.radio(
    "WORKSPACE",
    ["Overview", "Sentiment Studio", "Dataset Explorer", "Model Performance", "Project Guide"],
)
st.sidebar.markdown("---")
st.sidebar.markdown("<span class='tag'>LSTM</span><span class='tag'>TensorFlow</span><span class='tag'>40K reviews</span>", unsafe_allow_html=True)
st.sidebar.markdown("<div class='small-muted' style='margin-top:24px'>7th semester · Deep Learning Mini Project</div>", unsafe_allow_html=True)

st.markdown("<div class='eyebrow'>DEEP LEARNING · PRODUCT REVIEWS</div>", unsafe_allow_html=True)
st.title("ReviewSense")
st.caption("Understand customer feedback with a trained LSTM sentiment model.")


if page == "Overview":
    hero("LIVE PROJECT DASHBOARD", "Turn reviews into clear signals.", "Explore the review dataset, test a live prediction, and inspect how the model performs—all in one place.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Test accuracy", "87.29%", "8,000 unseen reviews")
    c2.metric("Training dataset", "40,000", "balanced reviews")
    c3.metric("Sentiment classes", "2", "positive · negative")
    c4.metric("Model", "LSTM", "3 training epochs")

    st.markdown("### Sentiment snapshot")
    left, right = st.columns([1.05, 1])
    with left:
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.markdown("**Balanced review set**")
        st.bar_chart(pd.DataFrame({"Reviews": [20000, 20000]}, index=["Positive", "Negative"]), color="#8b7cf6")
        st.markdown("<div class='small-muted'>The prepared dataset contains equal numbers of positive and negative reviews.</div></div>", unsafe_allow_html=True)
    with right:
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.markdown("**What happens behind the scenes**")
        st.markdown("<span class='tag'>Review text</span> → <span class='tag'>Word vectors</span> → <span class='tag'>LSTM</span> → <span class='tag'>Sentiment</span>", unsafe_allow_html=True)
        st.write("The model reads word order and context, then returns a probability for positive sentiment.")
        st.markdown("<div class='small-muted'>Try a review in Sentiment Studio to see it work.</div></div>", unsafe_allow_html=True)
    st.markdown("### Explore the workspace")
    a, b, c = st.columns(3)
    a.markdown("<div class='glass'><h3>01 · Sentiment Studio</h3><p>Enter any review and get a prediction with a confidence score.</p></div>", unsafe_allow_html=True)
    b.markdown("<div class='glass'><h3>02 · Dataset Explorer</h3><p>Browse labeled examples and compare the class balance.</p></div>", unsafe_allow_html=True)
    c.markdown("<div class='glass'><h3>03 · Model Performance</h3><p>Review precision, recall, F1 score, and the confusion matrix.</p></div>", unsafe_allow_html=True)


elif page == "Sentiment Studio":
    hero("INTERACTIVE PREDICTION", "What do customers think?", "Paste a product review below. The saved LSTM model will analyze it and return a sentiment prediction.")
    examples = {
        "Write my own review": "",
        "Positive example": "The product is excellent, tastes fresh, and arrived quickly. I will definitely buy it again!",
        "Negative example": "Poor quality and disappointing taste. The package arrived damaged and I would not buy it again.",
    }
    chosen = st.selectbox("Quick start", list(examples.keys()))
    review = st.text_area("Product review", value=examples[chosen], height=155, placeholder="Type or paste a product review…")
    submitted = st.button("✦ Analyze review", type="primary")
    if submitted:
        if not review.strip():
            st.warning("Enter a review first, or select one of the sample reviews.")
        else:
            with st.spinner("Reading review…"):
                score = float(load_model().predict(tf.constant([review]), verbose=0)[0][0])
            sentiment = "Positive" if score >= 0.5 else "Negative"
            confidence = score if score >= 0.5 else 1 - score
            st.session_state["last_result"] = {"review": review, "sentiment": sentiment, "confidence": confidence, "positive_probability": score}
            history = st.session_state.setdefault("history", [])
            history.insert(0, {"Sentiment": sentiment, "Confidence": f"{confidence:.1%}", "Review": review[:100]})
            st.session_state["history"] = history[:5]

    if "last_result" in st.session_state:
        result = st.session_state["last_result"]
        css_class = "result-positive" if result["sentiment"] == "Positive" else "result-negative"
        st.markdown(f"<div class='{css_class}'><span class='eyebrow'>MODEL PREDICTION</span><h2>{result['sentiment']} sentiment</h2><span class='small-muted'>Confidence: {result['confidence']:.1%}</span></div>", unsafe_allow_html=True)
        st.markdown("#### Positive sentiment probability")
        st.progress(result["positive_probability"])
        st.caption(f"Positive {result['positive_probability']:.1%} · Negative {1-result['positive_probability']:.1%}")
    if st.session_state.get("history"):
        st.markdown("### Recent analyses")
        st.dataframe(pd.DataFrame(st.session_state["history"]), use_container_width=True, hide_index=True)


elif page == "Dataset Explorer":
    hero("DATA EXPLORATION", "Meet the reviews behind the model.", "Explore the balanced set of Amazon food reviews used for training and evaluation.")
    try:
        data = load_reviews()
        counts = data["sentiment"].value_counts()
        x1, x2, x3 = st.columns(3)
        x1.metric("Prepared reviews", f"{len(data):,}")
        x2.metric("Positive", f"{counts.get('positive', 0):,}")
        x3.metric("Negative", f"{counts.get('negative', 0):,}")
        left, right = st.columns([.8, 1.2])
        with left:
            st.markdown("### Class distribution")
            st.bar_chart(counts.rename_axis("Sentiment").to_frame("Reviews"), color="#36c9b4")
        with right:
            st.markdown("### Browse examples")
            chosen_sentiment = st.selectbox("Filter by sentiment", ["All", "positive", "negative"])
            view = data if chosen_sentiment == "All" else data[data["sentiment"] == chosen_sentiment]
            st.dataframe(view.sample(min(12, len(view)), random_state=7)[["sentiment", "Text"]], use_container_width=True, hide_index=True, height=390)
    except FileNotFoundError:
        st.error("Could not find data/reviews_prepared.csv. Run prepare_data.py first.")


elif page == "Model Performance":
    hero("MODEL EVALUATION", "Strong results on unseen reviews.", "Performance is measured on a held-out test set that the model did not train on.")
    p, r, f = st.columns(3)
    p.metric("Test accuracy", "87.29%")
    r.metric("Test examples", "8,000", "4,000 per class")
    f.metric("Macro F1", "0.87")
    st.markdown("### Per-class performance")
    metrics = pd.DataFrame({"Precision": [0.90, 0.85], "Recall": [0.84, 0.91], "F1-score": [0.87, 0.88]}, index=["Negative", "Positive"])
    st.dataframe(metrics.style.format("{:.2f}"), use_container_width=True)
    st.bar_chart(metrics, color=["#8b7cf6", "#36c9b4", "#f4b860"])
    matrix_path = Path("confusion_matrix.png")
    if matrix_path.exists():
        st.markdown("### Confusion matrix")
        st.image(str(matrix_path), caption="Actual vs. predicted sentiment on the test set", use_container_width=True)
    with st.expander("How to read these metrics"):
        st.write("Precision tells how often a predicted class was correct. Recall tells how many real examples of that class were found. F1-score balances precision and recall.")


else:
    hero("PROJECT NOTES", "How ReviewSense was built.", "A complete mini-project workflow: data preparation, model training, evaluation, and an interactive prediction interface.")
    st.markdown("### End-to-end workflow")
    steps = st.columns(4)
    for col, number, title, detail in zip(steps, ["01", "02", "03", "04"], ["Prepare", "Represent", "Learn", "Evaluate"], ["Filter ratings; balance classes", "Convert text to token sequences", "Train an embedding + LSTM", "Measure test-set performance"]):
        col.markdown(f"<div class='glass'><span class='eyebrow'>{number}</span><h3>{title}</h3><p>{detail}</p></div>", unsafe_allow_html=True)
    st.markdown("### Model architecture")
    st.markdown("<span class='tag'>TextVectorization · 20K vocabulary</span> → <span class='tag'>Embedding · 64 dimensions</span> → <span class='tag'>LSTM · 64 units</span> → <span class='tag'>Dense · ReLU</span> → <span class='tag'>Sigmoid · 2 classes</span>", unsafe_allow_html=True)
    st.markdown("### Dataset labels")
    st.write("Amazon Fine Food Reviews. Ratings 1–2 are labeled negative, ratings 4–5 positive, and neutral 3-star reviews are excluded. The project uses 20,000 reviews per class.")
    st.markdown("### Tools")
    st.markdown("<span class='tag'>Python</span><span class='tag'>TensorFlow / Keras</span><span class='tag'>Pandas</span><span class='tag'>Scikit-learn</span><span class='tag'>Streamlit</span>", unsafe_allow_html=True)

st.markdown("<hr><div class='small-muted'>ReviewSense · Product Review Sentiment Analysis · Deep Learning Mini Project</div>", unsafe_allow_html=True)
