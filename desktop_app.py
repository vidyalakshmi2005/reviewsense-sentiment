from collections import Counter
from pathlib import Path
import csv
import re
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import tensorflow as tf


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "sentiment_model.keras"
DATA_PATH = BASE_DIR / "data" / "reviews_prepared.csv"
_model = None

BG = "#101629"
PANEL = "#1a2340"
TEXT = "#eef1ff"
MUTED = "#aab4d0"
PURPLE = "#8065e8"

ASPECT_PATTERNS = {
    "Taste / flavor": r"\b(taste|flavou?r|bland|tasteless|delicious|salty|sweet|bitter|spicy|seasoning)\b",
    "Freshness / expiry": r"\b(fresh|stale|expired|expiry|spoiled|rotten|old)\b",
    "Texture": r"\b(texture|chewy|hard|soft|crunchy|soggy|dry|mushy|gritty|smooth)\b",
    "Packaging": r"\b(package|packaging|seal|sealed|leak|leaking|damaged|broken|container|wrapper)\b",
    "Delivery": r"\b(delivery|delivered|shipping|shipped|arrived|arrival|delayed|late delivery)\b",
    "Price / value": r"\b(price|cost|expensive|overpriced|cheap|value|worth|affordable)\b",
    "Portion / quantity": r"\b(size|portion|quantity|amount|ounces|ounce|small|large)\b",
    "Ingredients": r"\b(ingredient|ingredients|sugar|salt|preservative|artificial|allergen|allergy)\b",
}
NEGATIVE_CUES = re.compile(r"\b(bad|poor|awful|terrible|disgusting|stale|spoiled|expired|bland|tasteless|weak|burnt|bitter|cold|late|delayed|damaged|broken|leaking|expensive|overpriced|small|missing|hard|soggy|dry|wrong|disappointed|problem|issue|worse|lacks?|not enough|too salty|too sweet|too bitter|too spicy|not\s+(?:good|great|fresh|worth|recommend))\b", re.I)
POSITIVE_CUES = re.compile(r"\b(good|great|excellent|delicious|tasty|fresh|flavorful|smooth|crunchy|affordable|worth|fast|intact|recommend|love|enjoy|perfect|amazing|wonderful|no\s+(?:issues?|problems?|complaints?))\b", re.I)


def analyze_aspects(text):
    """Return transparent keyword-based food aspect clues, not trained explanations."""
    findings = []
    for sentence in re.split(r"(?<=[.!?])\s+", text.strip()):
        if not sentence:
            continue
        for aspect, pattern in ASPECT_PATTERNS.items():
            if not re.search(pattern, sentence, re.I):
                continue
            cleaned_for_negative = re.sub(r"\bnot\s+(?:bad|poor|awful|terrible|expensive|overpriced|small)\b", "", sentence, flags=re.I)
            neg = bool(NEGATIVE_CUES.search(cleaned_for_negative))
            # Avoid treating the word after a negation (for example, "not good") as praise.
            cleaned_for_positive = re.sub(r"\bnot\s+(?:good|great|fresh|worth|recommend)\b", "", sentence, flags=re.I)
            pos = bool(POSITIVE_CUES.search(cleaned_for_positive)) or bool(re.search(r"\bnot\s+(?:bad|poor|awful|terrible|expensive|overpriced|small)\b", sentence, re.I))
            if neg and pos:
                label = "Mixed signals"
            elif neg:
                label = "Possible issue"
            elif pos:
                label = "Positive point"
            else:
                label = "Mentioned — check context"
            findings.append({"aspect": aspect, "finding": label, "evidence": sentence[:220]})
    # Keep one row per aspect while preserving the most informative clue.
    merged = {}
    priority = {"Mentioned — check context": 0, "Positive point": 1, "Possible issue": 2, "Mixed signals": 3}
    for item in findings:
        previous = merged.get(item["aspect"])
        if previous is None or priority[item["finding"]] > priority[previous["finding"]]:
            merged[item["aspect"]] = item
    return list(merged.values())


ASPECT_ACTIONS = {
    "Taste / flavor": (
        "Taste-test small recipe variations; adjust salt, sweetness, or spice gradually and compare batches.",
        "Keep the liked flavor profile consistent; ask tasters which flavor notes they value before changing the recipe.",
    ),
    "Freshness / expiry": (
        "Check supplier quality, expiry dates, storage temperature, and first-in/first-out stock rotation.",
        "Maintain the current sourcing and storage routine; spot-check freshness for every batch.",
    ),
    "Texture": (
        "Review cooking time, temperature, moisture, mixing, and holding time; compare a test batch.",
        "Record the cooking time, temperature, and holding steps that produce the preferred texture.",
    ),
    "Packaging": (
        "Inspect sealing and containers, then test whether packaging prevents leaks or damage during handling.",
        "Keep the packaging checks that protect product quality and verify seals on each batch.",
    ),
    "Delivery": (
        "Check dispatch timing, transit time, and temperature control so food arrives in good condition.",
        "Maintain reliable dispatch and temperature-control steps; periodically review delivery feedback.",
    ),
    "Price / value": (
        "Review ingredient cost and portion value; test any recipe change for taste and quality before adopting it.",
        "Keep quality consistent while reviewing portion and ingredient costs before making price changes.",
    ),
    "Portion / quantity": (
        "Use a scale or standard scoop to check serving size and recipe yield across batches.",
        "Keep portion sizes consistent with a measured scoop or scale and check them periodically.",
    ),
    "Ingredients": (
        "Verify ingredient quality, substitutions, allergen handling, and ingredient labels with the food team.",
        "Maintain ingredient standards and clear allergen controls; document any recipe substitutions.",
    ),
}


def improvement_ideas(sentiment, findings):
    if not findings:
        if sentiment == "Positive":
            return ["No specific food aspect was identified. Keep the recipe consistent and ask customers what they would like refined."]
        return ["No specific food aspect was identified. Review the full comment and ask for details before changing the recipe or process."]

    ideas = []
    for item in findings:
        aspect = item["aspect"]
        is_positive = sentiment == "Positive" and item["finding"] not in ("Possible issue", "Mixed signals")
        if sentiment == "Negative" and item["finding"] not in ("Possible issue", "Mixed signals"):
            continue
        action_index = 1 if is_positive else 0
        action = ASPECT_ACTIONS.get(aspect)
        if action:
            lead = "Build on the positive feedback" if is_positive else "Check this area"
            ideas.append(f"{lead} ({aspect}): {action[action_index]}")
    if not ideas:
        return ["The review does not clearly identify a preparation issue. Read the quoted evidence and ask the reviewer what specifically should change."]
    return ideas[:4]


def get_model():
    global _model
    if _model is None:
        _model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return _model


def make_heading(parent, eyebrow, title, description):
    tk.Label(parent, text=eyebrow.upper(), bg=BG, fg="#b9a9ff", font=("Arial", 9, "bold")).pack(anchor="w", pady=(0, 5))
    tk.Label(parent, text=title, bg=BG, fg=TEXT, font=("Arial", 22, "bold")).pack(anchor="w")
    tk.Label(parent, text=description, bg=BG, fg=MUTED, font=("Arial", 10), wraplength=850, justify="left").pack(anchor="w", pady=(5, 18))


def main():
    root = tk.Tk()
    root.title("ReviewSense — Sentiment Analyzer")
    root.geometry("980x760")
    root.minsize(760, 600)
    root.configure(bg=BG)

    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure("TNotebook", background=BG, borderwidth=0)
    style.configure("TNotebook.Tab", background="#293451", foreground=TEXT, padding=(14, 9), font=("Arial", 10, "bold"))
    style.map("TNotebook.Tab", background=[("selected", PURPLE)], foreground=[("selected", "white")])
    style.configure("Treeview", background="#131d33", fieldbackground="#131d33", foreground=TEXT, rowheight=28)
    style.configure("Treeview.Heading", background="#35415e", foreground=TEXT, font=("Arial", 9, "bold"))
    style.configure("Review.Horizontal.TProgressbar", troughcolor="#35415e", background=PURPLE, bordercolor=PANEL, lightcolor=PURPLE, darkcolor=PURPLE)

    shell = tk.Frame(root, bg=BG, padx=28, pady=22)
    shell.pack(fill="both", expand=True)
    tk.Label(shell, text="REVIEWSENSE  ·  LOCAL DESKTOP APP", bg=BG, fg="#b9a9ff", font=("Arial", 10, "bold")).pack(anchor="w", pady=(0, 12))

    tabs = ttk.Notebook(shell)
    tabs.pack(fill="both", expand=True)
    pages = {}
    page_canvases = {}
    for name in ("Overview", "Sentiment Studio", "Dataset Explorer", "Model Performance", "Batch Analysis", "Project Guide"):
        tab_host = tk.Frame(tabs, bg=BG)
        tabs.add(tab_host, text=name)
        canvas = tk.Canvas(tab_host, bg=BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(tab_host, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        page = tk.Frame(canvas, bg=BG, padx=22, pady=22)
        page_window = canvas.create_window((0, 0), window=page, anchor="nw")
        page.bind("<Configure>", lambda _event, c=canvas: c.configure(scrollregion=c.bbox("all")))
        canvas.bind("<Configure>", lambda event, c=canvas, w=page_window: c.itemconfigure(w, width=event.width))
        pages[name] = page
        page_canvases[name] = canvas

    def scroll_active_page(event):
        selected_tab = tabs.select()
        if not selected_tab:
            return
        selected_name = tabs.tab(selected_tab, "text")
        canvas = page_canvases[selected_name]
        if getattr(event, "num", None) == 4:
            canvas.yview_scroll(-1, "units")
        elif getattr(event, "num", None) == 5:
            canvas.yview_scroll(1, "units")
        else:
            units = -int(event.delta / 120)
            if units:
                canvas.yview_scroll(units, "units")

    root.bind_all("<MouseWheel>", scroll_active_page)
    root.bind_all("<Button-4>", scroll_active_page)
    root.bind_all("<Button-5>", scroll_active_page)

    session_history = []
    session_counts = Counter()

    # Overview
    page = pages["Overview"]
    make_heading(page, "Your workspace", "Understand the sentiment in text", "ReviewSense estimates whether a piece of English text sounds positive or negative. Choose a section above to explore the project.")
    cards = tk.Frame(page, bg=BG)
    cards.pack(fill="x", pady=(0, 18))
    card_accents = ["#a895ff", "#36c9b4", "#f4b860", "#ff7585"]
    for (title, value, note), accent in zip([("TEST ACCURACY", "87.29%", "held-out test reviews"), ("TRAINING DATA", "40,000", "balanced examples"), ("SENTIMENTS", "2", "positive and negative"), ("MODEL", "LSTM", "TensorFlow neural network")], card_accents):
        card = tk.Frame(cards, bg=PANEL, padx=14, pady=14)
        card.pack(side="left", fill="both", expand=True, padx=(0, 9))
        tk.Frame(card, bg=accent, height=4).pack(fill="x", pady=(0, 10))
        tk.Label(card, text=title, bg=PANEL, fg=accent, font=("Arial", 8, "bold")).pack(anchor="w")
        tk.Label(card, text=value, bg=PANEL, fg=accent, font=("Arial", 19, "bold")).pack(anchor="w", pady=(5, 2))
        tk.Label(card, text=note, bg=PANEL, fg=MUTED, font=("Arial", 8), wraplength=150).pack(anchor="w")
    tk.Label(page, text="How it works", bg=BG, fg=TEXT, font=("Arial", 15, "bold")).pack(anchor="w", pady=(8, 8))
    tk.Label(page, text="Your text  →  word sequence  →  LSTM model  →  Positive / Negative prediction", bg=PANEL, fg=TEXT, padx=16, pady=18, font=("Arial", 12)).pack(fill="x")
    tk.Label(page, text="The model was trained on Amazon Fine Food product reviews. It accepts general text, but may be less reliable on topics unlike its training data. Neutral or mixed text is forced into one of the two labels.", bg=BG, fg=MUTED, wraplength=850, justify="left", font=("Arial", 9)).pack(anchor="w", pady=(18, 0))
    insight_row = tk.Frame(page, bg=BG)
    insight_row.pack(fill="both", expand=True, pady=(18, 0))
    session_chart_panel = tk.Frame(insight_row, bg=PANEL, padx=14, pady=12)
    session_chart_panel.pack(side="left", fill="both", expand=True, padx=(0, 10))
    tk.Label(session_chart_panel, text="LIVE SESSION SENTIMENT", bg=PANEL, fg="#36c9b4", font=("Arial", 9, "bold")).pack(anchor="w")
    session_summary = tk.Label(session_chart_panel, text="No predictions yet", bg=PANEL, fg=TEXT, font=("Arial", 11, "bold"))
    session_summary.pack(anchor="w", pady=(8, 8))
    session_chart = tk.Canvas(session_chart_panel, height=36, bg=PANEL, highlightthickness=0)
    session_chart.pack(fill="x")
    recent_panel = tk.Frame(insight_row, bg=PANEL, padx=12, pady=12)
    recent_panel.pack(side="left", fill="both", expand=True)
    tk.Label(recent_panel, text="RECENT PREDICTIONS", bg=PANEL, fg="#f4b860", font=("Arial", 9, "bold")).pack(anchor="w", pady=(0, 7))
    recent_table = ttk.Treeview(recent_panel, columns=("sentiment", "confidence", "text"), show="headings", height=4)
    for column, label, width in (("sentiment", "RESULT", 90), ("confidence", "CONF.", 65), ("text", "TEXT", 220)):
        recent_table.heading(column, text=label)
        recent_table.column(column, width=width, stretch=(column == "text"))
    recent_table.pack(fill="both", expand=True)
    session_chart.bind("<Configure>", lambda _event: update_session_dashboard())

    def update_session_dashboard():
        positive = session_counts.get("Positive", 0)
        negative = session_counts.get("Negative", 0)
        total = positive + negative
        session_summary.config(text=f"{total} analyzed  ·  {positive} positive  ·  {negative} negative")
        session_chart.delete("all")
        width = max(session_chart.winfo_width(), 220)
        positive_width = int((width - 4) * positive / total) if total else 0
        if total:
            session_chart.create_rectangle(2, 7, 2 + positive_width, 29, fill="#36c9b4", outline="")
            if positive_width < width - 4:
                session_chart.create_rectangle(2 + positive_width, 7, width - 2, 29, fill="#ff7585", outline="")
        else:
            session_chart.create_rectangle(2, 7, width - 2, 29, fill="#35415e", outline="")
        recent_table.delete(*recent_table.get_children())
        for item in reversed(session_history[-5:]):
            recent_table.insert("", "end", values=(item["sentiment"], item["confidence"], item["text"][:80]))

    # Sentiment Studio
    page = pages["Sentiment Studio"]
    make_heading(page, "Interactive prediction", "What does this text express?", "Type any English sentence, comment, or review. ReviewSense predicts overall sentiment and highlights possible food-related aspects in the wording.")
    tk.Label(page, text="TEXT TO ANALYZE", bg=BG, fg="#a895ff", font=("Arial", 9, "bold")).pack(anchor="w", pady=(0, 7))
    review_box = tk.Text(page, height=8, wrap="word", undo=True, bg="#131d33", fg=TEXT, insertbackground=TEXT, font=("Arial", 12), relief="flat", padx=12, pady=12)
    review_box.pack(fill="x")
    actions = tk.Frame(page, bg=BG)
    actions.pack(fill="x", pady=12)
    result_panel = tk.Frame(page, bg=PANEL, padx=18, pady=16)
    result_panel.pack(fill="x", pady=(4, 0))
    result_title = tk.Label(result_panel, text="Your prediction will appear here", bg=PANEL, fg=MUTED, font=("Arial", 16, "bold"))
    result_title.pack(anchor="w")
    confidence_label = tk.Label(result_panel, text="", bg=PANEL, fg=MUTED, font=("Arial", 10))
    confidence_label.pack(anchor="w", pady=(7, 0))
    confidence_bar = ttk.Progressbar(result_panel, style="Review.Horizontal.TProgressbar", maximum=100)
    confidence_bar.pack(fill="x", pady=(12, 0))
    tk.Label(result_panel, text="FOOD ASPECT CLUES  ·  keyword-based, explainable hints", bg=PANEL, fg="#f4b860", font=("Arial", 9, "bold")).pack(anchor="w", pady=(14, 6))
    aspect_table = ttk.Treeview(result_panel, columns=("aspect", "finding", "evidence"), show="headings", height=4)
    for column, label, width in (("aspect", "ASPECT", 145), ("finding", "FINDING", 170), ("evidence", "TEXT EVIDENCE", 520)):
        aspect_table.heading(column, text=label)
        aspect_table.column(column, width=width, stretch=(column == "evidence"))
    aspect_table.pack(fill="x")
    tk.Label(result_panel, text="PRACTICAL FOOD IMPROVEMENT IDEAS  ·  review before applying", bg=PANEL, fg="#36c9b4", font=("Arial", 9, "bold")).pack(anchor="w", pady=(12, 5))
    improvement_label = tk.Label(result_panel, text="Suggestions will appear after a prediction.", bg=PANEL, fg=TEXT, font=("Arial", 9), wraplength=820, justify="left")
    improvement_label.pack(anchor="w", fill="x")
    status = tk.Label(page, text="", bg=BG, fg=MUTED, font=("Arial", 9))
    status.pack(anchor="w", pady=(8, 0))

    def analyze():
        text = review_box.get("1.0", "end").strip()
        if not text:
            status.config(text="Enter some text first.", fg="#f4b860")
            return
        analyze_button.config(state="disabled", text="Analyzing…")
        status.config(text="Loading model and analyzing…", fg=MUTED)
        root.update_idletasks()
        try:
            score = float(get_model()(tf.constant([text]), training=False).numpy()[0][0])
            sentiment = "Positive" if score >= 0.5 else "Negative"
            confidence = score if score >= 0.5 else 1 - score
            result_title.config(text=f"{sentiment} sentiment", fg="#55c99a" if sentiment == "Positive" else "#ff7585")
            confidence_label.config(text=f"Confidence: {confidence:.1%}    ·    Positive {score:.1%}    ·    Negative {1-score:.1%}")
            confidence_bar["value"] = confidence * 100
            aspect_table.delete(*aspect_table.get_children())
            aspect_findings = analyze_aspects(text)
            if aspect_findings:
                for finding in aspect_findings:
                    aspect_table.insert("", "end", values=(finding["aspect"], finding["finding"], finding["evidence"]))
            else:
                aspect_table.insert("", "end", values=("No food aspect cues found", "—", "Try mentioning taste, freshness, texture, packaging, price, or delivery."))
            ideas = improvement_ideas(sentiment, aspect_findings)
            improvement_label.config(text="\n".join(f"• {idea}" for idea in ideas))
            session_counts[sentiment] += 1
            session_history.append({"sentiment": sentiment, "confidence": f"{confidence:.1%}", "text": text})
            update_session_dashboard()
            status.config(text="Prediction complete.", fg=MUTED)
        except Exception as error:
            status.config(text=f"Model error: {error}", fg="#ff7585")
        finally:
            analyze_button.config(state="normal", text="Analyze sentiment")

    analyze_button = tk.Button(actions, text="Analyze sentiment", command=analyze, bg=PURPLE, fg="white", activebackground="#6d55ca", relief="flat", cursor="hand2", padx=16, pady=9, font=("Arial", 10, "bold"))
    analyze_button.pack(side="left")
    food_positive_examples = [
        ("1 · Rich coffee flavor", "This coffee has a rich, smooth flavor and smells wonderfully fresh."),
        ("2 · Fresh snack", "These snacks taste fresh and have just the right amount of salt."),
        ("3 · Smooth soup texture", "The soup is delicious, creamy, and very easy to prepare."),
        ("4 · Crisp texture", "The crackers are crisp, tasty, and arrived in perfect condition."),
        ("5 · Great value", "Excellent quality for the price; this is worth buying again."),
        ("6 · Good packaging", "The package was sealed securely and everything inside was fresh."),
        ("7 · Sweet taste", "The tea has a pleasant aroma and a mild, refreshing taste."),
        ("8 · Good portion", "The portion size is generous and the product tastes wonderful."),
        ("9 · Good ingredients", "I like the simple ingredients and the natural flavor of this product."),
        ("10 · Fast delivery", "My order arrived quickly, the packaging was intact, and the food was delicious."),
    ]
    food_negative_examples = [
        ("1 · Too salty", "The flavor was much too salty, and I could barely taste anything else."),
        ("2 · Stale product", "The snacks tasted stale and were not fresh when the package arrived."),
        ("3 · Bland flavor", "The soup was bland and had almost no flavor, so I would not buy it again."),
        ("4 · Bad texture", "The crackers were soggy and soft instead of crisp."),
        ("5 · Damaged package", "The package was damaged and some of the product had spilled out."),
        ("6 · Late delivery", "The delivery was late, and the food arrived warm and disappointing."),
        ("7 · Poor value", "This product is overpriced for the small amount you get."),
        ("8 · Expired item", "The item arrived close to its expiry date and did not taste fresh."),
        ("9 · Strange aftertaste", "It has a bitter aftertaste and an unpleasant smell."),
        ("10 · Missing seal", "The container was not sealed properly, which made me concerned about its quality."),
    ]
    sample_row = tk.Frame(page, bg=BG)
    # Place examples directly under the text box, above results, so they remain visible.
    sample_row.pack(fill="x", pady=(0, 8), before=actions)
    tk.Label(sample_row, text="Try a food example:", bg=BG, fg=MUTED, font=("Arial", 9, "bold")).pack(side="left")
    sample_kind = tk.StringVar(value="Positive food reviews")
    sample_kind_box = ttk.Combobox(sample_row, textvariable=sample_kind, values=("Positive food reviews", "Negative food reviews"), state="readonly", width=22)
    sample_kind_box.pack(side="left", padx=(8, 8))
    sample_var = tk.StringVar()
    sample_box = ttk.Combobox(sample_row, textvariable=sample_var, state="readonly", width=34)
    sample_box.pack(side="left", padx=(0, 8))

    def update_food_samples(_event=None):
        examples = food_positive_examples if sample_kind.get().startswith("Positive") else food_negative_examples
        sample_box["values"] = [label for label, _text in examples]
        sample_var.set(examples[0][0])

    def load_food_sample():
        examples = food_positive_examples if sample_kind.get().startswith("Positive") else food_negative_examples
        selected_text = next((text for label, text in examples if label == sample_var.get()), "")
        review_box.delete("1.0", "end")
        review_box.insert("1.0", selected_text)
        update_text_stats()
        review_box.focus_set()

    sample_kind_box.bind("<<ComboboxSelected>>", update_food_samples)
    update_food_samples()
    tk.Button(sample_row, text="Load example", command=load_food_sample, bg="#293451", fg=TEXT, relief="flat", padx=11, pady=7, cursor="hand2").pack(side="left")
    text_stats = tk.Label(page, text="0 words · 0 characters", bg=BG, fg=MUTED, font=("Arial", 9))
    text_stats.pack(anchor="w", pady=(0, 4))

    def update_text_stats(_event=None):
        current_text = review_box.get("1.0", "end-1c")
        words = len(current_text.split())
        text_stats.config(text=f"{words} words · {len(current_text)} characters")

    review_box.bind("<KeyRelease>", update_text_stats)
    review_box.bind("<<Paste>>", lambda _event: root.after(50, update_text_stats))

    # Dataset Explorer
    page = pages["Dataset Explorer"]
    make_heading(page, "Data exploration", "Explore the examples used by the model", "The prepared dataset contains labeled Amazon Fine Food reviews. Browse a small sample below.")
    dataset_summary = tk.Label(page, text="", bg=BG, fg=TEXT, font=("Arial", 11, "bold"))
    dataset_summary.pack(anchor="w", pady=(0, 12))
    filter_row = tk.Frame(page, bg=BG)
    filter_row.pack(fill="x", pady=(0, 8))
    tk.Label(filter_row, text="Show:", bg=BG, fg=MUTED).pack(side="left")
    dataset_filter = tk.StringVar(value="All")
    filter_box = ttk.Combobox(filter_row, textvariable=dataset_filter, values=("All", "positive", "negative"), state="readonly", width=14)
    filter_box.pack(side="left", padx=8)
    table = ttk.Treeview(page, columns=("sentiment", "text"), show="headings", height=13)
    table.heading("sentiment", text="SENTIMENT")
    table.heading("text", text="REVIEW TEXT")
    table.column("sentiment", width=110, stretch=False)
    table.column("text", width=650)
    table.pack(fill="both", expand=True)
    table_rows = []

    def refresh_table(*_):
        table.delete(*table.get_children())
        selected = dataset_filter.get()
        shown = [row for row in table_rows if selected == "All" or row[0].lower() == selected]
        for sentiment, text in shown[:100]:
            table.insert("", "end", values=(sentiment.title(), text[:400]))

    filter_box.bind("<<ComboboxSelected>>", refresh_table)
    try:
        with DATA_PATH.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            table_rows = [(row.get("sentiment", ""), row.get("Text", "")) for row in reader]
        counts = Counter(label for label, _ in table_rows)
        dataset_summary.config(text=f"{len(table_rows):,} prepared examples     ·     {counts.get('positive', 0):,} positive     ·     {counts.get('negative', 0):,} negative")
        refresh_table()
    except (OSError, UnicodeError, csv.Error) as error:
        dataset_summary.config(text=f"Dataset unavailable: {error}", fg="#ff7585")

    # Model Performance
    page = pages["Model Performance"]
    make_heading(page, "Model evaluation", "How well did the model do?", "These scores were recorded from the held-out test split of 8,000 reviews.")
    perf_cards = tk.Frame(page, bg=BG)
    perf_cards.pack(fill="x", pady=(0, 18))
    for (title, value), accent in zip([("TEST ACCURACY", "87.29%"), ("TEST REVIEWS", "8,000"), ("MACRO F1", "0.87")], ["#a895ff", "#36c9b4", "#f4b860"]):
        card = tk.Frame(perf_cards, bg=PANEL, padx=18, pady=15)
        card.pack(side="left", fill="both", expand=True, padx=(0, 10))
        tk.Frame(card, bg=accent, height=4).pack(fill="x", pady=(0, 10))
        tk.Label(card, text=title, bg=PANEL, fg=accent, font=("Arial", 8, "bold")).pack(anchor="w")
        tk.Label(card, text=value, bg=PANEL, fg=accent, font=("Arial", 22, "bold")).pack(anchor="w", pady=(5, 0))
    tk.Label(page, text="Per-class metrics", bg=BG, fg=TEXT, font=("Arial", 15, "bold")).pack(anchor="w", pady=(2, 8))
    metrics = ttk.Treeview(page, columns=("precision", "recall", "f1", "support"), show="tree headings", height=3)
    metrics.heading("#0", text="CLASS")
    for col, label in (("precision", "PRECISION"), ("recall", "RECALL"), ("f1", "F1 SCORE"), ("support", "SUPPORT")):
        metrics.heading(col, text=label)
        metrics.column(col, anchor="center", width=140)
    metrics.insert("", "end", text="Negative", values=("0.90", "0.84", "0.87", "4,000"))
    metrics.insert("", "end", text="Positive", values=("0.85", "0.91", "0.88", "4,000"))
    metrics.pack(fill="x")
    tk.Label(page, text="Class score comparison", bg=BG, fg=TEXT, font=("Arial", 14, "bold")).pack(anchor="w", pady=(16, 6))
    performance_chart = tk.Canvas(page, height=175, bg=PANEL, highlightthickness=0)
    performance_chart.pack(fill="x")

    def draw_performance_chart(_event=None):
        performance_chart.delete("all")
        chart_width = max(performance_chart.winfo_width(), 600)
        colors = {"Precision": "#a895ff", "Recall": "#36c9b4", "F1": "#f4b860"}
        start_x = 125
        bar_width = chart_width - start_x - 65
        for index, (label, value) in enumerate((("Precision", 0.90), ("Recall", 0.84), ("F1", 0.87), ("Precision", 0.85), ("Recall", 0.91), ("F1", 0.88))):
            class_name = "Negative" if index < 3 else "Positive"
            row_y = 40 + index * 20
            if index in (0, 3):
                performance_chart.create_text(10, row_y + 4, text=class_name, anchor="w", fill="#eef1ff", font=("Arial", 9, "bold"))
            performance_chart.create_text(82, row_y + 4, text=label, anchor="w", fill="#aab4d0", font=("Arial", 8))
            performance_chart.create_rectangle(start_x, row_y - 3, start_x + bar_width, row_y + 11, fill="#35415e", outline="")
            performance_chart.create_rectangle(start_x, row_y - 3, start_x + int(bar_width * value), row_y + 11, fill=colors[label], outline="")
            performance_chart.create_text(start_x + bar_width + 8, row_y + 4, text=f"{value:.0%}", anchor="w", fill="#eef1ff", font=("Arial", 8, "bold"))
        legend_x = 125
        for label, color in colors.items():
            performance_chart.create_rectangle(legend_x, 10, legend_x + 10, 20, fill=color, outline="")
            performance_chart.create_text(legend_x + 15, 15, text=label, anchor="w", fill="#aab4d0", font=("Arial", 8))
            legend_x += 100

    performance_chart.bind("<Configure>", draw_performance_chart)
    tk.Label(page, text="Accuracy is not a guarantee for every new sentence. The training examples are food product reviews, so performance can change on other topics.", bg=BG, fg=MUTED, wraplength=850, justify="left", font=("Arial", 9)).pack(anchor="w", pady=(18, 0))

    # Batch Analysis
    page = pages["Batch Analysis"]
    make_heading(page, "Food review analysis", "Analyze a food-review CSV", "Choose a CSV of food reviews, select the review-text column, and export sentiment, confidence, aspect clues, and practical improvement ideas.")
    batch_path = tk.StringVar(value="No CSV selected")
    batch_rows = []
    batch_columns = []
    batch_output = []
    batch_status = tk.Label(page, textvariable=batch_path, bg=BG, fg=MUTED, font=("Arial", 9), wraplength=830, justify="left")
    batch_status.pack(anchor="w", pady=(0, 10))
    batch_controls = tk.Frame(page, bg=BG)
    batch_controls.pack(fill="x", pady=(0, 10))
    column_choice = tk.StringVar()
    column_box = ttk.Combobox(batch_controls, textvariable=column_choice, state="readonly", width=24)
    column_box.pack(side="left", padx=(10, 10))
    batch_tree = ttk.Treeview(page, columns=("sentiment", "confidence", "aspects", "text"), show="headings", height=12)
    for column, label, width in (("sentiment", "SENTIMENT", 105), ("confidence", "CONFIDENCE", 100), ("aspects", "ASPECT FINDINGS", 220), ("text", "TEXT", 440)):
        batch_tree.heading(column, text=label)
        batch_tree.column(column, width=width, stretch=(column == "text"))
    batch_insights = tk.Frame(page, bg=PANEL, padx=14, pady=10)
    batch_insights.pack(fill="x", pady=(0, 10))
    batch_summary = tk.Label(batch_insights, text="Batch metrics will appear after analysis.", bg=PANEL, fg=TEXT, font=("Arial", 10, "bold"))
    batch_summary.pack(anchor="w")
    batch_chart = tk.Canvas(batch_insights, height=24, bg=PANEL, highlightthickness=0)
    batch_chart.pack(fill="x", pady=(5, 0))
    batch_tree.pack(fill="both", expand=True)

    def update_batch_insights(scores):
        positive = sum(score >= 0.5 for score in scores)
        negative = len(scores) - positive
        mean_confidence = sum(score if score >= 0.5 else 1 - score for score in scores) / len(scores) if scores else 0
        batch_summary.config(text=f"{len(scores):,} analyzed  ·  {positive:,} positive  ·  {negative:,} negative  ·  mean confidence {mean_confidence:.1%}")
        batch_chart.delete("all")
        width = max(batch_chart.winfo_width(), 240)
        positive_width = int((width - 4) * positive / len(scores)) if scores else 0
        if scores:
            batch_chart.create_rectangle(2, 3, 2 + positive_width, 21, fill="#36c9b4", outline="")
            if positive_width < width - 4:
                batch_chart.create_rectangle(2 + positive_width, 3, width - 2, 21, fill="#ff7585", outline="")

    def choose_csv():
        nonlocal batch_rows, batch_columns, batch_output
        chosen_file = filedialog.askopenfilename(title="Choose a CSV file", filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if not chosen_file:
            return
        try:
            with open(chosen_file, "r", encoding="utf-8-sig", newline="") as source:
                reader = csv.DictReader(source)
                batch_columns = reader.fieldnames or []
                batch_rows = list(reader)
            if not batch_rows or not batch_columns:
                raise ValueError("The selected CSV is empty or has no column headings.")
            batch_output = []
            column_box["values"] = batch_columns
            likely_text_col = next((c for c in batch_columns if any(word in c.lower() for word in ("text", "review", "comment", "sentence", "message"))), batch_columns[0])
            column_choice.set(likely_text_col)
            batch_path.set(f"Selected: {chosen_file}  ·  {len(batch_rows):,} rows")
            batch_tree.delete(*batch_tree.get_children())
            for row in batch_rows[:8]:
                batch_tree.insert("", "end", values=("Preview", "—", "Run analysis to find aspects", str(row.get(likely_text_col, ""))[:200]))
        except Exception as error:
            messagebox.showerror("Could not read CSV", str(error), parent=root)

    def run_batch():
        nonlocal batch_output
        if not batch_rows:
            messagebox.showinfo("Choose a CSV", "Choose a CSV file first.", parent=root)
            return
        selected_column = column_choice.get()
        texts = [str(row.get(selected_column, "") or "") for row in batch_rows]
        if not selected_column or not any(text.strip() for text in texts):
            messagebox.showwarning("No text found", "Select a column that contains text.", parent=root)
            return
        batch_analyze_button.config(state="disabled", text="Analyzing…")
        batch_path.set(f"Analyzing {len(texts):,} rows. Please wait…")
        root.update_idletasks()
        try:
            model = get_model()
            scores = []
            for start in range(0, len(texts), 32):
                group = texts[start:start + 32]
                values = model(tf.constant(group), training=False).numpy().reshape(-1).tolist()
                scores.extend(values)
                batch_path.set(f"Analyzed {min(start + len(group), len(texts)):,} of {len(texts):,} rows…")
                root.update_idletasks()
            batch_output = []
            for original, text, score in zip(batch_rows, texts, scores):
                confidence = score if score >= 0.5 else 1 - score
                result = dict(original)
                result["predicted_sentiment"] = "Positive" if score >= 0.5 else "Negative"
                result["confidence"] = f"{confidence:.4f}"
                aspect_findings = analyze_aspects(text)
                result["aspect_findings"] = "; ".join(f"{item['aspect']}: {item['finding']} ({item['evidence']})" for item in aspect_findings)
                result["improvement_ideas"] = " ; ".join(improvement_ideas(result["predicted_sentiment"], aspect_findings))
                batch_output.append(result)
            update_batch_insights(scores)
            batch_tree.delete(*batch_tree.get_children())
            for original, text, score in zip(batch_rows[:100], texts[:100], scores[:100]):
                sentiment = "Positive" if score >= 0.5 else "Negative"
                confidence = score if score >= 0.5 else 1 - score
                aspects = analyze_aspects(text)
                aspect_summary = "; ".join(f"{item['aspect']}: {item['finding']}" for item in aspects) or "No known aspect cues"
                batch_tree.insert("", "end", values=(sentiment, f"{confidence:.1%}", aspect_summary, text[:200]))
            batch_path.set(f"Done: {len(batch_output):,} rows analyzed. Choose Export results to save a new CSV.")
            export_button.config(state="normal")
        except Exception as error:
            batch_path.set(f"Batch analysis failed: {error}")
        finally:
            batch_analyze_button.config(state="normal", text="Analyze CSV")

    def export_batch():
        if not batch_output:
            return
        output_file = filedialog.asksaveasfilename(title="Save predictions", defaultextension=".csv", initialfile="sentiment_predictions.csv", filetypes=[("CSV files", "*.csv")])
        if not output_file:
            return
        try:
            with open(output_file, "w", encoding="utf-8-sig", newline="") as destination:
                writer = csv.DictWriter(destination, fieldnames=list(batch_output[0].keys()))
                writer.writeheader()
                writer.writerows(batch_output)
            messagebox.showinfo("Export complete", f"Saved {len(batch_output):,} predictions to:\n{output_file}", parent=root)
        except Exception as error:
            messagebox.showerror("Could not save file", str(error), parent=root)

    tk.Button(batch_controls, text="Choose CSV…", command=choose_csv, bg="#293451", fg=TEXT, relief="flat", padx=13, pady=9, cursor="hand2").pack(side="left")
    batch_analyze_button = tk.Button(batch_controls, text="Analyze CSV", command=run_batch, bg=PURPLE, fg="white", relief="flat", padx=13, pady=9, cursor="hand2")
    batch_analyze_button.pack(side="left")
    export_button = tk.Button(batch_controls, text="Export results…", command=export_batch, bg="#293451", fg=TEXT, relief="flat", padx=13, pady=9, cursor="hand2", state="disabled")
    export_button.pack(side="left", padx=(8, 0))

    # Project Guide
    page = pages["Project Guide"]
    make_heading(page, "Project guide", "What is inside ReviewSense?", "A quick explanation of the project and how to use it.")
    guide = [
        ("1 · Sentiment Studio", "Enter your own text, or choose Positive food reviews / Negative food reviews to browse 10 sample texts in each group. Click Load example to put one into the text box, then click Analyze sentiment."),
        ("2 · Overview and session insights", "See the model summary, then watch a live positive/negative chart and the five most recent predictions from this app session."),
        ("3 · Dataset Explorer", "Browse labeled examples from data/reviews_prepared.csv and filter by sentiment."),
        ("4 · Model Performance", "Review held-out test accuracy, per-class precision, recall, F1, and a comparison chart."),
        ("5 · Batch Analysis", "Load a food-review CSV such as sample_food_reviews.csv, select its review-text column, analyze each row, see class counts and average confidence, then export predictions and food-aspect ideas. Your original file is not overwritten."),
        ("Food aspect clues", "The app scans the wording for taste/flavor, freshness, texture, packaging, delivery, price, portion, and ingredient terms. It shows the sentence as evidence and labels it as a possible issue, positive point, mixed signals, or just a mention. These are keyword-based hints, not explanations produced by the trained LSTM; review them yourself."),
        ("Food improvement ideas", "After a prediction, the app suggests practical checks related to the aspect mentioned, such as recipe taste tests, storage checks, cooking-time consistency, or package inspections. These are general suggestions; validate them with a food professional and the full customer feedback."),
        ("How the LSTM works", "TextVectorization turns text into sequences (20,000-word vocabulary, 150 tokens), an Embedding layer maps tokens to vectors, an LSTM reads word order, and a sigmoid output estimates positive sentiment."),
        ("Evaluation setup", "The dataset was split 80/20 with stratification. The model trained for 3 epochs and the reported test metrics come from 8,000 held-out reviews."),
        ("About the model", "This is an LSTM neural network trained on 40,000 Amazon Fine Food reviews. It accepts general English text, but it has not been trained on every subject. Its binary labels cannot represent neutral sentiment."),
        ("Run the app again", "Close this window. In File Explorer, open the project folder and double-click run_local.bat. The first run may take a while if TensorFlow needs installing."),
    ]
    for title, body in guide:
        box = tk.Frame(page, bg=PANEL, padx=15, pady=11)
        box.pack(fill="x", pady=(0, 9))
        tk.Label(box, text=title, bg=PANEL, fg="#a895ff", font=("Arial", 10, "bold")).pack(anchor="w")
        tk.Label(box, text=body, bg=PANEL, fg=TEXT, font=("Arial", 9), wraplength=830, justify="left").pack(anchor="w", pady=(4, 0))

    root.mainloop()


if __name__ == "__main__":
    main()


