from __future__ import annotations

import html
import os
import sys
from functools import lru_cache
from pathlib import Path

import gradio as gr
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from absa_project.predict import TransformerAspectSentiment, suggest_aspects

try:
    import spaces
except ImportError:
    class _LocalSpaces:
        @staticmethod
        def GPU(*args, **kwargs):
            if args and callable(args[0]):
                return args[0]

            def decorator(func):
                return func

            return decorator

    spaces = _LocalSpaces()

MODEL_REPO_ID = os.getenv("MODEL_REPO_ID")
SAMPLE_CSV = Path(__file__).resolve().parent / "examples" / "sample_reviews.csv"

SENTIMENT_META = {
    "positive": {"class": "positive", "label": "Positive", "tone": "Keeps this aspect working"},
    "negative": {"class": "negative", "label": "Negative", "tone": "Needs attention"},
    "neutral": {"class": "neutral", "label": "Neutral", "tone": "Monitor context"},
    "conflict": {"class": "conflict", "label": "Conflict", "tone": "Mixed signal"},
}


@lru_cache(maxsize=1)
def model() -> TransformerAspectSentiment:
    return TransformerAspectSentiment(MODEL_REPO_ID)


def parse_aspects(aspect_text: str, review: str) -> list[str]:
    raw = [item.strip() for item in (aspect_text or "").replace("\n", ",").split(",")]
    aspects = [item for item in raw if item]
    return aspects or suggest_aspects(review)


def _empty_summary(message: str) -> str:
    return f"""
<div class="result-empty">
  <span>{message}</span>
</div>
"""


def render_sentiment_summary(rows: list[dict]) -> str:
    cards = []
    for row in rows:
        sentiment = str(row["sentiment"]).lower()
        meta = SENTIMENT_META.get(sentiment, {"class": "neutral", "label": sentiment.title(), "tone": "Review"})
        confidence = int(round(float(row["confidence"]) * 100))
        aspect = html.escape(str(row["aspect"]))
        cards.append(
            f"""
<article class="sentiment-card {meta['class']}">
  <div class="sentiment-card-head">
    <span class="aspect-name">{aspect}</span>
    <span class="sentiment-label">{meta['label']}</span>
  </div>
  <div class="confidence-row">
    <span>{meta['tone']}</span>
    <strong>{confidence}%</strong>
  </div>
  <div class="confidence-track" aria-label="{confidence}% confidence"><span style="width: {confidence}%"></span></div>
</article>
"""
        )
    counts = pd.Series([str(row["sentiment"]).lower() for row in rows]).value_counts().to_dict()
    mix = " / ".join(f"{count} {html.escape(label)}" for label, count in counts.items())
    return f"""
<section class="result-panel">
  <div class="result-head">
    <div><span class="section-kicker">Analysis</span><h2>Aspect sentiment</h2></div>
    <span class="result-mix">{mix}</span>
  </div>
  <div class="sentiment-grid">{''.join(cards)}</div>
</section>
"""


@spaces.GPU(duration=30)
def analyze_review(review: str, aspect_text: str) -> tuple[pd.DataFrame, str]:
    review = (review or "").strip()
    if not review:
        return pd.DataFrame(columns=["aspect", "sentiment", "confidence"]), _empty_summary("Enter a review to see aspect sentiment.")
    aspects = parse_aspects(aspect_text, review)
    if not aspects:
        return pd.DataFrame(columns=["aspect", "sentiment", "confidence"]), _empty_summary("No aspect candidates were found.")
    rows = model().predict_many(review, aspects)
    table = pd.DataFrame(rows)
    table["confidence"] = table["confidence"].map(lambda value: round(value, 3))
    return table[["aspect", "sentiment", "confidence"]], render_sentiment_summary(rows)


@spaces.GPU(duration=60)
def analyze_csv(file) -> pd.DataFrame:
    if file is None:
        return pd.DataFrame(columns=["row", "aspect", "sentiment", "confidence"])
    frame = pd.read_csv(file.name)
    text_col = "review" if "review" in frame.columns else "text" if "text" in frame.columns else None
    if text_col is None:
        raise gr.Error("CSV must include a 'review' or 'text' column.")
    rows = []
    for index, record in frame.iterrows():
        review = str(record[text_col])
        if "aspects" in frame.columns and pd.notna(record.get("aspects")):
            aspects = parse_aspects(str(record["aspects"]), review)
        elif "aspect" in frame.columns and pd.notna(record.get("aspect")):
            aspects = [str(record["aspect"])]
        else:
            aspects = suggest_aspects(review, limit=5)
        for prediction in model().predict_many(review, aspects):
            rows.append(
                {
                    "row": index + 1,
                    "review": review[:120],
                    "aspect": prediction["aspect"],
                    "sentiment": prediction["sentiment"],
                    "confidence": round(prediction["confidence"], 3),
                }
            )
    return pd.DataFrame(rows)


THEME_CSS = """
:root {
  --ink: #151922;
  --subtle: #5a6474;
  --canvas: #f3f5f8;
  --panel: #ffffff;
  --blue: #2457d6;
  --blue-dark: #163eaa;
  --blue-soft: #e8efff;
  --mint: #1e8d74;
  --mint-soft: #e5f5ef;
  --coral: #d94b5f;
  --coral-soft: #fdecef;
  --gold: #b67a00;
  --gold-soft: #fff3d6;
  --neutral: #536f9f;
  --neutral-soft: #eaf0f8;
  --line: #dce1e8;
}
/* Gradio's API and settings overlays use their own dark-mode tokens. */
.dark,
[data-theme="dark"] {
  --body-background-fill: #f3f5f8 !important;
  --body-text-color: #151922 !important;
  --body-text-color-subdued: #5a6474 !important;
  --background-fill-primary: #ffffff !important;
  --background-fill-secondary: #f8f9fb !important;
  --block-background-fill: #ffffff !important;
  --block-border-color: #dce1e8 !important;
  --input-background-fill: #ffffff !important;
  --input-border-color: #cfd6e0 !important;
  --button-secondary-background-fill: #ffffff !important;
  --button-secondary-text-color: #151922 !important;
  --border-color-primary: #dce1e8 !important;
  color-scheme: light !important;
}
div[role="dialog"],
div[role="menu"],
.popover {
  background: #ffffff !important;
  border-color: var(--line) !important;
  color: var(--ink) !important;
}
div[role="dialog"] :is(h1, h2, h3, h4, p, label, span, button),
div[role="menu"] :is(p, label, span, button),
.popover :is(p, label, span, button) {
  color: var(--ink) !important;
}
div[role="dialog"] :is(input, textarea, select),
div[role="menu"] :is(input, textarea, select),
.popover :is(input, textarea, select) {
  background: #ffffff !important;
  border-color: #cfd6e0 !important;
  color: var(--ink) !important;
}
div[role="dialog"] pre,
div[role="dialog"] code {
  background: #f3f5f8 !important;
  color: #163eaa !important;
}
.gradio-container {
  background: var(--canvas);
  color: var(--ink);
  font-family: "Aptos", "Segoe UI", Inter, ui-sans-serif, system-ui, sans-serif;
  margin: 0 auto !important;
  max-width: 1240px !important;
  padding: 3.6rem 1.25rem 2rem !important;
}
#title-block {
  margin-bottom: 1rem;
}
.product-bar {
  align-items: center;
  background: var(--ink);
  color: #fff;
  display: grid;
  gap: 1rem;
  grid-template-columns: minmax(0, 1fr) auto;
  min-height: 92px;
  padding: 1.1rem 1.25rem;
}
.product-id {
  align-items: center;
  display: flex;
  gap: .9rem;
}
.product-mark {
  align-items: center;
  background: #8bc6ff;
  color: #0d2b68;
  display: flex;
  font-size: .92rem;
  font-weight: 900;
  height: 46px;
  justify-content: center;
  width: 46px;
}
.product-bar h1 {
  color: #fff !important;
  font-size: 1.55rem;
  line-height: 1.1;
  margin: 0;
}
.product-bar p {
  color: #b8c2d3 !important;
  font-size: .92rem;
  margin: .2rem 0 0;
}
.model-state {
  align-items: center;
  color: #dce5f5 !important;
  display: flex;
  font-size: .86rem;
  gap: .55rem;
  white-space: nowrap;
}
.model-state i {
  background: #51d4a8;
  height: 9px;
  width: 9px;
}
.project-strip {
  align-items: center;
  background: #fff;
  border: 1px solid var(--line);
  border-top: 0;
  display: grid;
  gap: 1.25rem;
  grid-template-columns: minmax(0, 1fr) repeat(3, auto);
  padding: .9rem 1.25rem;
}
.project-strip > p {
  color: var(--subtle);
  margin: 0;
  max-width: 620px;
}
.metric {
  border-left: 1px solid var(--line);
  min-width: 112px;
  padding-left: 1rem;
}
.metric strong,
.metric span {
  display: block;
}
.metric strong {
  color: var(--ink);
  font-size: 1.02rem;
}
.metric span {
  color: var(--subtle);
  font-size: .76rem;
  margin-top: .08rem;
}
.app-shell {
  background: var(--panel);
  border: 1px solid var(--line);
  padding: 1rem;
}
.app-shell .tabs {
  border-color: var(--line) !important;
}
.tab-nav {
  border-bottom: 1px solid var(--line) !important;
  gap: 1.1rem !important;
}
.tab-nav button {
  font-weight: 700 !important;
  padding-left: .15rem !important;
  padding-right: .15rem !important;
}
button.primary {
  background: var(--blue) !important;
  border-color: var(--blue) !important;
  box-shadow: none !important;
  font-weight: 750 !important;
  min-height: 46px !important;
}
button.primary:hover {
  background: var(--blue-dark) !important;
}
button.secondary {
  border-color: var(--line) !important;
}
textarea,
.dataframe,
.wrap,
.file-preview {
  border-color: var(--line) !important;
}
span[data-testid="block-info"] {
  color: var(--ink) !important;
  font-weight: 700 !important;
}
.info-text {
  color: var(--subtle) !important;
}
.label-wrap .label {
  border-radius: 4px !important;
}
.result-panel {
  border-top: 1px solid var(--line);
  margin-top: 1rem;
  padding-top: 1rem;
}
.result-head {
  align-items: end;
  display: flex;
  gap: 1rem;
  justify-content: space-between;
  margin-bottom: .8rem;
}
.section-kicker {
  color: var(--blue);
  font-size: .72rem;
  font-weight: 800;
  letter-spacing: .1em;
  text-transform: uppercase;
}
.result-head h2 {
  color: var(--ink);
  font-size: 1.22rem;
  margin: .15rem 0 0;
}
.result-mix {
  color: var(--subtle);
  font-size: .82rem;
}
.sentiment-grid {
  display: grid;
  gap: .65rem;
  grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
}
.sentiment-card {
  background: var(--mint-soft);
  border-left: 4px solid var(--mint);
  padding: .85rem;
}
.sentiment-card.negative { background: var(--coral-soft); border-color: var(--coral); }
.sentiment-card.neutral { background: var(--neutral-soft); border-color: var(--neutral); }
.sentiment-card.conflict { background: var(--gold-soft); border-color: var(--gold); }
.sentiment-card-head,
.confidence-row {
  align-items: center;
  display: flex;
  gap: .5rem;
  justify-content: space-between;
}
.aspect-name {
  color: var(--ink);
  font-size: 1.05rem;
  font-weight: 800;
}
.sentiment-label {
  color: var(--mint);
  font-size: .78rem;
  font-weight: 850;
  text-transform: uppercase;
}
.sentiment-card.negative .sentiment-label { color: var(--coral); }
.sentiment-card.neutral .sentiment-label { color: var(--neutral); }
.sentiment-card.conflict .sentiment-label { color: var(--gold); }
.confidence-row {
  color: var(--subtle);
  font-size: .82rem;
  margin-top: .85rem;
}
.confidence-row strong {
  color: var(--ink);
  font-size: .86rem;
  font-weight: 700;
}
.confidence-track {
  background: rgba(21, 25, 34, .10);
  height: 4px;
  margin-top: .45rem;
}
.confidence-track span {
  background: var(--mint);
  display: block;
  height: 100%;
}
.negative .confidence-track span { background: var(--coral); }
.neutral .confidence-track span { background: var(--neutral); }
.conflict .confidence-track span { background: var(--gold); }
.result-empty {
  border-top: 1px solid var(--line);
  color: var(--subtle);
  margin-top: 1rem;
  padding: 1rem .1rem .25rem;
}
.batch-copy {
  color: var(--subtle);
  font-size: .9rem;
  line-height: 1.5;
  margin: 0;
}
@media (max-width: 700px) {
  .gradio-container {
    padding: 3.7rem .7rem 1.25rem !important;
  }
  .product-bar,
  .project-strip {
    grid-template-columns: minmax(0, 1fr);
  }
  .model-state {
    border-top: 1px solid #313a49;
    padding-top: .75rem;
  }
  .project-strip {
    gap: .7rem;
  }
  .metric {
    border-left: 0;
    border-top: 1px solid var(--line);
    display: grid;
    grid-template-columns: 5rem 1fr;
    padding: .55rem 0 0;
  }
  .metric span {
    margin: 0;
  }
  .app-shell {
    padding: .7rem;
  }
  #single-input-row {
    flex-direction: column !important;
  }
  #single-input-row > div {
    min-width: 0 !important;
    width: 100% !important;
  }
  .result-head {
    align-items: start;
    flex-direction: column;
  }
}
"""

APP_THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.blue,
    neutral_hue=gr.themes.colors.slate,
).set(
    body_background_fill="#f3f5f8",
    body_background_fill_dark="#f3f5f8",
    body_text_color="#151922",
    body_text_color_dark="#151922",
    body_text_color_subdued="#5a6474",
    body_text_color_subdued_dark="#5a6474",
    block_background_fill="#ffffff",
    block_background_fill_dark="#ffffff",
    block_border_color="#dce1e8",
    block_border_color_dark="#dce1e8",
    block_info_text_color="#5a6474",
    block_info_text_color_dark="#5a6474",
    block_label_text_color="#151922",
    block_label_text_color_dark="#151922",
    input_background_fill="#ffffff",
    input_background_fill_dark="#ffffff",
    input_background_fill_hover="#ffffff",
    input_background_fill_hover_dark="#ffffff",
    input_border_color="#cfd6e0",
    input_border_color_dark="#cfd6e0",
    accordion_text_color="#151922",
    accordion_text_color_dark="#151922",
    table_border_color="#dce1e8",
    table_border_color_dark="#dce1e8",
    table_even_background_fill="#f8f9fb",
    table_even_background_fill_dark="#f8f9fb",
    table_odd_background_fill="#ffffff",
    table_odd_background_fill_dark="#ffffff",
    table_text_color="#151922",
    table_text_color_dark="#151922",
)


with gr.Blocks(title="AspectLens ABSA") as demo:
    with gr.Column(elem_id="title-block"):
        gr.HTML(
            """
<header class="product-bar">
  <div class="product-id">
    <span class="product-mark">AL</span>
    <div><h1>AspectLens</h1><p>Aspect-based sentiment analysis</p></div>
  </div>
  <div class="model-state"><i></i> Fine-tuned DistilBERT</div>
</header>
<div class="project-strip">
  <p>Compare sentiment across the specific parts of a customer review.</p>
  <div class="metric"><strong>78.3%</strong><span>accuracy</span></div>
  <div class="metric"><strong>+9.5 pts</strong><span>weighted F1</span></div>
  <div class="metric"><strong>4</strong><span>sentiment labels</span></div>
</div>
"""
        )
    with gr.Column(elem_classes=["app-shell"]):
      with gr.Tabs():
        with gr.Tab("Single review"):
            with gr.Row(elem_id="single-input-row"):
                with gr.Column(scale=1, min_width=320):
                    review_input = gr.Textbox(
                        label="Customer review",
                        lines=5,
                        value="The food was excellent, but the service was slow and the bill felt high.",
                    )
                with gr.Column(scale=1, min_width=320):
                    aspect_input = gr.Textbox(
                        label="Aspects",
                        lines=5,
                        value="food, service, bill",
                        placeholder="Comma-separated aspects. Leave blank to auto-suggest.",
                        info="Comma-separated; leave blank to suggest likely aspects.",
                    )
            analyze_button = gr.Button("Analyze review", variant="primary")
            sentiment_summary = gr.HTML(value=_empty_summary("Run the model to see the aspect readout."))
            with gr.Accordion("Prediction table", open=False):
                sentiment_table = gr.Dataframe(
                    label="Detailed predictions",
                    headers=["aspect", "sentiment", "confidence"],
                    interactive=False,
                )
            analyze_button.click(analyze_review, [review_input, aspect_input], [sentiment_table, sentiment_summary])
            with gr.Accordion("Example reviews", open=False):
                gr.Examples(
                    examples=[
                        [
                            "The waiter was friendly and the pasta tasted fresh, but the dining room was painfully loud.",
                            "waiter, pasta, dining room",
                        ],
                        [
                            "Delivery was fast, packaging was careful, and the price was reasonable for the portion size.",
                            "delivery, packaging, price, portion size",
                        ],
                        [
                            "The menu looked exciting, but our steak was overcooked and the server forgot our drinks.",
                            "menu, steak, server, drinks",
                        ],
                    ],
                    inputs=[review_input, aspect_input],
                )
        with gr.Tab("Batch CSV"):
            gr.HTML('<p class="batch-copy">Score multiple reviews using a <strong>review</strong> or <strong>text</strong> column. Add an optional <strong>aspect</strong> or <strong>aspects</strong> column.</p>')
            gr.DownloadButton("Download sample CSV", value=str(SAMPLE_CSV), variant="secondary")
            csv_input = gr.File(label="Review CSV", file_types=[".csv"], height=170)
            batch_button = gr.Button("Score CSV", variant="primary")
            batch_output = gr.Dataframe(label="Batch predictions", interactive=False)
            batch_button.click(analyze_csv, csv_input, batch_output)


if __name__ == "__main__":
    demo.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        css=THEME_CSS,
        theme=APP_THEME,
    )
