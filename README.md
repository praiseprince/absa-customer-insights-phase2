# AspectLens

AspectLens is an aspect-based sentiment analysis application for customer reviews. It classifies sentiment toward specific review aspects so a mixed review can be separated into signals such as `food: positive`, `service: negative`, and `bill: negative`.

- Live app: https://praiseprince-absa-customer-insights-phase2.hf.space
- Hugging Face Space: https://huggingface.co/spaces/praiseprince/absa-customer-insights-phase2
- Fine-tuned model: https://huggingface.co/praiseprince/absa-customer-insights-model

## Results

Evaluation uses a stratified 80/20 split of 3,668 labeled SemEval restaurant ABSA examples.

| Model | Accuracy | Macro-F1 | Weighted-F1 |
| --- | ---: | ---: | ---: |
| TF-IDF + Logistic Regression | 0.665 | 0.510 | 0.676 |
| Fine-tuned DistilBERT | **0.783** | **0.541** | **0.771** |

The fine-tuned model improves accuracy by 0.118 and weighted-F1 by 0.095. The main limitation is the rare `conflict` label, which has only 18 held-out examples.

## Repository

```text
Group-1-Phase-2-code-*.ipynb       complete modeling and evaluation workflow
app.py                             Gradio deployment interface
src/absa_project/predict.py        hosted-model inference
src/absa_project/text.py           shared input formatting
src/absa_project/config.py         label and model-repository configuration
examples/sample_reviews.csv        batch-scoring example
```

The notebook is the authoritative Phase II modeling artifact. It contains raw
dataset loading, preprocessing, baseline fitting, DistilBERT fine-tuning,
evaluation, comparison, and error analysis inline. The deployment remains a
separate application layer: the completed checkpoint was uploaded to the
Hugging Face model repository, and `app.py` loads that checkpoint for inference.

## Run Locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH="$PWD/src"

# Open and run Group-1-Phase-2-code-Praise-Marya-Ashmit.ipynb.
# Set RUN_FINE_TUNING=True only when retraining on a suitable GPU.
python app.py
```

To use the Hub-hosted checkpoint instead of a local model, set `MODEL_REPO_ID=praiseprince/absa-customer-insights-model`.

## Application

The Gradio interface supports:

- Single-review analysis with optional aspect suggestions
- Per-aspect sentiment and confidence scores
- CSV batch scoring using `review` or `text` and optional `aspect` or `aspects` columns
- A downloadable sample CSV

The public application runs on a Hugging Face Space and loads the fine-tuned DistilBERT checkpoint from the Hugging Face model repository.

## Team Contributions

| Member | Phase II contribution |
| --- | --- |
| Praise | Data preparation and baseline modeling, notebook and repository organization, metric verification, and presentation of the problem and data sections |
| Marya | DistilBERT fine-tuning and evaluation, error analysis, report methods and results, and presentation of the model and evaluation sections |
| Ashmit | Gradio application and batch workflow, Hugging Face Space deployment and testing, deployment documentation and submission QA, and presentation of the demo and next steps |

## Sources

- Pontiki et al. (2014), SemEval-2014 Task 4: https://aclanthology.org/S14-2004/
- Dataset: https://huggingface.co/datasets/tomaarsen/setfit-absa-semeval-restaurants
- DistilBERT: https://doi.org/10.48550/arXiv.1910.01108
- Hugging Face Spaces: https://huggingface.co/docs/hub/en/spaces-overview
