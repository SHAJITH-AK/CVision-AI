# Resume Analyzer & Job Suitability Predictor

A locally runnable Flask web app that reads a PDF resume, predicts the closest job category with a machine-learning classifier, and scores how well the resume evidences the skills for a chosen role (or a pasted job description).

## What it uses (and doesn't)

| Part | Technology |
|---|---|
| Category prediction | TF-IDF (word 1-2 grams) + **Logistic Regression** or **calibrated Linear SVM**, whichever scores better in cross-validation |
| Skill extraction | Curated skill lexicon with alias and boundary-aware matching |
| Suitability score | Documented weighted heuristic (see below) - *not* a trained model |
| PDF text | pdfplumber |
| Frontend | HTML, CSS, JavaScript, Chart.js (bundled locally, so it works offline) |

This is **classical machine learning, not deep learning**. Sentence Transformers / BERT are not used; the code is structured so they could be added if you have enough real labelled data.

## IMPORTANT: about the dataset

No real, redistributable resume dataset ships with this project. `dataset/resumes.csv` is **synthetic** (1,280 template-generated resumes, 160 per role, produced by `generate_dataset.py`). It is labelled as synthetic in `dataset/dataset_info.json` and on the Model performance page. Metrics measured on it (about 85% accuracy) describe separation of generated text and will be optimistic for real resumes. For real use, put real labelled resumes in `dataset/resumes.csv` (columns `resume_text`, `category`) and retrain.

## Features

- Drag-and-drop PDF upload with validation, size/page limits, text preview and clear errors
- Scanned/image PDFs detected and explained (OCR is not included)
- 8 target roles plus "Other" (requires a pasted job description), or paste any job description
- Predicted category with top-3 probabilities, matched/missing skills, evidence snippets, resume-quality checks, suggestions
- Charts: required-skill match, score components, fit across all roles
- Model performance page rendered from real evaluation output (accuracy, precision, recall, F1, confusion matrix, class distribution, limitations)

## Requirements

- Python 3.10 - 3.13 (developed on 3.12)
- Windows, macOS or Linux

## Setup (Windows, VS Code terminal / PowerShell)

```powershell
cd resume-analyzer
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

If PowerShell blocks activation: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, then activate again.

macOS/Linux: `python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt`

## Dataset and training

A trained model (`models/resume_model.joblib`) and its metrics (`models/metrics.json`) are already included, built with the pinned scikit-learn version. To regenerate everything:

```powershell
python generate_dataset.py     # rewrites dataset/resumes.csv (synthetic)
python train_model.py          # trains, compares models, evaluates, saves model + metrics
```

Training is reproducible (fixed seed 42). `train_model.py`:

1. Cleans data and removes duplicates **before** splitting
2. Makes a stratified 80/20 train/test split
3. Compares both candidates with 5-fold CV and grid search on the **training split only** (the TF-IDF vectoriser is inside the pipeline, so it is refit per fold - no leakage)
4. Evaluates the winner **once** on the untouched test split
5. Saves the pipeline and `metrics.json`

The model is loaded once at server start; it is never retrained per request.

## Run the app

```powershell
python app.py
```

Open **http://127.0.0.1:5000** in your browser.

## Run the tests

```powershell
pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
```

Tests cover PDF extraction, scanned/fake/oversized files, role validation, job-description mode, score ranges, temp-file cleanup, and that the metrics endpoint returns real evaluation output.

## How the scores work

**Model probability** - the classifier's probability for the predicted *category*. Both candidate models output real probabilities (Logistic Regression natively; the SVM via sigmoid calibration, never raw decision scores). Calibration error (ECE) on the test split is shown next to it. Similarity values are never labelled as confidence.

**Suitability score (0-100)** - weighted sum:

| Component | Weight |
|---|---|
| Required skills evidenced in the resume | 45 |
| Preferred skills evidenced | 15 |
| TF-IDF cosine similarity to the role profile / job description (similarity / 0.5, capped at 1) | 20 |
| Experience (section present +0.5, estimated years >=1 +0.25, >=3 +0.25) | 10 |
| Education (+0.5), projects (+0.3), certifications (+0.2) | 10 |

The weights and the 0.5 similarity cap are judgement calls, not fitted or validated against hiring outcomes. With a pasted job description, every recognised skill in it counts as required. The result is an automated estimate, not a hiring decision, and not a probability of being hired.

"Missing" means *not evidenced in the extracted text*. The skill may be phrased differently or lost in PDF extraction, so quality checks say "Not detected", not "absent".

## Privacy and security

- Uploads are limited to 5 MB, PDF extension **and** `%PDF-` header, max 15 pages
- Files are written to a randomly named temp file, parsed, and deleted in a `finally` block
- No resume text is stored server-side; results live in the browser tab's `sessionStorage`
- All dynamic text is inserted with `textContent` (no HTML injection)
- Runs on `127.0.0.1` only by default. Do not expose it publicly without adding authentication and HTTPS.

## Project structure

```
resume-analyzer/
  app.py                  Flask routes, upload handling, error handlers
  train_model.py          Training + evaluation script
  generate_dataset.py     Synthetic dataset generator
  requirements.txt / requirements-dev.txt
  models/                 resume_model.joblib, metrics.json
  dataset/                resumes.csv, dataset_info.json
  utils/                  resume_parser, text_preprocessing, skill_extractor, job_profiles, prediction
  templates/              index, analyze, results, model_performance, about, base, error
  static/                 css/style.css, js/{main,analyze,results}.js, js/vendor/chart.umd.js
  tests/                  test_pipeline.py
```

## Troubleshooting

| Problem | Fix |
|---|---|
| "Trained model not found" | Run `python train_model.py` |
| Warning about scikit-learn version when loading the model | Install the pinned versions, or retrain with your installed version |
| "Very little text could be extracted" | The PDF is a scan/image. Export a text-based PDF from Word or Google Docs |
| Skills missing from a multi-column resume | PDF extraction may scramble columns; check the text preview, try a single-column layout |
| `ModuleNotFoundError` | Activate the virtual environment, then `pip install -r requirements.txt` |
| Port 5000 in use | Change `port=5000` at the bottom of `app.py` |
| `pip` can't find a pinned version for your Python | Use Python 3.10-3.13, or relax pins in `requirements.txt` and retrain |

## Replace or retrain the model

1. Put your data in `dataset/resumes.csv` with columns `resume_text` and `category`. Category names should match the roles in `utils/job_profiles.py` (edit that file to add roles and their required/preferred skills).
2. Delete or edit `dataset/dataset_info.json` (the app labels data as synthetic only if it says so).
3. `python train_model.py`, then restart `python app.py`.

## Extending skills and roles

Add skills and aliases in `utils/skill_extractor.py`; change role requirements in `utils/job_profiles.py`.
