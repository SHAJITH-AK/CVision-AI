# CVision AI — Resume Analyzer & Job Suitability Predictor

An AI-powered resume analysis web application built with Flask and Machine Learning. CVision AI analyzes PDF resumes, predicts suitable job categories, identifies technical and soft skills, and evaluates how well a resume matches a selected role or job description.

## ✨ Features

- **PDF Resume Upload:** Upload a resume in PDF format using the web interface.
- **Job Category Prediction:** Predicts the most relevant job category using a trained Machine Learning classifier.
- **Top-3 Predictions:** View the leading job-category predictions and their model probabilities.
- **Skill Extraction:** Detects technical skills and soft skills from extracted resume text.
- **Job Suitability Score:** Calculates a score from 0–100 using skill matches, text similarity, experience indicators, education, projects, and certifications.
- **Job Description Matching:** Analyze a resume against a selected role or a custom job description.
- **Skill Gap Analysis:** Identify skills detected in the resume and skills that were not found.
- **Resume Quality Checks:** View feedback and suggestions to improve resume content.
- **Interactive Visualizations:** Explore skill matches, score components, and suitability across roles.
- **Model Performance Dashboard:** Review evaluation metrics, class distribution, and the confusion matrix.
- **Privacy-Conscious Processing:** Temporary uploaded PDF files are deleted after parsing, and resume text is not intentionally stored server-side.

## 🧠 Machine Learning Approach

CVision AI uses a classical Machine Learning pipeline rather than a deep learning model.

### Text Classification

1. Extract text from the uploaded PDF using `pdfplumber`.
2. Clean and normalize the extracted text.
3. Convert the text into numerical features using TF-IDF with word unigrams and bigrams.
4. Pass the features to the trained classification pipeline.
5. Predict the most relevant job category and return the model's class probabilities.

### Model Selection

Two candidate models are compared during training:

- **TF-IDF + Logistic Regression**
- **TF-IDF + Calibrated Linear Support Vector Machine (SVM)**

The project selects the candidate with the better cross-validation macro F1 score.

The currently included model is **TF-IDF + Logistic Regression**.

### Supported Job Categories

- Data Scientist
- Machine Learning Engineer
- Python Developer
- Web Developer
- Data Analyst
- Software Engineer
- Cybersecurity Analyst
- Cloud Engineer

The application also supports an **Other** option for custom job-description analysis.

### How the Suitability Score Works

The 0–100 suitability score uses a weighted heuristic based on:

| Component | Weight |
|---|---:|
| Required skills | 45% |
| Preferred skills | 15% |
| Text similarity | 20% |
| Experience indicators | 10% |
| Education, projects, and certifications | 10% |

**Important:** The suitability score is a rule-based estimate, not a trained model output or a probability of getting hired. Missing skills mean they were not detected in the extracted text; they may still be present in the original resume.

## 📊 Model Evaluation

The included evaluation report records the following results:

| Metric | Score |
|---|---:|
| Test accuracy | 85.55% |
| Macro precision | 86.58% |
| Macro recall | 85.55% |
| Macro F1 score | 85.59% |
| Expected calibration error | 3.36% |

### Dataset Disclaimer

The included dataset contains **1,280 synthetic, template-generated resume records**, with 160 records per job category.

These evaluation results measure performance on synthetic demonstration data. They do not establish equivalent performance on real-world resumes. The model should be evaluated on representative, properly labelled real data before being used in recruitment decisions.

## 🛠️ Technologies Used

### Backend and Machine Learning
- Python
- Flask
- Scikit-learn
- NumPy
- Pandas
- Joblib
- pdfplumber

### Frontend
- HTML5
- CSS3
- JavaScript
- Chart.js

### Development and Testing
- Git and GitHub
- Python virtual environments
- Python `unittest`

## 📁 Project Structure

```text
CVision-AI/
├── app.py
├── generate_dataset.py
├── train_model.py
├── requirements.txt
├── requirements-dev.txt
├── README.md
├── dataset/
│   ├── dataset_info.json
│   └── resumes.csv
├── models/
│   ├── metrics.json
│   └── resume_model.joblib
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       ├── analyze.js
│       ├── main.js
│       ├── results.js
│       └── vendor/
│           └── chart.umd.js
├── templates/
│   ├── about.html
│   ├── analyze.html
│   ├── base.html
│   ├── error.html
│   ├── index.html
│   ├── model_performance.html
│   └── results.html
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py
└── utils/
    ├── __init__.py
    ├── job_profiles.py
    ├── prediction.py
    ├── resume_parser.py
    ├── skill_extractor.py
    └── text_preprocessing.py
```

## ⚙️ Installation and Setup

### Prerequisites

Install the following before starting:

- Python 3.10–3.13
- Git
- A code editor such as Visual Studio Code
- A modern web browser

### 1. Clone the Repository

```bash
git clone https://github.com/SHAJITH-AK/CVision-AI.git
cd CVision-AI
```

### 2. Create a Virtual Environment

**Windows — Command Prompt**

```bat
python -m venv venv
venv\Scripts\activate
```

**Windows — PowerShell**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The trained model and evaluation metrics are already included, so you normally do not need to retrain the model just to run the application.

### 4. Run the Application

```bash
python app.py
```

Open the following URL in your browser:

**http://127.0.0.1:5000**

Keep the terminal running while using the application.

## 🚀 How to Use

1. Start CVision AI using `python app.py`.
2. Open the local URL in your browser.
3. Navigate to the resume analysis page.
4. Upload a PDF resume.
5. Select a target job role or provide a custom job description.
6. Submit the resume for analysis.
7. Review the predicted category and top-3 model probabilities.
8. Explore detected skills, missing skill evidence, suitability score, and improvement suggestions.
9. Open the Model Performance page to inspect the evaluation metrics.

### Upload Limitations

- Accepted file format: PDF
- Maximum upload size: 5 MB
- Maximum supported pages: 15
- Scanned/image-only PDFs may not contain extractable text because OCR is not included.

## 🧪 Testing

Install the development dependencies:

```bash
pip install -r requirements-dev.txt
```

Run the test suite:

```bash
python -m unittest discover -s tests -v
```

The tests cover important parts of the resume analysis pipeline, including PDF parsing, upload validation, role selection, score ranges, and metrics responses.

## 🔄 Retraining the Model

To regenerate the synthetic dataset and train a new model:

```bash
python generate_dataset.py
python train_model.py
```

The training process compares the candidate models and saves the selected pipeline and updated evaluation metrics.

For experiments with real labelled data, prepare `dataset/resumes.csv` with these columns:

```csv
resume_text,category
"Python, SQL, statistics and data analysis experience","Data Analyst"
"Software development, algorithms and unit testing","Software Engineer"
```

These rows are illustrative examples only. Use representative, appropriately licensed data for actual training and evaluation.

## 🔐 Privacy and Security

- Uploads are validated for PDF format and size.
- Temporary PDF copies are deleted after parsing.
- Resume text is not intentionally stored server-side by the application.
- Analysis results are held in the browser's session storage.
- Uploaded resumes may contain personal information; use test documents when experimenting.

**Before public deployment:** The development server binds to localhost by default. A production deployment requires an appropriate production server, HTTPS, and suitable access controls and operational safeguards. Do not treat the current application as a secure recruitment system without additional review.

## 🛣️ Future Improvements

- Add OCR support for scanned resumes.
- Evaluate the classifier on a properly labelled real-world dataset.
- Expand job categories and skill dictionaries.
- Improve support for multi-column PDF layouts.
- Add secure authentication and privacy controls for hosted deployments.
- Add more comprehensive automated tests and monitoring.

## ⚠️ Disclaimer

CVision AI is an educational Machine Learning project intended for resume analysis and career guidance. Predictions and suitability scores may be inaccurate or incomplete. They should not be used as the sole basis for hiring, rejection, or other consequential employment decisions.

## 👨‍💻 Author

**Shajith Abdul Kareem**

GitHub: [@SHAJITH-AK](https://github.com/SHAJITH-AK)

To Contact: shajith8047@gmail.com

## 📄 License

No license is specified in this repository yet. Add a suitable open-source license if you want others to have explicit permission to reuse, modify, and distribute the project.
