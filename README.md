# 🏥 Hospital Patient Readmission Predictor
### *Machine Learning Risk Stratification Engine & Data-Driven AI Clinical Copilot*

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/pandas-2.0%2B-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o%20Mini-412991?logo=openai&logoColor=white)](https://openai.com/)
[![Tests](https://img.shields.io/badge/Tests-28%20Passed-10B981?logo=pytest&logoColor=white)](#-testing--quality-assurance)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An enterprise-ready clinical decision-support and workflow prototype built for diabetic inpatient discharge planning. The system pairs a regularized gradient-boosted decision tree pipeline (`HistGradientBoostingClassifier`) with an **in-memory deterministic dataset computation engine (`data_analytics.py`)**, an **AI Clinical Copilot (`chat_assistant.py`)**, and an **AI Follow-up Draft Assistant** to deliver actionable risk stratification, conversational explanation, and post-discharge communication coordination.

---

## 📑 Table of Contents
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Dataset & Prediction Target](#-dataset--prediction-target)
- [Model Performance & Benchmarks](#-model-performance--benchmarks)
- [Talk to Your Data: Data-Driven AI Copilot](#-talk-to-your-data-data-driven-ai-copilot)
- [Supported Dataset Queries Catalog](#-supported-dataset-queries-catalog)
- [Security & Secrets Management](#-security--secrets-management)
- [Repository Structure](#-repository-structure)
- [Installation & Quickstart](#-installation--quickstart)
- [Testing & Quality Assurance](#-testing--quality-assurance)
- [Clinical Boundaries & Limitations](#-clinical-boundaries--limitations)
- [Author & Acknowledgments](#-author--acknowledgments)

---

## 🌟 Key Features

1. **Multi-Dimensional Clinical Feature Pipeline:** Ingests 18 raw clinical, demographic, procedural, and medication variables; cleans missing values; maps ICD-9 codes to high-level clinical categories; and constructs engineered interaction signals (e.g. `service_density`, `total_prior_visits`, `prior_inpatient_ratio`).
2. **Calibrated ML Risk Stratification:** Serialized gradient boosting pipeline predicts patient readmission probability (%) and assigns calibrated risk categories:
   - 🟢 **Low Risk (< 42%):** Routine discharge packet; outpatient follow-up in 3–4 weeks.
   - 🟡 **Moderate Risk (42% – 55%):** Outpatient follow-up within 10–14 days; structured medication guide.
   - 🔴 **High Risk (≥ 55%):** High-touch protocol with 48h nurse telephone check-in, 5–7 day clinical appointment, and pharmacist medication reconciliation.
3. **💬 Talk to Your Data (Data-Driven AI Copilot):** Natural language conversational assistant powered by live Pandas queries against the 99,343-encounter cohort. Computes exact demographics, age rankings, medication correlations, and diagnoses without statistical guessing or hallucination.
4. **✉️ Human-in-the-Loop Follow-up Draft Generator:** Automatically compiles a structured post-discharge outreach letter with patient-specific timing, placeholders for clinic identifiers, and a mandatory clinician verification checklist. *Strictly draft-only: never sends automatically.*
5. **💡 Dual Operating Mode (Live LLM + Offline Fallback):** Operates with live OpenAI models (`gpt-4o-mini`, `gpt-4o`) to explain verified calculation payloads in natural language when an API key is provided, or in local deterministic markdown table mode when offline.
6. **🔒 Zero-Exposure Security Architecture:** Streamlit secrets isolation (`.streamlit/secrets.toml`), `.env` fallbacks, session-only overrides, and strict `.gitignore` configurations ensure zero API keys are ever committed to version control.

---

## 🏛 System Architecture

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Streamlit User Interface                           │
│  [Demographics]   [Clinical Diagnoses]   [Prior Visits]   [Medications/Labs] │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ Raw Encounters Data (18 Features)
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     Preprocessing & Feature Engineering                     │
│  • ICD-9 Code Grouping (Circulatory, Respiratory, Diabetes, etc.)           │
│  • Interaction Engineering: service_density, prior_inpatient_ratio          │
│  • Deterministic Label Encoding                                             │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ Processed 1x21 Feature Vector
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│               HistGradientBoosting Classifier (Trained Model)               │
│  • Predicts Readmission Probability (%) & Risk Classification (Low/Mod/High)│
│  • Generates Rule-Based Contributing Risk Drivers                           │
└──────────────────┬───────────────────────────────────────┬──────────────────┘
                   │                                       │
                   ▼                                       ▼
┌──────────────────────────────────────┐ ┌────────────────────────────────────┐
│      Assessment Dashboard & UI       │ │     AI Clinical Copilot Module     │
│  • Risk Badge (🟢/🟡/🔴)              │ │  • "Talk to Your Data" (Chatbot)   │
│  • Metric Cards & Visual Gauge       │ │  • In-Memory Pandas Analytics Engine│
│  • Contributing Risk Factors         │ │  • Multi-turn Follow-up Context    │
│  • Recommended Discharge Protocol    │ │  • AI Follow-up Draft Assistant    │
│                                      │ │  • [Live OpenAI / Local Mode]      │
└──────────────────────────────────────┘ └────────────────────────────────────┘
```

---

## 📊 Dataset & Prediction Target

### Dataset Overview
* **Source:** UCI Machine Learning Repository — Diabetes 130-US Hospitals (1999–2008).
* **Raw Sample Size:** 101,766 encounters across 130 medical centers.
* **Eligible Cohort:** 99,343 encounters after clinical exclusion of in-hospital expired or hospice-transferred patients (clinical benchmark: expired patients cannot be readmitted).

### Verified Target Variable Definition
In standard medical literature, readmission is often loosely cited as "30-day readmission." However, the model is trained on:
$$	ext{Target } y =  egin{cases} 1 & 	ext{if readmitted } (<30 	ext{ or } >30 	ext{ days}) \ 0 & 	ext{if not readmitted } (	ext{'NO'}) \end{cases}$$

> [!IMPORTANT]
> **Prediction Target Rigor:** The model classifies **any recorded hospital readmission** versus no readmission. It is **not** restricted solely to 30-day early readmissions. Documenting this distinction maintains scientific transparency and prevents misleading clinical claims.

---

## 📈 Model Performance & Benchmarks

The project evaluated multiple architectures, selecting regularized gradient boosting to eliminate severe decision-tree overfitting while maximizing calibration and discrimination.

| Metric | Baseline Random Forest | HistGradientBoosting (Selected) | Delta |
| :--- | :---: | :---: | :---: |
| **Train Accuracy** | 90.41% | 65.44% | *Regularized* |
| **Test Accuracy** | 55.67% | **63.54%** | **+7.87%** |
| **ROC-AUC Score** | 0.5720 | **0.6828** | **+0.1108** |
| **Overfitting Gap** | 34.74% | **1.90%** | **Overfitting Eliminated** |
| **Class 1 Precision** | 52.08% | **62.88%** | **+10.80%** |
| **Class 1 Recall** | 50.48% | **55.22%** | **+4.74%** |

*Evaluated on an 80/20 stratified holdout split ($N=19,869$ test encounters).*

---

## 💬 Talk to Your Data: Data-Driven AI Copilot

The chatbot in `chat_assistant.py` is backed by a deterministic analytical computation engine (`data_analytics.py`) that executes live Pandas queries on the dataset rather than hallucinating statistics.

### Architectural Workflow:
1. **Context-Aware Intent Classification:** Distinguishes **cohort dataset queries** (demographics, rates, med distributions) from **individual patient queries** (current simulated encounter drivers, probabilities).
2. **Deterministic Pandas Execution:** Computes exact metrics across all 99,343 post-discharge records (denominators, numerators, percentage rates).
3. **Data Integrity Safeguards:** If an OpenAI API key is active, verified numerical calculations are passed as the immutable ground-truth context to the LLM. The AI is strictly instructed to explain the verified payload without inventing numbers.
4. **Transparent Data Boundaries:** When asked about variables not collected in `diabetic_data.csv` (e.g. blood pressure, smoking, BMI, billing costs), the engine explicitly states the variable is absent, explains why, lists available columns, and suggests relevant clinical alternatives.

---

## 🎯 Supported Dataset Queries Catalog

Users can ask natural language questions across a broad spectrum of clinical and demographic dimensions:

| Category | Example Question | Analytical Query Method |
| :--- | :--- | :--- |
| **Gender Demographics** | *"How many male and female patients were readmitted?"* | Computes dynamic breakdown of Female (53,454), Male (45,886), and Unknown/Invalid (3) with respective rates. |
| **Age Brackets** | *"Which age groups have the highest readmission rates?"* | Ranks all 10 age brackets from highest (`[80-90)` at 50.36%) to lowest (`[0-10)` at 18.12%). |
| **Overall Metrics** | *"What percentage of patients were readmitted overall?"* | Computes cohort denominator ($N=99,343$), total readmitted ($46,816$, 47.13%), and breakdown into $<30$ vs $>30$ days. |
| **Medication & Insulin** | *"How do medications and insulin relate to readmission?"* | Evaluates Insulin titration (`Down` 54.06%, `Up` 52.96%, `Steady` 46.19%, `No` 44.57%) and regimen changes (`Ch` 49.43% vs `No` 45.13%). |
| **Primary Diagnoses** | *"How do primary diagnoses relate to readmission?"* | Groups ICD-9 codes into clinical categories (Diabetes: 51.39%, Respiratory: 49.72%, Circulatory: 48.16%, Neoplasms: 36.28%). |
| **Discharge Destination** | *"How does discharge destination affect readmission?"* | Evaluates Home (45.02%), Transferred Facility (49.77%), Home Health (54.21%), and AMA (49.92%). |
| **Admission Acuity** | *"What is the difference between emergency and elective admissions?"* | Computes readmission rates across Emergency (48.63%), Urgent (46.91%), and Elective (41.24%) admissions. |
| **Lab Diagnostics** | *"How do HbA1c and glucose results affect readmission?"* | Analyzes HbA1c testing categories (`>8`, `>7`, `Norm`, `None`) and maximum serum glucose thresholds. |
| **Custom Characteristics** | *"How many patients stayed in the hospital for 7 or more days?"* | Filters encounters by duration ($\ge 7$ days: 20,408 encounters, 51.17% readmission rate). |
| **Boundary Handling** | *"What is the average patient blood pressure or smoking history?"* | Flags absent features honestly and redirects to circulatory diagnoses or prior utilization. |

---

## 🔒 Security & Secrets Management

Security best practices are strictly enforced to protect API credentials:
* **Streamlit Secrets:** Configured in `.streamlit/secrets.toml` (git-ignored).
* **Git Protection:** `.gitignore` includes `.streamlit/secrets.toml`, `.env`, `*.env`, `.venv/`, `__pycache__/`, and all local backup files (`*bak*`).
* **Template Provided:** `.streamlit/secrets.toml.example` provides onboarding instructions without exposing live credentials.
* **Session Override:** Evaluators without file access can enter an OpenAI key via the sidebar password input; keys remain in memory only and are never written to disk.

---

## 📂 Repository Structure

```text
Hospital-Patient-Readmission-Predictor/
├── .streamlit/
│   ├── secrets.toml.example      # Template for secrets configuration
│   └── secrets.toml              # Local secrets (strictly git-ignored)
├── tests/
│   ├── __init__.py               # Test package initialization
│   ├── test_data_analytics.py    # Unit tests for Pandas calculations with synthetic ground-truth fixtures
│   ├── test_chat_assistant.py    # Unit tests for copilot, draft generator, routing & context
│   ├── test_preprocess.py        # Unit tests for preprocessing & ICD-9 mappings
│   └── test_model.py             # Unit tests for model loading & inference shape
├── app.py                        # Main Streamlit clinical GUI application
├── data_analytics.py             # Deterministic Pandas dataset calculation engine
├── chat_assistant.py             # Dual-mode AI Clinical Copilot & Draft Generator engine
├── preprocess.py                 # Feature engineering, encoding & data preparation
├── train_model.py                # Model training & serialization script
├── diabetic_data.csv             # UCI 130-US Hospitals dataset (19 MB)
├── readmission_model.pkl         # Trained model bundle with feature encoders
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore rules for secrets and virtualenvs
└── README.md                     # Comprehensive project documentation
```

---

## 🚀 Installation & Quickstart

### Prerequisites
* Python 3.10, 3.11, or 3.12
* Git

### 1. Clone the Repository
```bash
git clone https://github.com/YashashreeB-02/Hospital-Patient-Readmission-Predictor.git
cd Hospital-Patient-Readmission-Predictor
```

### 2. Create and Activate Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. (Optional) Configure OpenAI Secrets
Copy the template and insert your OpenAI API key:
```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```
Edit `.streamlit/secrets.toml`:
```toml
OPENAI_API_KEY = "sk-your-openai-api-key-here"
OPENAI_MODEL = "gpt-4o-mini"
```
*(If no API key is provided, the application runs seamlessly in local verified calculation mode!)*

### 5. Launch the Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🧪 Testing & Quality Assurance

The test suite includes **28 automated tests** covering independent synthetic ground-truth fixtures, empty dataset edge cases, missing value handling, natural language routing, model inference, and feature preprocessing:

```bash
pytest -v
```

**Test Suite Coverage Summary:**
```text
tests/test_chat_assistant.py (10 tests)
  • API key validation and runtime override
  • Cohort summary and denominator verification
  • Real dataset gender, age, and medication calculations
  • Multi-turn conversation context resolution
  • Distinction between patient-specific predictions and cohort metrics
  • Clinical outreach draft format and pre-send safeguards

tests/test_data_analytics.py (11 tests)
  • Independent 10-row synthetic fixture with verified ground truth
  • Exact gender, age, and insulin rate computations on known fixtures
  • Graceful empty dataset handling (0 records without ZeroDivisionError)
  • Missing and null values handling
  • Transparent reporting for unrecorded clinical variables (e.g. Blood Pressure)
  • Natural-language resilience across 12 differently worded query phrasings
  • Substring collision prevention (e.g. "regimen" vs "men", "discharge" vs "charge")

tests/test_model.py (2 tests)
  • Serialized model bundle integrity and encoder mappings
  • Inference pipeline output shapes and valid probability bounds [0.0, 1.0]

tests/test_preprocess.py (5 tests)
  • ICD-9 primary diagnosis categorization
  • Discharge disposition mapping (excluding expired/hospice)
  • Admission acuity and source cleaning
  • 1-row input vector feature alignment and transformation

======================== 28 passed in 2.4s ========================
```

---

## ⚠️ Clinical Boundaries & Limitations

1. **Observational & Retrospective:** Trained on historical encounter records from 1999–2008. Clinical standards and diabetes pharmacotherapy have evolved since data collection.
2. **Correlation vs. Causation:** Model feature importances and dataset statistics denote statistical associations within the dataset; they do not establish causal medical etiology.
3. **Target Scope:** Classifies binary readmission across the encounter period, rather than isolating 30-day early readmissions.
4. **Regulatory Status:** This system is an academic engineering portfolio prototype. It is **not** cleared by the FDA or international regulatory bodies as Software as a Medical Device (SaMD) and must not be used for autonomous clinical decision-making.

---

## 👩‍💻 Author & Contact

**Yashashree Borkar**  
*Computer Science & Engineering*  
- GitHub: [@YashashreeB-02](https://github.com/YashashreeB-02)  
- Project Repository: [Hospital-Patient-Readmission-Predictor](https://github.com/YashashreeB-02/Hospital-Patient-Readmission-Predictor)
