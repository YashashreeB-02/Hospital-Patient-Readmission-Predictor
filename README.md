# 🏥 Hospital Patient Readmission Predictor
### *Machine Learning Risk Stratification Engine & AI Clinical Copilot*

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o%20Mini-412991?logo=openai&logoColor=white)](https://openai.com/)
[![Tests](https://img.shields.io/badge/Tests-12%20Passed-10B981?logo=pytest&logoColor=white)](#-testing--quality-assurance)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An enterprise-ready clinical decision-support and workflow prototype built for diabetic inpatient discharge planning. The system combines regularized gradient-boosted decision trees (`HistGradientBoostingClassifier`) with an **AI Clinical Copilot** and an **AI Follow-up Draft Assistant** to deliver actionable risk stratification, conversational explanation, and post-discharge communication coordination.

---

## 📑 Table of Contents
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Dataset & Prediction Target](#-dataset--prediction-target)
- [Model Performance & Benchmarks](#-model-performance--benchmarks)
- [AI Clinical Copilot & Draft Generator](#-ai-clinical-copilot--draft-generator)
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
3. **💬 Talk to Your Data (AI Clinical Copilot):** Natural language conversational assistant powered by OpenAI (with conversational memory across turns). Healthcare users can ask follow-up questions about risk drivers, lab indicators, and discharge planning.
4. **✉️ Human-in-the-Loop Follow-up Draft Generator:** Automatically compiles a structured post-discharge outreach letter with patient-specific timing, placeholders for clinic identifiers, and a mandatory clinician verification checklist. *Strictly draft-only: never sends automatically.*
5. **💡 Dual Operating Mode (Live LLM + Offline Fallback):** Seamlessly operates with live OpenAI models (`gpt-4o-mini`, `gpt-4o`) when an API key is provided, or in a local rule-based educational preview mode when offline.
6. **🔒 Zero-Exposure Security Architecture:** Streamlit secrets isolation, `.env` fallbacks, session-only overrides, and strict `.gitignore` configurations ensure zero API keys are ever committed to version control.

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
│  • Metric Cards & Visual Gauge       │ │  • Multi-turn Follow-up Context    │
│  • Contributing Risk Factors         │ │  • AI Follow-up Draft Assistant    │
│  • Recommended Discharge Protocol    │ │  • [Live OpenAI / Local Mode]      │
└──────────────────────────────────────┘ └────────────────────────────────────┘
```

---

## 📊 Dataset & Prediction Target

### Dataset Overview
* **Source:** UCI Machine Learning Repository — Diabetes 130-US Hospitals (1999–2008).
* **Sample Size:** 101,766 encounters across 130 hospitals.
* **Eligible Cohort:** 99,343 encounters after clinical exclusion of in-hospital expired or hospice-transferred patients (clinical standard: expired patients cannot be readmitted).

### Verified Target Variable Definition
In standard medical literature, readmission is often loosely cited as "30-day readmission." However, the model is trained on:
$$	ext{Target } y = egin{cases} 1 & 	ext{if readmitted } (<30 	ext{ or } >30 	ext{ days}) \ 0 & 	ext{if not readmitted } (	ext{'NO'}) \end{cases}$$

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

## 💬 AI Clinical Copilot & Draft Generator

### 1. Talk to Your Data (Conversational AI)
* **Context Grounding:** Injects full patient encounter context (risk tier, probability, factors, admission/discharge details) into the prompt context.
* **Conversational Memory:** Preserves multi-turn conversation history (`st.session_state["chat_messages"]`), allowing users to ask natural follow-up questions:
  * *User:* "Why is this patient at High Risk?"
  * *Copilot:* "The patient is classified as High Risk (60.1%) primarily due to 2 prior inpatient admissions, extended length of stay (7 days), and active insulin dosage titration."
  * *Follow-up:* "What was the insulin adjustment?"
  * *Copilot:* "The insulin regimen was titrated ('Up'), indicating dosage increases to manage elevated glycemic levels."
* **Suggested Quick Prompts:** One-click chips for common clinical inquiries.

### 2. Follow-up Draft Generator
* **Editable Draft:** Formats a structured clinical outreach letter with risk-tailored visit timing, medication reminders, and placeholder tokens (`[Patient Name]`, `[Doctor Name]`, `[Clinic Phone]`).
* **Pre-Send Verification Checklist:** Includes clinical safety steps that care teams must verify prior to sending.
* **Human-in-the-Loop Guarantee:** The draft is presented in an editable text area with prominent warnings stating that **no message is ever sent automatically**.

---

## 🔒 Security & Secrets Management

Security best practices are enforced to ensure API keys and credentials are never leaked:
* **Streamlit Secrets:** Configured in `.streamlit/secrets.toml` (ignored by Git).
* **Git Protection:** `.gitignore` includes `.streamlit/secrets.toml`, `.env`, `*.env`, `.venv/`, `__pycache__/`, and backup files.
* **Template Provided:** `.streamlit/secrets.toml.example` provides onboarding instructions without exposing live credentials.
* **Session Override:** Evaluators without file access can enter an OpenAI key via the sidebar password input; keys remain in memory only and are never persisted to disk.

---

## 📂 Repository Structure

```text
Hospital-Patient-Readmission-Predictor/
├── .streamlit/
│   ├── secrets.toml.example  # Template for secrets configuration
│   └── secrets.toml          # Local secrets (strictly git-ignored)
├── tests/
│   ├── __init__.py           # Test package initialization
│   ├── test_preprocess.py    # Unit tests for preprocessing & ICD-9 mappings
│   ├── test_model.py         # Unit tests for model loading & inference shape
│   └── test_chat_assistant.py# Unit tests for copilot, draft generator & mocks
├── app.py                    # Main Streamlit clinical GUI application
├── chat_assistant.py         # AI Clinical Copilot & Draft Generator engine
├── preprocess.py             # Feature engineering, encoding & data preparation
├── train_model.py            # Model training & serialization script
├── diabetic_data.csv         # UCI 130-US Hospitals dataset (19 MB)
├── readmission_model.pkl     # Trained model bundle with feature encoders
├── requirements.txt          # Python dependencies
├── .gitignore                # Git ignore rules for secrets and virtualenvs
└── README.md                 # Comprehensive project documentation
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
*(If no API key is provided, the application runs seamlessly in local educational mode!)*

### 5. Launch the Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🧪 Testing & Quality Assurance

The test suite covers preprocessing transformations, model inference consistency, API key validation, local fallback logic, and mocked OpenAI interactions:

```bash
pytest -v
```

**Test Output:**
```text
tests/test_chat_assistant.py::test_is_valid_api_key PASSED          [  8%]
tests/test_chat_assistant.py::test_get_api_config_override PASSED   [ 16%]
tests/test_chat_assistant.py::test_local_fallback_answer PASSED     [ 25%]
tests/test_chat_assistant.py::test_local_fallback_draft PASSED      [ 33%]
tests/test_chat_assistant.py::test_openai_live_call PASSED          [ 41%]
tests/test_model.py::test_bundle_structure PASSED                   [ 50%]
tests/test_model.py::test_inference_pipeline PASSED                 [ 58%]
tests/test_preprocess.py::test_categorize_diagnosis PASSED          [ 66%]
tests/test_preprocess.py::test_map_discharge PASSED                 [ 75%]
tests/test_preprocess.py::test_map_admission PASSED                 [ 83%]
tests/test_preprocess.py::test_map_source PASSED                    [ 91%]
tests/test_preprocess.py::test_prepare_single_input PASSED          [100%]

======================== 12 passed in 2.5s ========================
```

---

## ⚠️ Clinical Boundaries & Limitations

1. **Observational & Retrospective:** Trained on historical encounter records from 1999–2008. Healthcare practices and diabetes pharmacotherapy have evolved since data collection.
2. **Correlation vs. Causation:** Model feature importances and risk flags denote statistical associations within the dataset; they do not establish causal medical etiology.
3. **Target Scope:** Classifies binary readmission across the encounter period, rather than isolating isolated 30-day early readmissions.
4. **Regulatory Status:** This system is an academic engineering portfolio prototype. It is **not** cleared by the FDA or international regulatory bodies as Software as a Medical Device (SaMD) and must not be used for autonomous clinical decision-making.

---

## 👩‍💻 Author & Contact

**Yashashree Borkar**  
*Computer Science & Engineering*  
- GitHub: [@YashashreeB-02](https://github.com/YashashreeB-02)  
- Project Repository: [Hospital-Patient-Readmission-Predictor](https://github.com/YashashreeB-02/Hospital-Patient-Readmission-Predictor)
