# 🏥 Hospital Patient Readmission Predictor

## 📌 Overview
The **Hospital Patient Readmission Predictor** is a machine learning–based clinical decision support system designed to predict whether a diabetic patient is likely to be readmitted to the hospital within 30 days after discharge. 

The system leverages historical clinical encounter data from over 100,000 hospitalizations and provides risk assessments through an interactive, multi-tab **Streamlit GUI**. Healthcare professionals can view readmission probabilities, calibrated risk categories, key clinical risk drivers, and tailored discharge protocols.

---

## 🎯 Objectives
* Predict hospital readmission risk using multi-dimensional patient and encounter data.
* Assist healthcare teams in personalized post-discharge planning.
* Identify high-risk clinical factors (e.g., prior emergency visits, insulin adjustments, high comorbidity).
* Reduce avoidable hospital readmissions and healthcare costs.

---

## 🧠 Technologies Used
* **Programming Language:** Python 3.10+
* **Core ML Libraries:** Scikit-learn, Pandas, NumPy, Joblib
* **Algorithm:** Regularized Gradient Boosted Decision Trees (`HistGradientBoostingClassifier`)
* **GUI Framework:** Streamlit
* **Development Environment:** VS Code / Antigravity IDE

---

## 📊 Dataset & Feature Engineering
* **Source:** UCI Machine Learning Repository (Diabetes 130-US Hospitals Dataset)
* **Analyzed Encounters:** 101,766 (with 99,343 post-discharge eligible encounters)
* **Features Used:**
  * **Demographics:** Age group bracket
  * **Admission & Stay:** Length of stay (days), Admission type (Emergency, Urgent, Elective), Admission source
  * **Clinical Diagnoses:** Primary ICD-9 diagnosis categories (Circulatory, Respiratory, Diabetes, Digestive, Genitourinary, Musculoskeletal, Neoplasms, Injury, Other), Total number of diagnoses
  * **Procedures & Care:** Number of lab procedures, other medical/surgical procedures, medications prescribed
  * **Prior Healthcare Utilization:** Inpatient admissions, emergency department visits, outpatient clinic visits
  * **Diabetes Management & Labs:** Diabetes medication prescribed, medication dosage change status, insulin regimen status, HbA1c test result, glucose serum test result
  * **Engineered Signals:** Total prior visits, service density per hospital day, prior inpatient ratio

---

## 📈 Performance Benchmarks

| Metric | Initial Baseline Model (Random Forest) | Upgraded Model (Regularized Gradient Boosting) | Improvement |
| :--- | :---: | :---: | :---: |
| **Test Accuracy** | 55.67% | **63.54%** | **+7.87%** |
| **ROC-AUC Score** | 0.5720 | **0.6828** | **+0.1108** |
| **Class 1 Precision** | 52.08% | **62.88%** | **+10.80%** |
| **Class 1 Recall** | 50.48% | **55.22%** | **+4.74%** |
| **Class 1 F1-Score** | 0.5127 | **0.5880** | **+0.0753** |
| **Overfitting Gap** | ~35.0% (90.4% train vs 55.7% test) | **1.9%** (65.4% train vs 63.5% test) | **Overfitting Eliminated** |

---

## ⚙️ Project Structure
```text
Hospital_Readmission_Predictor/
│
├── preprocess.py          # Data ingestion, ICD-9 categorization, and encoding pipeline
├── train_model.py         # Model training, validation, and evaluation reporting
├── app.py                 # Interactive Streamlit clinical prediction interface
├── diabetic_data.csv      # Dataset (UCI 130-US Hospitals)
├── readmission_model.pkl  # Serialized trained model bundle and encoders
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
```

---

## 🚀 How to Run the Project

### 1. Clone the repository & install dependencies:
```bash
git clone https://github.com/YashashreeB-02/Hospital-Patient-Readmission-Predictor.git
cd Hospital-Patient-Readmission-Predictor
pip install -r requirements.txt
```

### 2. Train the Model:
```bash
python train_model.py
```
*Loads the dataset, executes feature engineering, prints full evaluation metrics, and serializes `readmission_model.pkl`.*

### 3. Launch the Streamlit GUI:
```bash
streamlit run app.py
```
*Open the provided local URL (typically `http://localhost:8501`) in your browser.*

---

## 🖥️ Application Features
* **Tabbed Patient Assessment Form:** Intuitive forms for Demographics, Diagnoses, Utilization, and Medications.
* **Calibrated Risk Stratification:** Provides predicted readmission probability (%) and assigns Low, Moderate, or High risk levels.
* **Dynamic Visual Risk Meter:** Visual progress gauge comparing the patient's probability against clinical baselines.
* **Clinical Factor Analysis:** Automatically flags high-risk triggers (e.g. repeated prior hospitalizations, multiple diagnoses, insulin adjustments).
* **Actionable Discharge Protocols:** Recommends specific clinical follow-up intervals, pharmacist reconciliation, and telehealth check-ins.

---

## 👩‍💻 Author
**Yashashree Borkar**  
B.Tech – Computer Science & Engineering
