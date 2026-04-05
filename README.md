🏥 Hospital Patient Readmission Predictor

📌 Overview

The Hospital Patient Readmission Predictor is a machine learning–based system designed to predict whether a patient is likely to be readmitted to the hospital after discharge. The project uses historical healthcare data and provides predictions through an interactive Streamlit-based GUI, making it easy for users to analyze patient risk in real time.



🎯 Objective

* Predict hospital readmission risk using patient data
* Assist healthcare professionals in decision-making
* Reduce unnecessary readmissions and improve patient care



🧠 Technologies Used

* Programming Language: Python
* Libraries: Pandas, NumPy, Scikit-learn, Joblib
* GUI Framework: Streamlit
* IDE: Visual Studio Code



📊 Dataset

* Source: UCI Machine Learning Repository
* Records: 101,766 patient entries
* Features Used:

  * Age
  * Time in hospital
  * Number of lab procedures
  * Number of medications
  * Number of inpatient visits
  * Readmission status (target)



⚙️ Project Structure


Hospital_Readmission_Predictor/
│
├── preprocess.py
├── train_model.py
├── app.py
├── diabetic_data.csv
├── readmission_model.pkl
└── requirements.txt



🚀 How to Run the Project

1. Clone or download the project folder
2. Open the folder in VS Code
3. Open terminal and install dependencies:
   pip install -r requirements.txt
   
4. Train the model:
   python train_model.py
  
5. Run the GUI:
   streamlit run app.py

6. Open the provided local URL in your browser
7. Enter patient details and click Predict



🖥️ Features

* User-friendly GUI for input and output
* Real-time prediction using trained ML model
* Accurate classification of readmission risk
* Supports multiple predictions without restart



📈 Model Details

* Algorithm Used: Random Forest Classifier
* Type: Supervised Learning
* Evaluation Metrics: Accuracy, Precision, Recall, F1-score



⚠️ Limitations

* Depends on dataset quality
* Does not use real-time hospital data
* Limited feature set



🔮 Future Enhancements

* Integration with real-time hospital databases
* Web/cloud deployment
* Improved model accuracy with advanced algorithms
* Mobile application interface



📄 License

This project is developed for academic purposes as part of a capstone project.



👩‍💻 Author

Yashashree Borkar
B.Tech – Computer Science & Engineering



⭐ Acknowledgement

This project uses publicly available healthcare datasets and open-source machine learning libraries.


