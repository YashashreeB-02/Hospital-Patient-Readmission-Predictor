import pandas as pd

def load_and_preprocess():
    df = pd.read_csv("diabetic_data.csv")

    df = df[['age', 'time_in_hospital', 'num_lab_procedures',
             'num_medications', 'number_inpatient', 'readmitted']]

    df['readmitted'] = df['readmitted'].apply(
        lambda x: 1 if x in ['<30', '>30'] else 0
    )

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    df['age'] = le.fit_transform(df['age'])

    X = df.drop('readmitted', axis=1)
    y = df['readmitted']

    return X, y
