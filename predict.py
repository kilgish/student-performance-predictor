import pandas as pd
import joblib

# Load models
clf = joblib.load('3models/classification_model.pkl')
reg = joblib.load('3models/regression_model.pkl')

# Example student input
student_input = {
    'sex': 1,
    'age': 17,
    'address': 1,
    'famsize': 1,
    'Pstatus': 1,
    'Medu': 2,
    'Fedu': 2,
    'traveltime': 1,
    'studytime': 2,
    'failures': 0,
    'schoolsup': 0,
    'famsup': 1,
    'paid': 0,
    'activities': 1,
    'nursery': 1,
    'higher': 1,
    'internet': 1,
    'romantic': 0,
    'famrel': 4,
    'freetime': 3,
    'goout': 3,
    'Dalc': 1,
    'Walc': 1,
    'health': 4,
    'absences': 2,
    'G1': 12,
    'G2': 13,
    # Include one-hot columns:
    'school_MS': 0,
    'Mjob_health': 0,
    'Mjob_other': 1,
    'Mjob_services': 0,
    'Mjob_teacher': 0,
    'Fjob_health': 0,
    'Fjob_other': 1,
    'Fjob_services': 0,
    'Fjob_teacher': 0,
    'reason_home': 0,
    'reason_other': 0,
    'reason_reputation': 1,
    'guardian_mother': 1,
    'guardian_other': 0
}


input_df = pd.DataFrame([student_input])

# Predict
predicted_grade = reg.predict(input_df)[0]
predicted_class = clf.predict(input_df)[0]

print("Predicted Final Grade (G3):", round(predicted_grade, 2))
print("Prediction: PASS" if predicted_class == 1 else "Prediction: FAIL")
