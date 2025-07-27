import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
import joblib

# Load cleaned data
df = pd.read_csv('1data/cleaned_student_data.csv')

# -------------------- Classification Model --------------------
X_class = df.drop(['pass_fail', 'G3'], axis=1)
y_class = df['pass_fail']

Xc_train, Xc_test, yc_train, yc_test = train_test_split(X_class, y_class, test_size=0.2, random_state=42)

clf = LogisticRegression(max_iter=1000)
clf.fit(Xc_train, yc_train)
joblib.dump(clf, '3models/classification_model.pkl')
print("Classification model saved.")

# -------------------- Regression Model --------------------
X_reg = df.drop(['pass_fail', 'G3'], axis=1)
y_reg = df['G3']

Xr_train, Xr_test, yr_train, yr_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

reg = LinearRegression()
reg.fit(Xr_train, yr_train)
joblib.dump(reg, '3models/regression_model.pkl')
print("Regression model saved.")
