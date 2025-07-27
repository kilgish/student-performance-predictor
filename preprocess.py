import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv('1data/student-por.csv')

# Create 'pass_fail' column
df['pass_fail'] = df['G3'].apply(lambda x: 1 if x >= 10 else 0)

# Encode binary categorical columns
le = LabelEncoder()
binary_cols = ['sex', 'address', 'famsize', 'Pstatus', 'schoolsup', 'famsup',
               'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic']
for col in binary_cols:
    df[col] = le.fit_transform(df[col])

# One-hot encode other categorical features
df = pd.get_dummies(df, columns=['school', 'Mjob', 'Fjob', 'reason', 'guardian'], drop_first=True)

# Save preprocessed data
df.to_csv('1data/cleaned_student_data.csv', index=False)

print("Preprocessing complete. Shape:", df.shape)

# Load your dataset
df = pd.read_csv('1data/student-por.csv')

# Map categorical values to numerical values
mapping_dict = {
    'sex': {'F': 0, 'M': 1},
    'address': {'U': 1, 'R': 0},
    'famsize': {'LE3': 0, 'GT3': 1},
    'Pstatus': {'T': 1, 'A': 0},
    'schoolsup': {'yes': 1, 'no': 0},
    'famsup': {'yes': 1, 'no': 0},
    'paid': {'yes': 1, 'no': 0},
    'activities': {'yes': 1, 'no': 0},
    'nursery': {'yes': 1, 'no': 0},
    'higher': {'yes': 1, 'no': 0},
    'internet': {'yes': 1, 'no': 0},
    'romantic': {'yes': 1, 'no': 0}
}

# Apply the mappings
df.replace(mapping_dict, inplace=True)

# Print to verify the transformation
print(df[['sex', 'address', 'famsize', 'Pstatus', 'schoolsup', 'famsup',
          'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic']].head())

# Optional: Save the updated DataFrame
df.to_csv('1data/updated_student_data.csv', index=False)
