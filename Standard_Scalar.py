import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split


file_path = r"C:\Users\vamsi\PycharmProjects\PythonProject1\placement_dataset.csv"
df = pd.read_csv(file_path)
#print(df)

num_cols =[
    "CGPA",
    "AttendancePercent",
    "AptitudeTestScore",
    "CodingTestScore",
    "Internships"
]
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["Internships"]
)

print("\nTraining Data before scaling:")
print(train_df[num_cols].head())

print("\nTesting Data before scaling:")
print(test_df[num_cols].head())

scaler = StandardScaler()
train_df[num_cols] = scaler.fit_transform(train_df[num_cols])

test_df[num_cols] = scaler.transform(test_df[num_cols])
print("\nScaled training data:")
print(train_df[num_cols].head())

print("\nScaled testing data:")
print(test_df[num_cols].head())