import pandas as pd
from sklearn.preprocessing import MiniMaxScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

file_path = r"C:\Users\vamsi\PycharmProjects\PythonProject1\placement_dataset.csv"

df = pd.read_csv(file_path)

feature = "Internships"
print("Original Statistics:")
print(df[feature].describe())

Q1 = df[feature].quantile(0.25)
Q3 = df[feature].quantile(0.75)
IQR = Q3 - Q1

lower_fence = Q1 - 1.5 * IQR
upper_fence = Q3 + 1.5 * IQR


print("\nQ1 =", Q1)
print("\nQ3 =", Q3)
print("\nIQR =", IQR)
print("\nlower_fence =", lower_fence)
print("\nupper_fence =", upper_fence)

outliers = df[
    (df[feature] < lower_fence) |
    (df[feature] > upper_fence)
]

print("\nNumber of Outliers =:", len(outliers))

df["CodingTest_Clipped"] = df[feature].clip(
    lower=lower_fence,
    upper=upper_fence
)

print("\nMinimum BEFORE clipping:")
print(df[feature].min())

print("\nMinimum AFTER clipping:")
print(df["CodingTest_Clipped"].min())

scaler = MinMaxScaler()

df["CodingTest_Clipped"] = scaler.fit_transform(
    df[["CodingTest_Clipped"]]
)




train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["Internships"]
)