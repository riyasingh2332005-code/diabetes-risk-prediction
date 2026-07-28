import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('cdc_brfss_diabetes_2023.csv')
print("Shape:", df.shape)
print(df.head())


print("\nMissing values:\n", df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())

df = df.drop_duplicates()
print("\nShape after cleaning:", df.shape)

print("\nTarget distribution:\n", df['Diabetes_binary'].value_counts())
print("\nTarget %:\n", df['Diabetes_binary'].value_counts(normalize=True) * 100)

df.to_csv('cleaned_data.csv', index=False)