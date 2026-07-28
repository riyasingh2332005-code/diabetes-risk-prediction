import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('cleaned_data.csv')

# 1. Target distribution
df['Diabetes_binary'].value_counts().plot(kind='bar')
plt.title('Diabetes Distribution')
plt.xlabel('Diabetes_binary (0=No, 1=Yes)')
plt.ylabel('Count')
plt.savefig('plot_target_distribution.png')
plt.show()

# 2. BMI vs Diabetes
sns.boxplot(x='Diabetes_binary', y='BMI', data=df)
plt.title('BMI by Diabetes Status')
plt.savefig('plot_bmi_boxplot.png')
plt.show()

# 3. Age vs Diabetes rate
age_diabetes = df.groupby('Age')['Diabetes_binary'].mean() * 100
age_diabetes.plot(kind='line', marker='o')
plt.title('Diabetes Rate by Age Group')
plt.xlabel('Age Group (1=youngest, 13=oldest)')
plt.ylabel('Diabetes Rate (%)')
plt.grid(True)
plt.savefig('plot_age_diabetes.png')
plt.show()

# 4. GenHlth vs Diabetes rate
genhlth_diabetes = df.groupby('GenHlth')['Diabetes_binary'].mean() * 100
genhlth_diabetes.plot(kind='bar', color='steelblue')
plt.title('Diabetes Rate by General Health Rating')
plt.xlabel('GenHlth (1=Excellent, 5=Poor)')
plt.ylabel('Diabetes Rate (%)')
plt.savefig('plot_genhlth_diabetes.png')
plt.show()