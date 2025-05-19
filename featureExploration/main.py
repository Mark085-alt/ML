import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Load dataset
data = data = pd.read_csv(r"D:\College Project\Combined\featureExploration\data\dataset.csv")

# 1. Basic Info
print("📊 Dataset Head:")
print(data.head())

print("\n🧱 Dataset Info:")
print(data.info())

# 2. Check for missing values
print("\n❓ Missing Values:")
print(data.isnull().sum())

# 3. Statistical summary
print("\n📈 Statistical Summary:")
print(data.describe(include='all'))

# 4. Encode categorical features if any
le = LabelEncoder()
if 'Gender' in data.columns:
    data['Gender_encoded'] = le.fit_transform(data['Gender'])

# 5. Correlation Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(data.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()

# 6. Distribution plots
for col in data.select_dtypes(include=['int64', 'float64']).columns:
    plt.figure()
    sns.histplot(data[col], kde=True)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.show()
