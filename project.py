import pandas as pd
import numpy as np

from google.colab import files
uploaded = files.upload()
df = pd.read_csv('Customer Purchase Data.csv')

# Display first 5 rows
df.head()
# Shape of dataset
print("Shape:", df.shape)

# Column names
print("Columns:", df.columns.tolist())

# Dataset info
df.info()

# Statistical summary
df.describe()
# Missing values
print("Missing values:\n", df.isnull().sum())

# Duplicate rows
print("Duplicate rows:", df.duplicated().sum())
# Fill numeric missing values with mean
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# Fill categorical missing values with mode
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])
  import seaborn as sns
import matplotlib.pyplot as plt
sns.histplot(df['Purchase_Frequency'], kde=True)
plt.title("Purchase Frequency Distribution")
plt.xlabel("Purchase Frequency")
plt.show()
sns.scatterplot(x='Income', y='Purchase_Frequency', data=df)
plt.title("Income vs Purchase Frequency")
plt.show()
X = df.drop('Purchase_Frequency', axis=1)
y = df['Purchase_Frequency']


print("Features:\n", X.columns)
print("Target:\n Purchase_Frequency")
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)
from sklearn.metrics import mean_squared_error, r2_score

y_pred = model.predict(X_test)

print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))
new_customer = {
    'Age': 40,
    'Income': 52000,
    'Spending_Score': 9500,
    'Membership_Years': 8,
    'Last_Purchase_Amount': 4800
}

new_df = pd.DataFrame([new_customer])

new_scaled = scaler.transform(new_df)
prediction = model.predict(new_scaled)

print("🛒 Predicted Purchase Frequency:", round(prediction[0], 2))
!pip install gradio
import gradio as gr

def predict_purchase_frequency(age, income, spending_score, membership_years, last_purchase_amount):
    
    input_df = pd.DataFrame([{
        'Age': age,
        'Income': income,
        'Spending_Score': spending_score,
        'Membership_Years': membership_years,
        'Last_Purchase_Amount': last_purchase_amount
    }])
    
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    
    return round(prediction[0], 2)
  inputs = [
    gr.Number(label="Age"),
    gr.Number(label="Income"),
    gr.Number(label="Spending Score"),
    gr.Number(label="Membership Years"),
    gr.Number(label="Last Purchase Amount")
]

output = gr.Number(label="Predicted Purchase Frequency")

gr.Interface(
    fn=predict_purchase_frequency,
    inputs=inputs,
    outputs=output,
    title="🛒 Customer Purchase Frequency Predictor",
    description="Predict customer purchase frequency using machine learning"
).launch()
