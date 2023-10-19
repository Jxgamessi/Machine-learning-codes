import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

train_path = "C:/Users/Lenovo/Desktop/train.csv"
test_path = "C:/Users/Lenovo/Desktop/test.csv"

train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

st.title("Titanic Survival Prediction")

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch']
X = train_data[features]
y = train_data['Survived']

st.subheader("Age Distribution by Survival")
sns.set_style("whitegrid")
fig, ax = plt.subplots(figsize=(10, 5))
sns.histplot(train_data[train_data['Survived'] == 0]['Age'], kde=True, label='Did Not Survive', color='red', ax=ax)
sns.histplot(train_data[train_data['Survived'] == 1]['Age'], kde=True, label='Survived', color='blue', ax=ax)
ax.set_title('Age Distribution by Survival')
ax.set_xlabel('Age')
ax.set_ylabel('Count')
ax.legend()
st.pyplot(fig)

X = pd.get_dummies(X, columns=['Sex'])
X_test = test_data[features]
X_test = pd.get_dummies(X_test, columns=['Sex'])

X['Age'].fillna(X['Age'].mean(), inplace=True)
X_test['Age'].fillna(X_test['Age'].mean(), inplace=True)

scaler = StandardScaler()
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train, y_train)

test_predictions = model.predict(X_test)

color_map = {0: "red", 1: "green"}
colored_predictions = [f"**{'Yes' if pred == 1 else 'No'}**" for pred in test_predictions]

submission = pd.DataFrame({
    "PassengerId": test_data["PassengerId"],
    "Survived": colored_predictions
})

st.subheader("Survival Predictions")
st.dataframe(submission, width=600, height=300)
