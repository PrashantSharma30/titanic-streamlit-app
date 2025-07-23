import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load and preprocess dataset
@st.cache_data
def load_model_data():
    df = pd.read_csv('https://github.com/YBI-Foundation/Dataset/raw/main/Titanic.csv')
    df['age'] = df['age'].fillna(df['age'].mean())
    df['fare'] = df['fare'].fillna(df['fare'].mean())
    df.drop(['cabin', 'boat', 'body'], axis=1, inplace=True)
    df.dropna(inplace=True)

    Sex = pd.get_dummies(df['sex'])
    Embarked = pd.get_dummies(df['embarked'])
    df = pd.concat([df, Sex, Embarked], axis=1)
    df.drop(['sex', 'embarked'], axis=1, inplace=True)

    y = df['survived']
    X = df[['pclass', 'age', 'female', 'male', 'C', 'Q', 'S']]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=2529)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    return model

# Load model
model = load_model_data()

# Streamlit UI
st.title("🚢 Titanic Passenger Survival Prediction")

class_inp = st.selectbox("Select Passenger Class", options=[1, 2, 3], index=0)
age_inp = st.slider("Select Age of Passenger", min_value=1, max_value=100, value=25)
gender = st.radio("Select Gender", options=["Male", "Female"])
embarked = st.radio("Select Embarked Port", options=["Cherbourg (C)", "Queensville (Q)", "Southampton (S)"])

# Encode gender
mal_inp = 1 if gender == "Male" else 0
fem_inp = 1 - mal_inp

# Encode embarked
if embarked == "Cherbourg (C)":
    C_inp, Q_inp, S_inp = 1, 0, 0
elif embarked == "Queensville (Q)":
    C_inp, Q_inp, S_inp = 0, 1, 0
else:
    C_inp, Q_inp, S_inp = 0, 0, 1

user_input = {
    'pclass': class_inp,
    'age': age_inp,
    'female': fem_inp,
    'male': mal_inp,
    'C': C_inp,
    'Q': Q_inp,
    'S': S_inp
}

user_df = pd.DataFrame(user_input, index=[0])
st.write("### Details Entered:")
st.table(user_df)

if st.button("Predict Survival"):
    prediction = model.predict(user_df)[0]
    if prediction == 1:
        st.success("✅ The Passenger **Did Survive**.")
    else:
        st.error("❌ The Passenger **Did Not Survive**.")