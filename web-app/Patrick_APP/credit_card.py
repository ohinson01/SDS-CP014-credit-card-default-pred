import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Page configuration
st.set_page_config(layout="wide")

@st.cache_resource
def load_model_and_scaler():
    """
    Loads the pre-trained model and scaler.
    
    Returns:
    model: The pre-trained model.
    scaler: The scaler for feature normalization.
    """
    model = joblib.load("web-app/Patrick_APP/gb_credit_model.pk1")
    scaler = joblib.load("web-app/Patrick_APP/gb_credit_scaler.pk1")
    return model, scaler

# Load the model and scaler
model, scaler = load_model_and_scaler()

def predict_default(input_data):
    """
    Predicts whether a credit card payment will default.
    
    Parameters:
    input_data (list): List of feature values entered by the user.
    
    Returns:
    prediction: Prediction class (0 or 1).
    probabilities: Probabilities of each class.
    """
    # Scale the input data
    input_data_scaled = scaler.transform([input_data])
    prediction = model.predict(input_data_scaled)[0]
    probabilities = model.predict_proba(input_data_scaled)[0]
    return prediction, probabilities

# Main app function
def main():
    st.title("💳 Credit Card Default Prediction")
    
    st.sidebar.title("User Guide")
    st.sidebar.markdown("""
    This app predicts whether a credit card holder is likely to default on their payment based on user inputs.
    Enter the required details on the main panel and click **Predict** to get the result.
    """)
    
    st.sidebar.markdown("### Feature Descriptions")
    st.sidebar.markdown("""
    - **limit_bal**: Credit limit (e.g., 10000).
    - **sex**: Gender (1 = Male, 2 = Female).
    - **education**: (1 = Graduate School, 2 = University, 3 = High School, 4 = Others).
    - **marriage**: Marital Status (1 = Married, 2 = Single, 3 = Others).
    - **age**: Age in years.
    - **repayment statuses**: Payment delays (-1 = pay duly, 1 = 1-month delay, 2 = 2-month delay, 3 = 3 months or more).
    - **bill amounts**: Monthly bill statements.
    - **payment amounts**: Monthly payment amounts.
    """)

    st.header("Enter Customer Details")
    
    # User inputs for features
    limit_bal = st.number_input("Credit Limit (limit_bal)", min_value=0, step=1000)
    sex = st.selectbox("Gender (sex)", [1, 2], format_func=lambda x: "Male" if x == 1 else "Female")
    education = st.selectbox("Education Level (education)", [1, 2, 3, 4], 
                              format_func=lambda x: {1: "Graduate School", 2: "University", 
                                                     3: "High School", 4: "Others"}[x])
    marriage = st.selectbox("Marital Status (marriage)", [1, 2, 3], 
                             format_func=lambda x: {1: "Married", 2: "Single", 3: "Others"}[x])
    age = st.number_input("Age", min_value=18, step=1)
    
    repayment_features = []
    for month in ['April', 'May', 'June', 'July', 'August', 'September']:
        repayment_features.append(st.selectbox(f"{month} Repayment Status", [-1, 1, 2, 3],
                                               format_func=lambda x: { 
                                                   -1: "Pay Duly", 1: "1-Month Delay",
                                                   2: "2-Month Delay", 3: "3+ Months Delay"
                                               }[x]))
    
    bill_features = []
    for month in ['April', 'May', 'June', 'July', 'August', 'September']:
        bill_features.append(st.number_input(f"{month} Bill Amount", min_value=0, step=500))
    
    payment_features = []
    for month in ['April', 'May', 'June', 'July', 'August', 'September']:
        payment_features.append(st.number_input(f"{month} Payment Amount", min_value=0, step=500))
    
    # Combine all inputs into a feature vector
    input_features = [limit_bal, sex, education, marriage, age] + repayment_features + bill_features + payment_features
    
    # Predict button
    if st.button("Predict"):
        prediction, probabilities = predict_default(input_features)
        prediction_class = "Default Likely" if prediction == 1 else "No Default Likely"
        confidence = probabilities[prediction]
        result_color = "red" if prediction == 1 else "green"
        
        st.markdown(
            f"<span style='color:{result_color}; font-weight:bold;'>Predicted Class by Gradient Boosting Classifier: {prediction_class} (Confidence: {confidence:.2%})</span>",
            unsafe_allow_html=True,
        )
        
        # Plot the probabilities as a bar chart
        prob_df = pd.DataFrame({
            "Class": ["No Default", "Default"],
            "Probability": probabilities
        })
        st.bar_chart(data=prob_df.set_index("Class"), use_container_width=True)
    
    st.sidebar.markdown("""
    #### Note:
    This is a demonstration of a machine learning model's predictive capabilities. Predictions should not be used as the sole basis for decision-making.
    """)

# Run the app
if __name__ == "__main__":
    main()
