import streamlit as st 
import numpy as np 
import pandas as pd
import joblib 
import os

######### initial setup 

st.set_page_config(page_title='CCDP@streamlit', layout='wide')
st.title(':orange[Credit Card Default Prediction]')
payment_status = ['Paid On Time', 'One Month Delay', 'Two Month Delay', 
                  'Three Month Delay', 'Four Month Delay', 'Five Month Delay',
                  'Six Month Delay', 'Seven Month Delay', 'Eight Month Delay']

@st.cache_resource
def load_model(): 
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'credit_model.pkl')   
    model = joblib.load(model_path)
    
    return model

model = load_model()

def creditDefaultPredict(input_data): 
    X = pd.DataFrame([input_data])
    
    # Convert 'Yes'/'No' to 1/0
    X['pay_1'] = X['MemoryComplaints'].map({'Paid On Time': -1, 'One Month Delay': 1, 'Two Month Delay': 2, 
                                            'Three Month Delay': 3, 'Four Month Delay': 4, 'Five Month Delay': 5,
                                            'Six Month Delay': 6, 'Seven Month Delay': 7, 'Eight Month Delay': 8})
    X['pay_2'] = X['BehavioralProblems'].map({'Paid On Time': -1, 'One Month Delay': 1, 'Two Month Delay': 2, 
                                              'Three Month Delay': 3, 'Four Month Delay': 4, 'Five Month Delay': 5,
                                              'Six Month Delay': 6, 'Seven Month Delay': 7, 'Eight Month Delay': 8})
    X['pay_3'] = X['BehavioralProblems'].map({'Paid On Time': -1, 'One Month Delay': 1, 'Two Month Delay': 2, 
                                              'Three Month Delay': 3, 'Four Month Delay': 4, 'Five Month Delay': 5,
                                              'Six Month Delay': 6, 'Seven Month Delay': 7, 'Eight Month Delay': 8})
    X['pay_4'] = X['BehavioralProblems'].map({'Paid On Time': -1, 'One Month Delay': 1, 'Two Month Delay': 2, 
                                              'Three Month Delay': 3, 'Four Month Delay': 4, 'Five Month Delay': 5,
                                              'Six Month Delay': 6, 'Seven Month Delay': 7, 'Eight Month Delay': 8})
    X['pay_5'] = X['BehavioralProblems'].map({'Paid On Time': -1, 'One Month Delay': 1, 'Two Month Delay': 2, 
                                              'Three Month Delay': 3, 'Four Month Delay': 4, 'Five Month Delay': 5,
                                              'Six Month Delay': 6, 'Seven Month Delay': 7, 'Eight Month Delay': 8})
    X['pay_6'] = X['BehavioralProblems'].map({'Paid On Time': -1, 'One Month Delay': 1, 'Two Month Delay': 2, 
                                              'Three Month Delay': 3, 'Four Month Delay': 4, 'Five Month Delay': 5,
                                              'Six Month Delay': 6, 'Seven Month Delay': 7, 'Eight Month Delay': 8})
    # Grab first prediction of model
    prediction = model.predict(X)[0]
    
    return prediction

######### About the app ###############

about_msg = """
        
        This app allows you to predict whether you should default to 
        using a credit card based on your past payment history. 
        
        Currently, the app is uisng a model dataset (source below)

        **Instructions**:
        1. Enter your past payment history for the past 6 months
        2. Click "Predict" to get the result.

        **Note**: Input data must match the expected format.
        
        **Data Source**
        
         UC Machine Learning Repository (https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)
    """

############ SIDEBAR introduction to project ##########################

with st.sidebar: 
    st.header("Welcome to the Credit Card Default Prediction App!")
    st.info(about_msg)

############ SETTING the parameters ##########################

# Define the input features
feature_names = {
            'pay_1': 'September', 
            'pay_2': 'August', 
            'pay_3': 'July', 
            'pay_4': 'June', 
            'pay_5': 'May', 
            'pay_6': 'April'
}

# Collect user input
st.header("Input Features")
user_input = []
for feature, month in feature_names.items():
    #value = st.number_input(f"Payment Status for {month}", key=feature)
    value = st.selectbox(f"Payment Status for {month}", key=feature, options=payment_status)
    user_input.append(value)

# Convert input to NumPy array
input_data = np.array(user_input).reshape(1,1,6)

# Validate input
if input_data.shape == (1,1,6):
    # Predict button
    if st.button("Predict"):
        # Generate prediction
        prediction = creditDefaultPredict(input_data)
        st.success(f"Prediction: {prediction[0]}")
else:
    st.error("Please ensure all input fields are filled.")    









