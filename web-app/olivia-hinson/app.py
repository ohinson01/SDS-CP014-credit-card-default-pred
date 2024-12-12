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
all_fields_filled = False 

@st.cache_resource
def load_model(): 
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'credit_model.pkl')   
    model = joblib.load(model_path)
    
    return model

model = load_model()

def creditDefaultPredict(input_data): 
    X = pd.DataFrame([input_data])
    
    # Convert values to proper format
    X['pay_1'] = X['pay_1'].map({'Paid On Time': -1, 'One Month Delay': 1, 'Two Month Delay': 2, 
                                 'Three Month Delay': 3, 'Four Month Delay': 4, 'Five Month Delay': 5,
                                 'Six Month Delay': 6, 'Seven Month Delay': 7, 'Eight Month Delay': 8})
    X['pay_2'] = X['pay_2'].map({'Paid On Time': -1, 'One Month Delay': 1, 'Two Month Delay': 2, 
                                 'Three Month Delay': 3, 'Four Month Delay': 4, 'Five Month Delay': 5,
                                 'Six Month Delay': 6, 'Seven Month Delay': 7, 'Eight Month Delay': 8})
    X['pay_3'] = X['pay_3'].map({'Paid On Time': -1, 'One Month Delay': 1, 'Two Month Delay': 2, 
                                 'Three Month Delay': 3, 'Four Month Delay': 4, 'Five Month Delay': 5,
                                 'Six Month Delay': 6, 'Seven Month Delay': 7, 'Eight Month Delay': 8})
    X['pay_4'] = X['pay_4'].map({'Paid On Time': -1, 'One Month Delay': 1, 'Two Month Delay': 2, 
                                 'Three Month Delay': 3, 'Four Month Delay': 4, 'Five Month Delay': 5,
                                 'Six Month Delay': 6, 'Seven Month Delay': 7, 'Eight Month Delay': 8})
    X['pay_5'] = X['pay_5'].map({'Paid On Time': -1, 'One Month Delay': 1, 'Two Month Delay': 2, 
                                 'Three Month Delay': 3, 'Four Month Delay': 4, 'Five Month Delay': 5,
                                 'Six Month Delay': 6, 'Seven Month Delay': 7, 'Eight Month Delay': 8})
    X['pay_6'] = X['pay_6'].map({'Paid On Time': -1, 'One Month Delay': 1, 'Two Month Delay': 2, 
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
input_data = {}
for feature, month in feature_names.items():
    #value = st.number_input(f"Payment Status for {month}", key=feature)
    value = st.selectbox(f"Payment Status for {month}", key=feature, options=payment_status, index=None)
    input_data.update({feature: value})
    
if (input_data.get('pay_1') != None and input_data.get('pay_2') != None and 
    input_data.get('pay_3') != None and input_data.get('pay_4') != None and 
    input_data.get('pay_5') != None and input_data.get('pay_6') != None): 
    all_fields_filled = True

# Validate input
if all_fields_filled:
    # Predict button
    if st.button("Predict"):
        # Generate prediction
        prediction = creditDefaultPredict(input_data)
        
        if (prediction == 1): 
            st.success('You should use a credit card :thumbsup:')
        else: 
            st.error('Don\'t use a credit card :thumbsdown:')
else:
    st.error("Please ensure all input fields are filled.")    









