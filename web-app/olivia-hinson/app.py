import streamlit as st 
import numpy as np 
import pandas as pd
import joblib 
import os


######### initial setup 
st.set_page_config(page_title='CCDP@streamlit', layout='wide')
st.title(':orange[Credit Card Default Prediction]')

@st.cache_resource
def load_model(): 
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'credit_model.pkl')   
    model = joblib.load(model_path)
    
    return model

model = load_model()

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
    value = st.number_input(f"Enter value for {month}", key=feature)
    user_input.append(value)

# Convert input to NumPy array
input_data = np.array(user_input).reshape(1, -1)

# Validate input
if input_data.shape[1] == len(feature_names):
    # Predict button
    if st.button("Predict"):
        # Generate prediction
        prediction = model.predict(input_data)
        st.success(f"Prediction: {prediction[0]}")
else:
    st.error("Please ensure all input fields are filled.")

# Add documentation
st.sidebar.header("App Information")
st.sidebar.info(
    """
    This app allows you to make predictions using a pre-trained machine learning model.
    
    **Instructions**:
    1. Enter values for all features.
    2. Click "Predict" to get the result.

    **Note**: Input data must match the expected format.
    """
)

######### About the app ###############
'''
markdown_about_msg = """
        
        ## Welcome to the Alzheimer's Prediction Project
        
        This web application empowers you to analyze critical patient data that could revolutionize early Alzheimer's detection. 
        It lets you play around with key parameters to distinguish between patients with and without Alzheimer's. Currently the app
        is using a model data set (source below)
        
        Data source :  KAGGLE : [link](https://www.kaggle.com/datasets/rabieelkharoua/alzheimers-disease-dataset) to the data set

        Field meanings: 
          - :blue[MMSE]: Mini-Mental State Examination score, ranging from 0 to 30. Lower scores indicate cognitive impairment.
          - :blue[ADL]: Activities of Daily Living score, ranging from 0 to 10. Lower scores indicate greater impairment.
          - :blue[Functional Assessment]: Functional assessment score, ranging from 0 to 10. Lower scores indicate greater impairment.

    """
    
############ SIDEBAR introduction to project ##########################
with st.sidebar:
    st.image(os.path.join(os.path.dirname(__file__), 'image', 'alzheimer_image.jpg'))
    st.markdown(markdown_about_msg)

col1,col2 = st.columns(2,gap="medium")

with col1: 
    mmse = st.number_input('Mini-Mental State Examination (MMSE) Rating', min_value=0, max_value=30, value='min')
    adl = st.number_input('Activities of Daily Living (ADL) Rating', min_value=0, max_value=10, value='min')
    functAsses = st.number_input('Functional Assessment Rating', min_value=0, max_value=10, value='min')

with col2: 
    memCompl = st.selectbox('Is the patient experiencing any :orange[Memory Complaints]?', ('No', 'Yes'))
    behaviorProbs = st.selectbox('Is the patient experiencing any :orange[Behavioral Problems]?', ('No', 'Yes'))

if (mmse != None and adl != None and functAsses != None and
    memCompl!= None and  behaviorProbs != None):
    
    if st.button('Generate Alzheimer Prediction'): 
        input_data = {
            'MMSE': mmse, 
            'ADL': adl,
            'FunctionalAssessment': functAsses, 
            'MemoryComplaints': memCompl, 
            'BehavioralProblems': behaviorProbs, 
        }
            
        # Make prediction
        result = alzheimersPredict(input_data)
            
        # Display result
        if (result == 1): 
            st.error('Alzheimer Diagnosis :thumbsdown:')
        else: 
            st.success('This patient doesn\'t appear to be diagnosed with Alzheimers :thumbsup:')
else: 
    alzheimerPredictButton = st.button('Generate Alzheimer Prediction', disabled=True)
'''         









