import numpy as np
import pickle
import streamlit as st



loaded_model = pickle.load(open('./diabetes_model.sav', 'rb'))

def diabetes_prediction(input_data):

    i_d_as_numpy_array = np.asarray(input_data)

    i_d_reshaped = i_d_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(i_d_reshaped)
    print(prediction)

    if (prediction[0] == 0):
        return('## :blue[The person is not diabetic]')
    else:
        return('## :red[The person is diabetic]')
    





def main():

    st.title("Diabetes Prediction App")
    st.image("diabetes2.jpg", width=700)
    st.header("Enter Your Health Information:")



    # Getting input
    col1, col2 = st.columns(2)
    with col1:
        Pregnancies = st.text_input("Number of Pregnancies", placeholder="Enter value")
        Glucose = st.text_input("Glucose Level", placeholder="Enter value")
        BloodPressure = st.text_input("Blood Pressure value", placeholder="Enter value")
        SkinThickness = st.text_input("Skin Thickness value", placeholder="Enter value")
    with col2:
        Insulin = st.text_input("Insulin Level", placeholder="Enter value")
        BMI = st.text_input("BMI value", placeholder="Enter value")
        DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function value", placeholder="Enter value")
        Age = st.text_input("Age of the Person", placeholder="Enter value")

    # Code for Prediction
    diagnosis = ''
	
	
    # Creating a button for Prediction
    if st.button("Get Diabetes Test Result", help="Click to get your diabetes test result"):
        diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])

    # Displaying the result
    st.success(diagnosis)

    



if __name__ == "__main__":
    main()
    
    
    




# streamlit run ./diabetes_prediction_web.py
