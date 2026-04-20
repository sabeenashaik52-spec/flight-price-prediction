import streamlit as st
import pickle
import pandas as pd


model = pickle.load(open('rf_model.pkl', 'rb'))
columns = pickle.load(open('columns.pkl', 'rb'))


st.set_page_config(page_title="Flight Price Predictor", page_icon="✈️", layout="centered")


st.title("✈️ Indian Flight Price Prediction")
st.markdown("### Predict flight prices using Machine Learning")
st.info("Prediction based on historical data using Random Forest model")



airline = st.selectbox("Airline", 
    ['Air India', 'Vistara', 'Indigo', 'SpiceJet', 'GO_FIRST', 'AirAsia'])

source = st.selectbox("Source City", 
    ['Delhi', 'Mumbai', 'Bangalore', 'Kolkata', 'Hyderabad', 'Chennai'])

destination = st.selectbox("Destination City", 
    ['Delhi', 'Mumbai', 'Bangalore', 'Kolkata', 'Hyderabad', 'Chennai'])

departure_time = st.selectbox("Departure Time", 
    ['Morning', 'Afternoon', 'Evening', 'Night', 'Early_Morning', 'Late_Night'])

arrival_time = st.selectbox("Arrival Time", 
    ['Morning', 'Afternoon', 'Evening', 'Night', 'Early_Morning', 'Late_Night'])

stops = st.selectbox("Stops", 
    ['zero', 'one', 'two_or_more'])

flight_class = st.selectbox("Class", 
    ['Economy', 'Business'])

duration = st.number_input("Duration (hours)", min_value=0.5, step=0.1)
days_left = st.number_input("Days Left", min_value=1, step=1)



if st.button("Predict Price "):

    # Input validation
    if duration <= 0:
        st.warning(" Please enter a valid flight duration")
    else:
        # Create input dictionary
        input_dict = {col: 0 for col in columns}

        # Numerical values
        input_dict['duration'] = duration
        input_dict['days_left'] = days_left

        # One-hot encoding mapping
        if f'airline_{airline}' in columns:
            input_dict[f'airline_{airline}'] = 1

        if f'source_city_{source}' in columns:
            input_dict[f'source_city_{source}'] = 1

        if f'destination_city_{destination}' in columns:
            input_dict[f'destination_city_{destination}'] = 1

        if f'departure_time_{departure_time}' in columns:
            input_dict[f'departure_time_{departure_time}'] = 1

        if f'arrival_time_{arrival_time}' in columns:
            input_dict[f'arrival_time_{arrival_time}'] = 1

        if f'stops_{stops}' in columns:
            input_dict[f'stops_{stops}'] = 1

        if f'class_{flight_class}' in columns:
            input_dict[f'class_{flight_class}'] = 1

        # Convert to DataFrame
        input_df = pd.DataFrame([input_dict])

        # Prediction
        prediction = model.predict(input_df)[0]

        # Display result
        st.success(f" Estimated Flight Price: ₹ {int(prediction):,}")

# Footer
st.markdown("---")
st.markdown("Developed using Machine Learning  | Random Forest Model")