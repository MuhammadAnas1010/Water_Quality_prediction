import joblib
import streamlit as st
import numpy as np

# Load model
model = joblib.load('water_model.joblib')


# Initialize session state for tracking input interactions
if 'input_counter' not in st.session_state:
    st.session_state.input_counter = 0
    st.session_state.fields_filled = [False] * 9  # track individual field state

st.header('Predict water quality Using :rainbow[_Machine Learning_]', divider=True)
st.markdown("Using **Random Forest Classifier**")
st.subheader("✅ Safe Ranges for Water Quality Parameters")

st.markdown("""
- **pH**: 6.0 – 8.5
- **Hardness**: 130 – 240 mg/L
- **Solids (TDS)**: 19000 – 30000 mg/L
- **Chloramines**: 5.5 – 9.5 mg/L
- **Sulfate**: 285 – 366 mg/L
- **Conductivity**: 350 – 550 µS/cm
- **Organic Carbon (TOC)**: 12.0 – 15.5 mg/L
- **Trihalomethanes (THMs)**: 48 – 75 µg/L
- **Turbidity**: 3.5 – 4.7 NTU
""")
st.subheader("Fill in all the fields:")

col1, col2, col3 = st.columns(3)

def update_counter(index):
    if not st.session_state.fields_filled[index]:
        st.session_state.fields_filled[index] = True
        st.session_state.input_counter += 1

with col1:
    ph = st.number_input("pH (0-14)", min_value=0.0, max_value=14.0, key='ph', on_change=update_counter, args=(0,))
    hardness = st.number_input("Hardness (47-323)", min_value=47.0, max_value=323.0, key='hardness', on_change=update_counter, args=(1,))
    solids = st.number_input("Solids (320-61227)", min_value=320.0, max_value=62000.0, key='solids', on_change=update_counter, args=(2,))

with col2:
    chloramines = st.number_input("Chloramines (0.3-13)", min_value=0.3, max_value=13.0, key='chloramines', on_change=update_counter, args=(3,))
    sulfate = st.number_input("Sulfate (129-481)", min_value=129.0, max_value=481.0, key='sulfate', on_change=update_counter, args=(4,))
    conductivity = st.number_input("Conductivity (181-753)", min_value=181.0, max_value=753.0, key='conductivity', on_change=update_counter, args=(5,))

with col3:
    organic_carbon = st.number_input("Organic Carbon (2.2-28)", min_value=2.2, max_value=28.0, key='organic_carbon', on_change=update_counter, args=(6,))
    trihalomethanes = st.number_input("Trihalomethanes (0.7-124)", min_value=0.7, max_value=124.0, key='trihalomethanes', on_change=update_counter, args=(7,))
    turbidity = st.number_input("Turbidity (1.4-6.7)", min_value=1.4, max_value=6.7, key='turbidity', on_change=update_counter, args=(8,))

# Button to classify only if all inputs were interacted with
if st.button("Classify"):
    if st.session_state.input_counter < 9:
        st.warning("Please interact with all 9 input fields before classifying.")
    else:
        x = np.array([[st.session_state.ph, st.session_state.hardness, st.session_state.solids,
                       st.session_state.chloramines, st.session_state.sulfate, st.session_state.conductivity,
                       st.session_state.organic_carbon, st.session_state.trihalomethanes, st.session_state.turbidity]])

        pred = model.predict(x)[0]
        prob = model.predict_proba(x)[0]

        result_text = "✅ Safe to Drink (Potable)" if pred == 1 else "❌ Not Safe to Drink (Non-Potable)"
        st.subheader(f"Prediction: **{result_text}**")
        st.write(f"Probability (Potable = 1): **{prob[1]:.2f}**")
        st.write(f"Probability (Non-Potable = 0): **{prob[0]:.2f}**")
