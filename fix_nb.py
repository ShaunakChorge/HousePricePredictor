import json

with open('House_Price_Assessment.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Truncate at cell 14 (which is the first Cell 8)
nb['cells'] = nb['cells'][:14]

new_md = {
 'cell_type': 'markdown',
 'id': 'md-streamlit',
 'metadata': {},
 'source': [
  '---\n',
  '## Cell 8 — The Streamlit Dashboard UI\n',
  '\n',
  'This final cell writes the Streamlit application code to `app.py` and immediately launches the web server.\n',
  '\n',
  '> **Note:** Running this cell will block the notebook execution as the Streamlit server runs continuously. A new browser tab will automatically open with your interactive dashboard. To stop the server, click the **Stop (Interrupt Kernel)** button at the top of Jupyter.'
 ]
}

app_code = '''import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the saved model and scaler
@st.cache_resource
def load_assets():
    model = joblib.load('house_price_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_assets()

# Load dataset to compute medians for background features
@st.cache_data
def load_background_medians():
    df = pd.read_csv('house_prices.csv')
    return {
        'TotalBsmtSF': float(df['TotalBsmtSF'].median()),
        'FullBath': float(df['FullBath'].median()),
        'LotArea': float(df['LotArea'].median()),
        'YearRemodAdd': float(df['YearRemodAdd'].median())
    }

background_medians = load_background_medians()

st.title("🏡 Real Estate Price Predictor")
st.write("Adjust the primary features to see the estimated house price in real-time.")

# Sidebar for user inputs
st.sidebar.header("Input House Features")
overall_qual = st.sidebar.slider("Overall Quality (1-10)", 1, 10, 6)
gr_liv_area = st.sidebar.number_input("Total Living Area (sq ft)", min_value=500, max_value=10000, value=2000)
garage_cars = st.sidebar.slider("Garage Capacity (Cars)", 0, 5, 2)
year_built = st.sidebar.slider("Year Built", 1872, 2010, 1990)

if st.button("Predict Price"):
    # Combine the 4 active UI inputs and the 4 background medians in the exact order the model expects
    input_data = pd.DataFrame([{
        'OverallQual': overall_qual,
        'GrLivArea': gr_liv_area,
        'GarageCars': garage_cars,
        'YearBuilt': year_built,
        'TotalBsmtSF': background_medians['TotalBsmtSF'],
        'FullBath': background_medians['FullBath'],
        'LotArea': background_medians['LotArea'],
        'YearRemodAdd': background_medians['YearRemodAdd']
    }])
    
    # Scale the input data using the trained scaler
    input_data_scaled = scaler.transform(input_data)
    
    # Predict the price
    price = model.predict(input_data_scaled)[0]
    
    st.success(f"### Estimated Value: ${price:,.2f}")
    st.balloons()
'''

new_code = {
 'cell_type': 'code',
 'execution_count': None,
 'id': 'code-streamlit',
 'metadata': {},
 'outputs': [],
 'source': [
  'streamlit_code = """\n',
  app_code + '"""\n',
  '\n',
  'with open("app.py", "w", encoding="utf-8") as f:\n',
  '    f.write(streamlit_code)\n',
  '\n',
  'print("✅ Saved app.py. Launching Streamlit server...")\n',
  'print("⏳ A new browser tab should open automatically. To stop, interrupt the Jupyter kernel.")\n',
  '!streamlit run app.py\n'
 ]
}

nb['cells'].extend([new_md, new_code])

with open('House_Price_Assessment.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('Notebook completely overwritten with exact Cell 8 requirements.')
