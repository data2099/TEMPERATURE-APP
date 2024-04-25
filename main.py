import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle

# Load the trained XGBoost model
model = pickle.load(open("model.pkl", "rb"))

def load_data(data):
    data = pd.read_csv(data)
    return data

def show_histograms(data):
    st.subheader('Histograms for Numerical Columns')
    for MONTH in data.select_dtypes(include='number').columns:
        fig, ax = plt.subplots()
        sns.histplot(data[MONTH], ax=ax, kde=True)
        st.pyplot(fig)

def show_line_plots(data):
    st.subheader('Line Plots')
    for MONTHLY_AVG_TEMP in data.select_dtypes(include='number').columns:
        fig, ax = plt.subplots()
        sns.lineplot(data=data, x=data.index, y=MONTHLY_AVG_TEMP, ax=ax, label=MONTHLY_AVG_TEMP)
        st.pyplot(fig)

def show_scatter_plots(data):
    st.subheader('Scatter Plots')
    num_cols = data.select_dtypes(include='number').columns
    for i in range(len(num_cols)):
        for j in range(i+1, len(num_cols)):
            fig, ax = plt.subplots()
            sns.scatterplot(data=data, x=num_cols[i], y=num_cols[j], ax=ax)
            st.pyplot(fig)

def show_box_plots(data):
    st.subheader('Box Plots')
    for DAY in data.select_dtypes(include='number').columns:
        fig, ax = plt.subplots()
        sns.boxplot(data=data[DAY], ax=ax)
        st.pyplot(fig)

def predict(month, day, year):
    # Perform prediction
    features = np.array([[month, day, year]])
    features = np.array([[month, day, year, 0, 0, 0, 0, 0]])
    prediction = model.predict(features)[0]
    prediction = float(prediction)
    return prediction

def temp_prediction():
    st.title('Temperature Prediction App')
    
    # Collect user input
    month = st.number_input('Enter the month (1-12)', max_value=12)
    day = st.number_input('Enter the day (1-31)', max_value=31)
    year = st.number_input('Enter the year', max_value=2100)

    if st.button('Predict'):
        prediction = predict(month, day, year)
        st.success(f'The prediction is: {prediction}')

def eda_app():
    st.title('Exploratory Data Analysis (EDA) App')
    uploaded_file = st.file_uploader('Upload CSV file', type=['csv'])
    
    if uploaded_file is not None:
        # Load data
        data = load_data(uploaded_file)
        st.write('**Data Preview:**')
        st.write(data.head())
        show_histograms(data)
        show_line_plots(data)
        show_scatter_plots(data)
        show_box_plots(data)

def main():
    st.sidebar.title('Navigation')
    app_selection = st.sidebar.radio('Go to:', ('Temperature Prediction', 'Exploratory Data Analysis'))
    
    if app_selection == 'Temperature Prediction':
        temp_prediction()
    elif app_selection == 'Exploratory Data Analysis':
        eda_app()

if __name__ == '__main__':
    main()
