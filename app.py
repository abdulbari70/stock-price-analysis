import streamlit as st
import pandas as pd
import pickle

# Load data and models
companies = pd.read_csv("dataset-names.csv")
unique_symbols = companies['stock_symbol'].unique()
models = {}

for symbol in unique_symbols:
    with open(f'models/linear_regression/{symbol}.pkl', 'rb') as file:
        models[symbol] = pickle.load(file=file)

# Streamlit app
def main():
    st.title('Stock Price Forecaster')

    # Sidebar with input fields
    st.sidebar.header('Input Data')
    stock_symbol = st.sidebar.selectbox('Select Stock Symbol', unique_symbols)
    stock_open = st.sidebar.number_input('Open Price')
    stock_high = st.sidebar.number_input('High Price')
    stock_low = st.sidebar.number_input('Low Price')
    stock_volume = st.sidebar.number_input('Volume')

    if st.sidebar.button('Predict'):
        try:
            model = models[stock_symbol]
            prediction = model.predict([[stock_open, stock_high, stock_low, stock_volume]])
            st.success(f'Predicted Close Price: {prediction[0]}')
        except KeyError:
            st.error(f'Error: Model for {stock_symbol} not found.')
        except Exception as e:
            st.error(f'Error occurred: {str(e)}')

if __name__ == '__main__':
    main()
