import streamlit as st
import pandas as pd
import lightgbm as lgb
import json

from preprocess_input import preprocess_input_data

@st.cache_resource
def load_model(path='model_lgb.txt'):
    model = lgb.Booster(model_file=path)
    return model

def load_mappings(path='preprocessing_config.json'):
    with open(path, 'r') as f:
        data = json.load(f)
    return data['mappings']

def main():
    st.title("📊 Customer Churn Prediction")

    uploaded_file = st.file_uploader("Upload a CSV file with customer data", type=['csv'])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        if 'customerID' not in df.columns:
            st.error("The file must contain a 'customerID' column.")
            return
        
        customer_ids = df['customerID']

        st.success("File uploaded successfully!")
        st.write("🔍 Preview:", df.head())

        mappings = load_mappings()
        df_proc = preprocess_input_data(df, mappings)

        if 'Churn' in df_proc.columns:
            df_proc.drop('Churn', axis=1, inplace=True)

        df_proc = df_proc.drop(columns=['customerID'], errors='ignore')

        model = load_model()

        y_pred_proba = model.predict(df_proc)
        churn_percent = (y_pred_proba * 100).round(2)

        result_df = pd.DataFrame({
            'customerID': customer_ids,
            'ChurnProbability (%)': churn_percent
        })

        result_df = result_df.sort_values(by='ChurnProbability (%)', ascending=False)

        top_n = st.slider("How many top customers to show?", 5, 100, 10)
        st.subheader(f"🔝 Top {top_n} customers with highest churn risk:")
        st.write(result_df.head(top_n))

        csv = result_df.to_csv(index=False).encode('utf-8')
        st.download_button("💾 Download results as CSV", data=csv, file_name="churn_predictions.csv")

if __name__ == '__main__':
    main()
