import streamlit as st
import pandas as pd
import plotly.express as px
from informer_model import load_model_and_predict

st.set_page_config(layout="wide")
st.title("📈 Prediksi Emisi Karbon Global dengan Informer")

uploaded_file = st.file_uploader("Unggah Dataset Excel", type=["xlsx"])

if uploaded_file is not None:
    with st.spinner("📊 Memproses data dan memuat model..."):
        try:
            df_all = pd.read_excel(uploaded_file, sheet_name=None)
            df_blue = df_all.get("BLUE")

            if df_blue is None:
                st.error("Sheet 'BLUE' tidak ditemukan dalam file.")
            else:
                st.subheader("Data Historis - Emisi Karbon Global")
                df_blue.rename(columns={df_blue.columns[0]: 'Year'}, inplace=True)
                df_global = df_blue[['Year', 'Global']].dropna()

                fig = px.line(df_global, x='Year', y='Global', markers=True,
                              title="Total Emisi Karbon Global (1850–2023)",
                              labels={"Global": "Emisi (MtCO₂)", "Year": "Tahun"})
                st.plotly_chart(fig, use_container_width=True)

                # Jalankan prediksi
                df_pred = load_model_and_predict(df_global)
                df_pred['Type'] = 'Prediksi'
                df_global['Type'] = 'Historis'
                df_all = pd.concat([df_global, df_pred])

                st.subheader("Prediksi 50 Tahun ke Depan")
                fig_pred = px.line(df_all, x='Year', y='Global', color='Type', markers=True,
                                   title="Prediksi Emisi Karbon Global (1850–2073)",
                                   labels={"Global": "Emisi (MtCO₂)", "Year": "Tahun"})
                st.plotly_chart(fig_pred, use_container_width=True)
        except Exception as e:
            st.exception(e)
else:
    st.info("Silakan unggah file Excel yang berisi data emisi karbon dengan sheet 'BLUE'.")
