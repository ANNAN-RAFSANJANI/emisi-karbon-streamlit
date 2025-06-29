import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from informer_model import load_model_and_predict

st.set_page_config(layout="wide")
st.title("🌍 Prediksi Emisi Karbon Global dengan Informer & Visualisasi Dunia")

uploaded_file = st.file_uploader("Unggah Dataset Excel", type=["xlsx"])

SHEETS = ["BLUE", "H&C2023", "OSCAR", "LUCE"]
EXCLUDE_COUNTRIES = ['OTHER', 'DISPUTED', 'EU27', 'Global']

def world_map_animation(df_long):
    years = sorted(df_long['Year'].unique())
    frames = []

    for year in years:
        yearly_data = df_long[df_long['Year'] == year]
        map_data = yearly_data[~yearly_data['Country'].isin(EXCLUDE_COUNTRIES)]
        stats_data = yearly_data[~yearly_data['Country'].isin(['Global'])]
        global_row = yearly_data[yearly_data['Country'] == 'Global']

        top10 = stats_data.nlargest(10, 'Emissions')
        text_lines = [f"<b>10 Negara Tertinggi di tahun {year}:</b>"] + \
                     [f"{row['Country']}: {row['Emissions']:.2f}" for _, row in top10.iterrows()]

        if not global_row.empty:
            text_lines += ["", "<b>Total Global Emissions:</b>", f"{global_row.iloc[0]['Emissions']:.2f}"]

        choromap = go.Choropleth(
            locations=map_data['Country'],
            locationmode='country names',
            z=map_data['Emissions'],
            colorscale='YlOrRd',
            zmin=0,
            zmax=df_long[~df_long['Country'].isin(EXCLUDE_COUNTRIES)]['Emissions'].max(),
            colorbar_title="Emisi",
            hovertemplate="%{location}<br>Emisi: %{z:.2f}<extra></extra>"
        )

        frames.append(go.Frame(
            data=[choromap],
            name=str(year),
            layout=go.Layout(annotations=[dict(
                text="<br>".join(text_lines),
                x=0.01, y=0.98, xref='paper', yref='paper',
                showarrow=False,
                align='left',
                font=dict(size=12),
                bgcolor='rgba(255,255,255,0.7)',
                bordercolor='black',
                borderwidth=1
            )])
        ))

    fig = go.Figure(
        data=frames[0].data,
        frames=frames,
        layout=go.Layout(
            title='🌐 Peta Dunia Emisi Karbon - Animasikan Berdasarkan Tahun',
            width=1400, height=600,
            geo=dict(showframe=False, showcoastlines=True, projection_type='natural earth'),
            annotations=frames[0].layout.annotations,
            updatemenus=[{
                "type": "buttons",
                "buttons": [
                    {"label": "Play", "method": "animate", "args": [None, {"frame": {"duration": 500}, "fromcurrent": True}]},
                    {"label": "Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0}, "mode": "immediate"}]}
                ],
                "direction": "left", "pad": {"r": 10, "t": 0}, "showactive": False,
                "x": 0.05, "xanchor": "left", "y": 0.1, "yanchor": "bottom"
            }],
            sliders=[{
                "active": 0,
                "pad": {"t": 50},
                "steps": [dict(label=str(year), method="animate", args=[[str(year)], {"mode": "immediate"}]) for year in years]
            }]
        )
    )
    return fig

if uploaded_file:
    with st.spinner("📊 Memproses data..."):
        try:
            df_all_sheets = pd.read_excel(uploaded_file, sheet_name=None)

            sheet_choice = st.selectbox("Pilih Sheet", SHEETS)
            df_sheet = df_all_sheets.get(sheet_choice)

            if df_sheet is None:
                st.error(f"Sheet '{sheet_choice}' tidak ditemukan.")
            else:
                df_sheet.rename(columns={df_sheet.columns[0]: 'Year'}, inplace=True)

                # Tampilkan data historis global jika kolom "Global" ada
                if 'Global' in df_sheet.columns:
                    df_global = df_sheet[['Year', 'Global']].dropna()
                    df_global['Type'] = 'Historis'

                    st.subheader("📉 Data Historis Global")
                    fig = px.line(df_global, x='Year', y='Global', markers=True,
                                  title=f"Total Emisi Karbon Global ({df_global['Year'].min()}–{df_global['Year'].max()})",
                                  labels={"Global": "Emisi (MtCO₂)", "Year": "Tahun"})
                    st.plotly_chart(fig, use_container_width=True)

                    # Prediksi hanya untuk BLUE
                    if sheet_choice == 'BLUE':
                        df_pred = load_model_and_predict(df_global)
                        df_pred['Type'] = 'Prediksi'
                        df_all = pd.concat([df_global, df_pred])

                        st.subheader("📈 Prediksi Emisi 50 Tahun ke Depan (BLUE)")
                        fig_pred = px.line(df_all, x='Year', y='Global', color='Type', markers=True,
                                           title="Prediksi Emisi Karbon Global (1850–2073)",
                                           labels={"Global": "Emisi (MtCO₂)", "Year": "Tahun"})
                        st.plotly_chart(fig_pred, use_container_width=True)

                # Siapkan dan tampilkan visualisasi peta dunia
                df_long = df_sheet.melt(id_vars='Year', var_name='Country', value_name='Emissions')
                st.subheader("🗺️ Peta Dunia Emisi Karbon per Negara")
                fig_map = world_map_animation(df_long)
                st.plotly_chart(fig_map, use_container_width=True)

        except Exception as e:
            st.exception(e)
else:
    st.info("Silakan unggah file Excel yang berisi sheet: BLUE, H&C2023, OSCAR, atau LUCE.")
