import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Dashboard Penilaian Atlet", layout="wide")

st.title("🏋️‍♂️ Aplikasi Penilaian Performa Atlet")

st.sidebar.header("Input Nilai Atlet")
jumlah_atlet = st.sidebar.number_input("Jumlah Atlet", 1, 10, 3)

# Input 5 variabel untuk setiap atlet
data = []
for i in range(int(jumlah_atlet)):
    st.sidebar.subheader(f"Atlet {i+1}")
    kecepatan = st.sidebar.number_input(f"Kecepatan Atlet {i+1} (m/s)", 0.0, 20.0, 9.0)
    daya_tahan = st.sidebar.number_input(f"Daya Tahan Atlet {i+1} (%)", 0.0, 100.0, 70.0)
    kekuatan = st.sidebar.number_input(f"Kekuatan Atlet {i+1} (kg)", 0.0, 200.0, 80.0)
    kelincahan = st.sidebar.number_input(f"Kelincahan Atlet {i+1}", 0.0, 10.0, 7.0)
    reaksi = st.sidebar.number_input(f"Waktu Reaksi Atlet {i+1} (detik)", 0.0, 5.0, 1.5)
    data.append({
        "Atlet": f"Atlet {i+1}",
        "Kecepatan": kecepatan,
        "Daya Tahan": daya_tahan,
        "Kekuatan": kekuatan,
        "Kelincahan": kelincahan,
        "Reaksi": reaksi
    })

df = pd.DataFrame(data)

# Algoritma penilaian sederhana
df["Skor Total"] = (
    0.3 * df["Kecepatan"] +
    0.25 * df["Daya Tahan"] +
    0.2 * df["Kekuatan"] +
    0.15 * df["Kelincahan"] +
    0.1 * (10 - df["Reaksi"])
)

# Tentukan atlet terbaik
best_athlete = df.loc[df["Skor Total"].idxmax()]

st.subheader("📊 Data Performa Atlet")
st.dataframe(df)

# Grafik perbandingan skor total
fig = px.bar(df, x="Atlet", y="Skor Total", color="Atlet",
             title="Perbandingan Skor Total Atlet", text_auto=True)
st.plotly_chart(fig, use_container_width=True)

# Radar Chart (visualisasi kekuatan tiap aspek)
import plotly.graph_objects as go

radar = go.Figure()

for _, row in df.iterrows():
    radar.add_trace(go.Scatterpolar(
        r=[row["Kecepatan"], row["Daya Tahan"], row["Kekuatan"], row["Kelincahan"], (10-row["Reaksi"])],
        theta=["Kecepatan", "Daya Tahan", "Kekuatan", "Kelincahan", "Reaksi (dibalik)"],
        fill='toself',
        name=row["Atlet"]
    ))

radar.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
    title="Radar Chart Performa Atlet",
    showlegend=True
)
st.plotly_chart(radar, use_container_width=True)

# Output atlet terbaik
st.success(f"🏆 Atlet dengan performa terbaik adalah **{best_athlete['Atlet']}** dengan skor {best_athlete['Skor Total']:.2f}")
