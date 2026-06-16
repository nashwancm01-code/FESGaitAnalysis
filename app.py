import streamlit as st
import math
import matplotlib.pyplot as plt
import pandas as pd

# --- 1. FUNGSI LPF MANUAL ---
@st.cache_data
def apply_manual_lpf(data, dt, cutoff, order):
    if cutoff <= 0 or order < 1: return data
    fc_adj = cutoff / math.sqrt(2**(1.0 / order) - 1.0)
    tau = 1.0 / (2.0 * math.pi * fc_adj)
    alpha = dt / (tau + dt)
    y = list(data)
    for _ in range(order):
        y_new = [y[0]]
        for i in range(1, len(y)):
            curr_y = alpha * y[i] + (1.0 - alpha) * y_new[-1]
            y_new.append(curr_y)
        y = y_new
    return y

# --- 2. FUNGSI LOAD DATA YANG FLEKSIBEL ---
@st.cache_data
def load_and_process_data(file_bytes):
    column_names = [
        "time", "heel", "toe", "hip", "knee", "ankle", 
        "gluteus maximus", "biceps femoris short", "biceps femoris long", 
        "vastus medialis", "vastus lateralis", "rectus femoris", 
        "soleus", "gastrocnemius", "tibialis anterior"
    ]
    raw_text = file_bytes.decode('utf-8').splitlines()
    parsed_data = {col: [] for col in column_names}
    
    for line in raw_text:
        parts = line.split()
        if len(parts) >= len(column_names):
            try:
                for idx, col in enumerate(column_names):
                    parsed_data[col].append(float(parts[idx]))
            except ValueError: continue
            
    dt = (parsed_data["time"][1] - parsed_data["time"][0]) if len(parsed_data["time"]) > 1 else 0.001
    emg_cols = column_names[6:15]
    rect_dict = {col: [abs(x) for x in parsed_data[col]] for col in emg_cols}
    return parsed_data, dt, emg_cols, rect_dict

# --- 3. UI APLIKASI ---
st.set_page_config(page_title="Aplikasi Biomekanik", layout="wide")
st.title("Aplikasi Pemrosesan Data Biomekanik")

uploaded_file = st.file_uploader("Unggah file data (.txt)", type=["txt"])

if uploaded_file is not None:
    parsed_data, dt, emg_cols, rect_dict = load_and_process_data(uploaded_file.getvalue())
    
    # Menampilkan tabel data
    with st.expander("Lihat Data Mentah"):
        st.dataframe(pd.DataFrame(parsed_data).head(10))

    tab1, tab2 = st.tabs(["Grafik EMG & Aktivasi", "Gait Parameters & Kinematics"])
    
    with tab1:
        st.header("Analisis Sinyal EMG")
        cutoff = st.slider("Cutoff Frequency (Hz)", 0.5, 20.0, 5.0)
        muscle = st.selectbox("Pilih Otot:", emg_cols)
        
        # Plotting Tab 1
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(parsed_data['time'][::10], parsed_data[muscle][::10], color='gray', alpha=0.5, label="Raw")
        lpf_data = apply_manual_lpf(rect_dict[muscle], dt, cutoff, 2)
        ax.plot(parsed_data['time'][::10], lpf_data[::10], color='red', label="Filtered")
        ax.legend()
        st.pyplot(fig)

    with tab2:
        st.header("Analisis Gait & Kinematika")
        # Plotting Heel/Toe
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.plot(parsed_data['time'], parsed_data['heel'], label='Heel')
        ax2.plot(parsed_data['time'], parsed_data['toe'], label='Toe')
        ax2.legend()
        st.pyplot(fig2)
        
        # Plotting Sendi
        fig3, ax3 = plt.subplots(1, 3, figsize=(15, 4))
        ax3[0].plot(parsed_data['time'], parsed_data['hip']); ax3[0].set_title("Hip")
        ax3[1].plot(parsed_data['time'], parsed_data['knee']); ax3[1].set_title("Knee")
        ax3[2].plot(parsed_data['time'], parsed_data['ankle']); ax3[2].set_title("Ankle")
        st.pyplot(fig3)
