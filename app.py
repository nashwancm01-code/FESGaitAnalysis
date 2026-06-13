import streamlit as st
import pandas as pd
import math
import io
import matplotlib.pyplot as plt
import numpy as np

# --- 1. FUNGSI LPF MANUAL (WITH CACHE) ---
@st.cache_data
def apply_manual_lpf(data, dt, cutoff, order):
    if cutoff <= 0 or order < 1:
        return data
    
    fc_adj = cutoff / math.sqrt(2**(1.0 / order) - 1.0)
    tau = 1.0 / (2.0 * math.pi * fc_adj)
    alpha = dt / (tau + dt)
    
    y = list(data)
    for _ in range(order):
        y_new = []
        prev_y = y[0]
        y_new.append(prev_y)
        for i in range(1, len(y)):
            curr_y = alpha * y[i] + (1.0 - alpha) * prev_y
            y_new.append(curr_y)
            prev_y = curr_y
        y = y_new
    return y

# --- 2. FUNGSI LOAD & RECTIFY (WITH CACHE) ---
@st.cache_data
def load_and_rectify(file_bytes):
    column_names = [
        "time", "heel", "toe", "hip", "knee", "ankle", 
        "gluteus maximus", "biceps femoris short", "biceps femoris long", 
        "vastus medialis", "vastus lateralis", "rectus femoris", 
        "soleus", "gastrocnemius", "tibialis anterior"
    ]
    
    df = pd.read_csv(io.BytesIO(file_bytes), sep='\s+', header=None, names=column_names)
    
    dt = df['time'].iloc[1] - df['time'].iloc[0]
    if dt <= 0:
        dt = 0.001
        
    emg_columns = column_names[6:15]
    
    rect_dict = {}
    for col in emg_columns:
        rect_dict[col] = [val if val >= 0 else -val for val in df[col]]
        
    return df, dt, emg_columns, rect_dict

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Aplikasi Biomekanik & EMG", layout="wide")

st.title("Aplikasi Pemrosesan Data Biomekanik")

uploaded_file = st.file_uploader("Unggah file data (.txt)", type=["txt"])

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    df, dt, emg_columns, rect_dict = load_and_rectify(file_bytes)
    
    # MENU 1: Cuplikan Data
    st.markdown("---")
    st.subheader("📋 Cuplikan Data Asli (Menu 1)")
    st.dataframe(df.head())
    st.markdown("---")
    
    tab1, tab2 = st.tabs(["Grafik EMG & Aktivasi", "Tab Analisis Lainnya"])
    
    with tab1:
        st.header("Analisis Sinyal EMG & Aktivasi Otot (Menu 2)")
        
        # Kontrol Global untuk LPF
        col1, col2 = st.columns(2)
        with col1:
            cutoff_freq = st.slider("Cutoff Frequency LPF (Hz)", min_value=0.5, max_value=20.0, value=5.0, step=0.5)
        with col2:
            filter_order = st.slider("Orde Filter LPF", min_value=1, max_value=5, value=2, step=1)
            
        st.markdown("---")
        
        # =========================================================
        # SEKSI BARU: MAP AKTIVASI SEMUA OTOT (SESUAI SLIDE DOSEN)
        # =========================================================
        st.subheader("📊 Peta Aktivasi Semua Otot (Threshold 5% Max)")
        st.write("Grafik horizontal di bawah menunjukkan kapan setiap otot aktif (ON) secara bersamaan.")
        
        # Downsampling biar grafik enteng dan anti-lag / Page Unresponsive
        step = 10
        time_steps = df['time'][::step].tolist()
        
        # Siapkan canvas matplotlib
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Lakukan perulangan untuk memproses semua otot sekaligus
        for idx, muscle in enumerate(emg_columns):
            r_data = rect_dict[muscle]
            l_data = apply_manual_lpf(r_data, dt, cutoff_freq, filter_order)
            
            # Hitung otomatis threshold 5% dari Nilai Maksimum LPF otot ini
            max_lpf = max(l_data)
            auto_threshold = 0.05 * max_lpf
            
            # Ambil sampel data sesuai step downsampling
            l_data_sampled = l_data[::step]
            
            # Bikin list koordinat Y: isi dengan index otot jika ON, isi NaN jika OFF
            y_vals = []
            for val in l_data_sampled:
                if val >= auto_threshold:
                    y_vals.append(idx) # Taruh di baris ototnya
                else:
                    y_vals.append(np.nan) # Biarkan bolong/kosong
            
            # Plot baris horizontal tebal untuk otot ini
            ax.plot(time_steps, y_vals, linewidth=8, solid_capstyle='butt', color='#1f77b4')
            
        # Percantik grafik agar persis seperti di diktat dosen
        ax.set_yticks(range(len(emg_columns)))
        ax.set_yticklabels([m.title() for m in emg_columns], fontsize=10)
        ax.set_xlabel("Time (seconds)", fontsize=11)
        ax.set_ylabel("Muscles", fontsize=11)
        ax.set_title("Muscle Activation Profile Each Cycle", fontsize=12, fontweight='bold')
        ax.grid(axis='x', linestyle='--', alpha=0.5)
        ax.set_ylim(-0.5, len(emg_columns) - 0.5)
        
        # Tampilkan di Streamlit
        st.pyplot(fig)
        
        # =========================================================
        # SEKSI DETAIL: TAMPILAN PER INDIVIDU OTOT (UNTUK VALIDASI)
        # =========================================================
        st.markdown("---")
        st.subheader("🔍 Analisis Detail per Otot")
        selected_muscle = st.selectbox("Pilih satu otot untuk melihat proses filternya:", emg_columns)
        
        raw_data = df[selected_muscle].tolist()
        rectified_data = rect_dict[selected_muscle]
        lpf_data = apply_manual_lpf(rectified_data, dt, cutoff_freq, filter_order)
        
        # Tampilkan grafik detail pembersihan sinyal individu
        df_detail = pd.DataFrame({
            'time': df['time'][::step],
            'Raw EMG': raw_data[::step],
            'Rectified': rectified_data[::step],
            'LPF Envelope': lpf_data[::step]
        }).set_index('time')
        
        st.line_chart(df_detail, height=250)

    with tab2:
        st.header("Ruang Kosong untuk Analisis Lanjut")
        st.write("Tab ini disiapkan untuk pengerjaan data Sudut Sendi & Gait Phase (Menu 3 & 4).")
