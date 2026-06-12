import streamlit as st
import pandas as pd
import math
import io

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
    
    # ==========================================
    # MENU 1: MENAMPILKAN TABEL DATA ASLI
    # ==========================================
    st.markdown("---")
    st.subheader("📋 Cuplikan Data Asli (Menu 1)")
    st.write("Tabel di bawah menampilkan 5 baris pertama dari data mentah untuk memastikan file terbaca dengan benar.")
    st.dataframe(df.head())
    st.markdown("---")
    
    # ==========================================
    # MEMBUAT TAB UNTUK MENU SELANJUTNYA
    # ==========================================
    tab1, tab2 = st.tabs(["Grafik EMG & Aktivasi", "Tab Lainnya"])
    
    with tab1:
        st.header("Analisis Sinyal EMG & Aktivasi Otot")
        
        # Dropdown Pilih Otot agar browser ringan
        selected_muscle = st.selectbox("Pilih Otot yang Ingin Dianalisis:", emg_columns)
        
        # Kolom Parameter untuk Slider LPF dan Threshold
        col1, col2, col3 = st.columns(3)
        with col1:
            cutoff_freq = st.slider("Cutoff Frequency LPF (Hz)", min_value=0.5, max_value=20.0, value=5.0, step=0.5)
        with col2:
            filter_order = st.slider("Orde Filter LPF", min_value=1, max_value=5, value=2, step=1)
        with col3:
            threshold_val = st.slider("Threshold Aktivasi Otot", min_value=0.00, max_value=2.00, value=0.20, step=0.01)
        
        st.markdown("---")
        
        # Proses Pengolahan Data
        rectified_data = rect_dict[selected_muscle]
        lpf_data = apply_manual_lpf(rectified_data, dt, cutoff_freq, filter_order)
        
        # Logika Thresholding Manual (ON=1, OFF=0)
        activation_data = [1 if val >= threshold_val else 0 for val in lpf_data]
        threshold_line = [threshold_val] * len(lpf_data)
        
        # Trik downsampling visual (titik ke-10) biar anti-lag
        step = 10 
        
        # 1. Plot Grafik Utama (Raw Rectified vs LPF Envelope)
        df_main_plot = pd.DataFrame({
            'time': df['time'][::step],
            'Rectified Signal': rectified_data[::step],
            'LPF (Envelope)': lpf_data[::step],
            'Threshold Line': threshold_line[::step]
        }).set_index('time')
        
        st.subheader(f"1. Pemrosesan Sinyal EMG: {selected_muscle.title()}")
        st.line_chart(df_main_plot, height=350)
        
        # 2. Plot Grafik Aktivasi Kotak-Kotak (Hasil Thresholding)
        df_activation_plot = pd.DataFrame({
            'time': df['time'][::step],
            'Muscle Activation (ON/OFF)': activation_data[::step]
        }).set_index('time')
        
        st.subheader(f"2. Grafik Aktivasi Otot (Thresholding Result)")
        st.write("Nilai 1 berarti otot Aktif (ON), nilai 0 berarti otot Istirahat (OFF).")
        st.line_chart(df_activation_plot, height=200)
            
    with tab2:
        st.header("Ruang Kosong untuk Analisis Lanjut")
        st.write("Tab ini bisa kamu kembangkan nanti untuk data Sudut & Gait (Menu 3 & 4).")    
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
    
    tab1, tab2 = st.tabs(["Grafik EMG", "Tab Lainnya"])
    
    with tab1:
        st.header("Sinyal EMG")
        
        # --- DROPDOWN UNTUK MEMILIH OTOT ---
        selected_muscle = st.selectbox("Pilih Otot yang Ingin Ditampilkan:", emg_columns)
        
        # --- SLIDER LPF ---
        col1, col2 = st.columns(2)
        with col1:
            cutoff_freq = st.slider("Cutoff Frequency (Hz)", min_value=0.5, max_value=20.0, value=5.0, step=0.5)
        with col2:
            filter_order = st.slider("Orde Filter", min_value=1, max_value=5, value=2, step=1)
        
        st.markdown("---")
        
        # Ambil data rektifikasi khusus otot yang dipilih
        rectified_data = rect_dict[selected_muscle]
        
        # Aplikasikan LPF
        lpf_data = apply_manual_lpf(rectified_data, dt, cutoff_freq, filter_order)
        
        # --- TRICK ANTI-LAG: DOWNSAMPLING PLOT ---
        # Mengambil setiap titik ke-10 (step=10) agar browser tidak berat
        # Data aslinya tidak rusak, kita hanya mengurangi titik visualnya
        step = 10 
        
        df_single_plot = pd.DataFrame({
            'time': df['time'][::step],
            'Rectified': rectified_data[::step],
            'LPF (Envelope)': lpf_data[::step]
        }).set_index('time')
        
        st.subheader(f"Grafik: {selected_muscle.title()}")
        st.line_chart(df_single_plot, height=400)
            
    with tab2:
        st.header("Ruang Kosong untuk Analisis Lanjut")
        st.write("Tab ini bisa kamu kembangkan nanti.")
