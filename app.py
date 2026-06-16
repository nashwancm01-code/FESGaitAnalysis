import streamlit as st
import math
import io
import matplotlib.pyplot as plt
import pandas as pd  # Hanya digunakan di layer paling akhir untuk merender komponen UI tabel

# --- 1. FUNGSI LPF MANUAL (PURE PYTHON) ---
@st.cache_data
def apply_manual_lpf(data, dt, cutoff, order):
    if cutoff <= 0 or order < 1:
        return data
    
    # Menghitung koefisien filter secara manual berdasarkan rumus cutoff adj
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

# --- 2. FUNGSI PARSING & RECTIFY MANUAL (TANPA READ_CSV) ---
@st.cache_data
def load_and_rectify_manual(file_bytes):
    column_names = [
        "time", "heel", "toe", "hip", "knee", "ankle", 
        "gluteus maximus", "biceps femoris short", "biceps femoris long", 
        "vastus medialis", "vastus lateralis", "rectus femoris", 
        "soleus", "gastrocnemius", "tibialis anterior"
    ]
    
    raw_text = file_bytes.decode('utf-8').splitlines()
    parsed_data = {col: [] for col in column_names}
    
    for line in raw_text:
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) == len(column_names):
            try:
                for idx, col in enumerate(column_names):
                    parsed_data[col].append(float(parts[idx]))
            except ValueError:
                continue
                
    if len(parsed_data["time"]) > 1:
        dt = parsed_data["time"][1] - parsed_data["time"][0]
    else:
        dt = 0.001
    if dt <= 0:
        dt = 0.001
        
    emg_columns = column_names[6:15]
    
    rect_dict = {}
    for col in emg_columns:
        rectified_list = []
        for val in parsed_data[col]:
            if val < 0:
                rectified_list.append(-val)
            else:
                rectified_list.append(val)
        rect_dict[col] = rectified_list
        
    return parsed_data, dt, emg_columns, rect_dict

# --- 3. KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Aplikasi Biomekanik & EMG", layout="wide")
st.title("Aplikasi Pemrosesan Data Biomekanik")

uploaded_file = st.file_uploader("Unggah file data (.txt)", type=["txt"])

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    parsed_data, dt, emg_columns, rect_dict = load_and_rectify_manual(file_bytes)
    
    # ---------------------------------------------------------
    # OUTPUT 1: CUPLIKAN TABEL DATA MENTAH (DI LUAR TAB - PALING ATAS)
    # ---------------------------------------------------------
    st.markdown("---")
    st.subheader("📋 Cuplikan Data Asli")
    df_display = pd.DataFrame(parsed_data)
    st.dataframe(df_display.head())
    st.markdown("---")
    
    # =========================================================
    # INSENTIF STRUKTUR TABS KEMBALI DI DEKLEASIKAN
    # =========================================================
    tab1, tab2 = st.tabs(["Grafik EMG & Aktivasi", "Gait Parameters & Kinematics"])
    
    with tab1:
        st.header("Analisis Sinyal EMG & Aktivasi Otot")
        
        # PANEL KONTROL PARAMETER & SELEKSI OTOT (Masuk ke dalam Tab 1)
        ctrl_col1, ctrl_col2, ctrl_col3 = st.columns(3)
        with ctrl_col1:
            cutoff_freq = st.slider("Cutoff Frequency LPF (Hz)", min_value=0.5, max_value=20.0, value=5.0, step=0.5)
        with ctrl_col2:
            filter_order = st.slider("Orde Filter LPF", min_value=1, max_value=5, value=2, step=1)
        with ctrl_col3:
            selected_muscle = st.selectbox("Pilih Otot untuk Visualisasi Detail:", emg_columns)
            
        st.markdown("---")
        
        # Downsampling data menggunakan slicing [::step] agar grafik anti-lag
        step = 10
        time_steps = parsed_data['time'][::step]
        
        # Menyiapkan data spesifik otot yang dipilih user
        raw_emg_selected = parsed_data[selected_muscle][::step]
        rect_emg_selected = rect_dict[selected_muscle][::step]
        
        full_lpf = apply_manual_lpf(rect_dict[selected_muscle], dt, cutoff_freq, filter_order)
        lpf_emg_selected = full_lpf[::step]
        
        # ---------------------------------------------------------
        # OUTPUT 2: GRAFIK 1 - RAW EMG SIGNAL
        # ---------------------------------------------------------
        st.subheader(f"🔍 Analisis Sinyal Sektor Otot: {selected_muscle.title()}")
        
        fig1, ax1 = plt.subplots(figsize=(14, 3.5))
        ax1.plot(time_steps, raw_emg_selected, color='#333333', linewidth=0.7)
        ax1.set_title("RAW EMG SIGNAL", fontsize=11, fontweight='bold', color='#1f77b4', loc='center')
        ax1.set_xlabel("time (sec)", fontsize=9)
        ax1.set_ylabel("EMG (mv)", fontsize=9)
        ax1.grid(True, linestyle='--', alpha=0.5)
        ax1.set_xlim(time_steps[0], time_steps[-1])
        st.pyplot(fig1)
        
        # ---------------------------------------------------------
        # OUTPUT 3: GRAFIK 2 - PREPROCESSED EMG (RECTIFIED & LPF)
        # ---------------------------------------------------------
        fig2, ax2 = plt.subplots(figsize=(14, 3.5))
        ax2.plot(time_steps, rect_emg_selected, color='black', linewidth=0.5, alpha=0.4, label='Rectified')
        ax2.plot(time_steps, lpf_emg_selected, color='red', linewidth=1.5, label='Low-pass Filtered')
        
        ax2.set_title("PREPROCESSED EMG (RECTIFIED & LPF)", fontsize=11, fontweight='bold', color='#333333', loc='center')
        ax2.set_xlabel("time (sec)", fontsize=9)
        ax2.set_ylabel("Processed EMG (mv)", fontsize=9)
        ax2.grid(True, linestyle='--', alpha=0.5)
        ax2.legend(loc='upper right', fontsize=9)
        ax2.set_xlim(time_steps[0], time_steps[-1])
        st.pyplot(fig2)
        
        st.markdown("---")
        
        # ---------------------------------------------------------
        # OUTPUT 4: GRAFIK 3 - THRESHOLDING PETA AKTIVASI 9 OTOT
        # ---------------------------------------------------------
        st.subheader("📊 Peta Aktivasi Semua Otot (Threshold 5% Max)")
        
        fig3, ax3 = plt.subplots(figsize=(14, 6))
        
        for idx, muscle in enumerate(emg_columns):
            r_data = rect_dict[muscle]
            l_data = apply_manual_lpf(r_data, dt, cutoff_freq, filter_order)
            
            max_lpf = max(l_data)
            auto_threshold = 0.05 * max_lpf
            
            l_data_sampled = l_data[::step]
            
            y_vals = []
            for val in l_data_sampled:
                if val >= auto_threshold:
                    y_vals.append(idx)
                else:
                    y_vals.append(float('nan'))
            
            ax3.plot(time_steps, y_vals, linewidth=7.5, solid_capstyle='butt', color='#1f77b4')
            
        ax3.set_yticks(range(len(emg_columns)))
        ax3.set_yticklabels([m.title() for m in emg_columns], fontsize=9)
        ax3.set_xlabel("Time (seconds)", fontsize=10)
        ax3.set_ylabel("Muscles", fontsize=10)
        ax3.set_title("Muscle Activation Profile Each Cycle", fontsize=11, fontweight='bold')
        ax3.grid(axis='x', linestyle='--', alpha=0.5)
        ax3.set_ylim(-0.5, len(emg_columns) - 0.5)
        ax3.set_xlim(time_steps[0], time_steps[-1])
        
        st.pyplot(fig3)

    # =========================================================
    # TAB 2: AMAN DAN KEMBALI BERDIRI KOKOH
    # =========================================================
    with tab2:
        st.header("Ruang Kosong untuk Analisis Lanjut")
        st.write("Tab ini disiapkan untuk pengerjaan data Sudut Sendi & Gait Phase (Menu 3 & 4) alias Tugas 2.")
