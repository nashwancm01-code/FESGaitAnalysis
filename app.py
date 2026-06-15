import streamlit as st
import math
import io
import matplotlib.pyplot as plt
import pandas as pd  # Hanya digunakan di layer akhir untuk visualisasi UI

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
    
    # Decode file bytes menjadi text strings per baris
    raw_text = file_bytes.decode('utf-8').splitlines()
    
    # Inisialisasi dictionary kosong untuk menampung data mentah per kolom
    parsed_data = {col: [] for col in column_names}
    
    # Proses parsing baris per baris secara manual (Pengganti pd.read_csv)
    for line in raw_text:
        if not line.strip():
            continue  # Lewati baris kosong
            
        # Pisahkan data berdasarkan spasi/tab (whitespace)
        parts = line.split()
        
        # Validasi: pastikan jumlah kolom pas (15 kolom)
        if len(parts) == len(column_names):
            try:
                # Ubah string menjadi float manual satu per satu
                for idx, col in enumerate(column_names):
                    parsed_data[col].append(float(parts[idx]))
            except ValueError:
                # Jika baris berisi header teks atau error, otomatis dilewati (skip)
                continue
                
    # Menghitung interval waktu (dt) manual
    if len(parsed_data["time"]) > 1:
        dt = parsed_data["time"][1] - parsed_data["time"][0]
    else:
        dt = 0.001
        
    if dt <= 0:
        dt = 0.001
        
    # Ambil daftar nama otot (kolom indeks ke 6 sampai 14)
    emg_columns = column_names[6:15]
    
    # Proses Rektifikasi Manual (Mengubah nilai negatif ke positif tanpa .abs())
    rect_dict = {}
    for col in emg_columns:
        rectified_list = []
        for val in parsed_data[col]:
            if val < 0:
                rectified_list.append(-val)  # Dikali -1 jika negatif
            else:
                rectified_list.append(val)   # Tetap jika positif
        rect_dict[col] = rectified_list
        
    return parsed_data, dt, emg_columns, rect_dict

# --- 3. KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Aplikasi Biomekanik & EMG", layout="wide")
st.title("Aplikasi Pemrosesan Data Biomekanik")

uploaded_file = st.file_uploader("Unggah file data (.txt)", type=["txt"])

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    
    # Panggil fungsi parser manual kita
    parsed_data, dt, emg_columns, rect_dict = load_and_rectify_manual(file_bytes)
    
    # MENU 1: Tampilkan Cuplikan Data Menggunakan Pandas Hanya untuk Render UI Tabel
    st.markdown("---")
    st.subheader("📋 Cuplikan Data Asli")
    df_display = pd.DataFrame(parsed_data)
    st.dataframe(df_display.head())
    st.markdown("---")
    
    tab1, tab2 = st.tabs(["Grafik EMG & Aktivasi", "Tab Analisis Lainnya"])
    
    with tab1:
        st.header("Analisis Sinyal EMG & Aktivasi Otot")
        
        col1, col2 = st.columns(2)
        with col1:
            cutoff_freq = st.slider("Cutoff Frequency LPF (Hz)", min_value=0.5, max_value=20.0, value=5.0, step=0.5)
        with col2:
            filter_order = st.slider("Orde Filter LPF", min_value=1, max_value=5, value=2, step=1)
            
        st.markdown("---")
        
        # =========================================================
        # SEKSI MAP AKTIVASI SEMUA OTOT (LOGIKA MANUAL & ANTI-LAG)
        # =========================================================
        st.subheader("📊 Peta Aktivasi Semua Otot (Threshold 5% Max)")
        st.write("Grafik horizontal di bawah menunjukkan kapan setiap otot aktif (ON) secara bersamaan.")
        
        # Downsampling menggunakan slicing bawaan Python list [::step]
        step = 10
        time_steps = parsed_data['time'][::step]
        
        # Setup canvas gambar
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for idx, muscle in enumerate(emg_columns):
            r_data = rect_dict[muscle]
            l_data = apply_manual_lpf(r_data, dt, cutoff_freq, filter_order)
            
            # Mencari nilai maksimum secara manual tanpa fungsi library khusus
            max_lpf = max(l_data)
            auto_threshold = 0.05 * max_lpf
            
            # Sampling data envelope sesuai step
            l_data_sampled = l_data[::step]
            
            # Logika Thresholding Manual menggunakan Float NaN bawaan Python untuk memutus garis
            y_vals = []
            for val in l_data_sampled:
                if val >= auto_threshold:
                    y_vals.append(idx)  # Isi nilai indeks y jika otot aktif (ON)
                else:
                    y_vals.append(float('nan'))  # Isi Kosong (NaN) jika otot rileks (OFF)
            
            # Plot baris horizontal tebal untuk masing-masing otot
            ax.plot(time_steps, y_vals, linewidth=8, solid_capstyle='butt', color='#1f77b4')
            
        # Desain grafik agar sesuai standard diktat laboratorium
        ax.set_yticks(range(len(emg_columns)))
        ax.set_yticklabels([m.title() for m in emg_columns], fontsize=10)
        ax.set_xlabel("Time (seconds)", fontsize=11)
        ax.set_ylabel("Muscles", fontsize=11)
        ax.set_title("Muscle Activation Profile Each Cycle", fontsize=12, fontweight='bold')
        ax.grid(axis='x', linestyle='--', alpha=0.5)
        ax.set_ylim(-0.5, len(emg_columns) - 0.5)
        
        st.pyplot(fig)
        
        # =========================================================
        # SEKSI DETAIL: TAMPILAN PER INDIVIDU OTOT FOR VALIDASI
        # =========================================================
        st.markdown("---")
        st.subheader("🔍 Analisis Detail per Otot")
        selected_muscle = st.selectbox("Pilih satu otot untuk melihat proses filternya:", emg_columns)
        
        raw_data = parsed_data[selected_muscle]
        rectified_data = rect_dict[selected_muscle]
        lpf_data = apply_manual_lpf(rectified_data, dt, cutoff_freq, filter_order)
        
        # Casting ke DataFrame di baris terakhir ini HANYA karena komponen st.line_chart butuh format ini
        df_detail = pd.DataFrame({
            'time': parsed_data['time'][::step],
            'Raw EMG': raw_data[::step],
            'Rectified': rectified_data[::step],
            'LPF Envelope': lpf_data[::step]
        }).set_index('time')
        
        st.line_chart(df_detail, height=250)

    with tab2:
        st.header("Ruang Kosong untuk Analisis Lanjut")
        st.write("Tab ini disiapkan untuk pengerjaan data Sudut Sendi & Gait Phase (Menu 3 & 4).")
