import streamlit as st
import pandas as pd
import math

# --- FUNGSI LPF MANUAL ---
def apply_manual_lpf(data, dt, cutoff, order):
    """
    Mengaplikasikan Low Pass Filter (IIR orde-1) secara manual.
    Untuk orde > 1, filter diaplikasikan secara bertingkat (cascaded).
    """
    if cutoff <= 0 or order < 1:
        return data
    
    # Penyesuaian cutoff frequency untuk filter bertingkat (cascaded)
    # Tujuannya agar titik -3dB tetap akurat di frekuensi cutoff yang diinginkan
    fc_adj = cutoff / math.sqrt(2**(1.0 / order) - 1.0)
    
    # Menghitung konstanta filter (alpha)
    tau = 1.0 / (2.0 * math.pi * fc_adj)
    alpha = dt / (tau + dt)
    
    # Copy data agar tidak mengubah array aslinya
    y = list(data)
    
    # Mengaplikasikan filter sebanyak 'order' kali
    for _ in range(order):
        y_new = []
        # Inisialisasi nilai pertama
        prev_y = y[0]
        y_new.append(prev_y)
        
        # Looping perhitungan filter (y[i] = alpha * x[i] + (1 - alpha) * y[i-1])
        for i in range(1, len(y)):
            curr_y = alpha * y[i] + (1.0 - alpha) * prev_y
            y_new.append(curr_y)
            prev_y = curr_y
            
        # Update y untuk iterasi orde selanjutnya
        y = y_new
        
    return y

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Aplikasi Biomekanik & EMG", layout="wide")

st.title("Aplikasi Pemrosesan Data Biomekanik")

# 1. Membaca File
uploaded_file = st.file_uploader("Unggah file data (.txt)", type=["txt"])

if uploaded_file is not None:
    # Definisi 15 nama kolom
    column_names = [
        "time", "heel", "toe", "hip", "knee", "ankle", 
        "gluteus maximus", "biceps femoris short", "biceps femoris long", 
        "vastus medialis", "vastus lateralis", "rectus femoris", 
        "soleus", "gastrocnemius", "tibialis anterior"
    ]
    
    # Membaca file txt
    df = pd.read_csv(uploaded_file, sep='\s+', header=None, names=column_names)
    
    # Menghitung interval waktu (dt) atau sampling time dari kolom 'time'
    dt = df['time'].iloc[1] - df['time'].iloc[0]
    if dt <= 0:
        dt = 0.001  # Fallback default ke 1000 Hz jika terjadi error waktu
    
    st.subheader("Cuplikan Data Asli")
    st.dataframe(df.head())
    
    # 2. Membuat Tab
    tab1, tab2 = st.tabs(["Grafik EMG", "Tab Lainnya"])
    
    with tab1:
        st.header("Sinyal EMG (Rectified & Low Pass Filter)")
        st.write("Atur parameter LPF menggunakan slider di bawah. Grafik akan menampilkan perbandingan antara sinyal rektifikasi (mentah) dengan sinyal hasil LPF (Envelope).")
        
        # --- SLIDER LPF ---
        col1, col2 = st.columns(2)
        with col1:
            # Slider frekuensi cutoff (biasanya 2-10 Hz cukup untuk EMG envelope)
            cutoff_freq = st.slider("Cutoff Frequency (Hz)", min_value=0.5, max_value=20.0, value=5.0, step=0.5)
        with col2:
            # Slider orde filter
            filter_order = st.slider("Orde Filter", min_value=1, max_value=5, value=2, step=1)
        
        # Daftar kolom otot
        emg_columns = column_names[6:15]
        
        st.markdown("---")
        
        # Looping untuk membuat grafik terpisah
        for col in emg_columns:
            st.subheader(f"Otot: {col.title()}")
            
            # Langkah 1: Full Wave Rectification (Manual)
            rectified_data = [val if val >= 0 else -val for val in df[col]]
            
            # Langkah 2: Low Pass Filter (Manual)
            lpf_data = apply_manual_lpf(rectified_data, dt, cutoff_freq, filter_order)
            
            # Langkah 3: Gabungkan ke DataFrame sementara khusus untuk 1 plot
            df_single_plot = pd.DataFrame({
                'time': df['time'],
                'Rectified': rectified_data,
                'LPF (Envelope)': lpf_data
            }).set_index('time')
            
            # Plot grafik untuk otot ini saja
            # Warna akan terpisah secara otomatis (Satu untuk Rectified, satu untuk LPF)
            st.line_chart(df_single_plot, height=300)
            
    with tab2:
        st.header("Ruang Kosong untuk Analisis Lanjut")
        st.write("Tab ini bisa kamu kembangkan nanti untuk memplot data sendi (hip, knee, ankle) atau data gait (heel, toe).")