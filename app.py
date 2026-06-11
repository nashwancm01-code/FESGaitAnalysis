import streamlit as st
import pandas as pd
import math
import io

# --- 1. FUNGSI LPF MANUAL (DENGAN CACHE) ---
# st.cache_data menyimpan hasil fungsi ini. Jika cutoff/order tidak berubah, hasilnya langsung diambil dari memori.
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

# --- 2. FUNGSI MEMBACA FILE & REKTIFIKASI (DENGAN CACHE) ---
# File hanya dibaca dan direktifikasi 1 kali saja saat diunggah
@st.cache_data
def load_and_rectify(file_bytes):
    column_names = [
        "time", "heel", "toe", "hip", "knee", "ankle", 
        "gluteus maximus", "biceps femoris short", "biceps femoris long", 
        "vastus medialis", "vastus lateralis", "rectus femoris", 
        "soleus", "gastrocnemius", "tibialis anterior"
    ]
    
    # Membaca data dari format bytes
    df = pd.read_csv(io.BytesIO(file_bytes), sep='\s+', header=None, names=column_names)
    
    dt = df['time'].iloc[1] - df['time'].iloc[0]
    if dt <= 0:
        dt = 0.001
        
    emg_columns = column_names[6:15]
    
    # Melakukan rektifikasi manual dan menyimpannya di dalam dictionary
    rect_dict = {}
    for col in emg_columns:
        rect_dict[col] = [val if val >= 0 else -val for val in df[col]]
        
    return df, dt, emg_columns, rect_dict

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Aplikasi Biomekanik & EMG", layout="wide")

st.title("Aplikasi Pemrosesan Data Biomekanik")

uploaded_file = st.file_uploader("Unggah file data (.txt)", type=["txt"])

if uploaded_file is not None:
    # Memanggil fungsi baca dengan .getvalue() agar bisa dicache oleh Streamlit
    file_bytes = uploaded_file.getvalue()
    df, dt, emg_columns, rect_dict = load_and_rectify(file_bytes)
    
    st.subheader("Cuplikan Data Asli")
    st.dataframe(df.head())
    
    tab1, tab2 = st.tabs(["Grafik EMG", "Tab Lainnya"])
    
    with tab1:
        st.header("Sinyal EMG (Rectified & Low Pass Filter)")
        st.write("Aplikasi sekarang menggunakan cache. Loading hanya terjadi saat slider digeser ke angka baru.")
        
        col1, col2 = st.columns(2)
        with col1:
            cutoff_freq = st.slider("Cutoff Frequency (Hz)", min_value=0.5, max_value=20.0, value=5.0, step=0.5)
        with col2:
            filter_order = st.slider("Orde Filter", min_value=1, max_value=5, value=2, step=1)
        
        st.markdown("---")
        
        # Looping untuk memplot grafik
        for col in emg_columns:
            st.subheader(f"Otot: {col.title()}")
            
            # Ambil data rektifikasi dari hasil cache
            rectified_data = rect_dict[col]
            
            # Aplikasikan LPF (juga menggunakan cache)
            lpf_data = apply_manual_lpf(rectified_data, dt, cutoff_freq, filter_order)
            
            # Gabungkan ke DataFrame untuk plot
            df_single_plot = pd.DataFrame({
                'time': df['time'],
                'Rectified': rectified_data,
                'LPF (Envelope)': lpf_data
            }).set_index('time')
            
            # Plot grafik
            st.line_chart(df_single_plot, height=300)
            
    with tab2:
        st.header("Ruang Kosong untuk Analisis Lanjut")
        st.write("Tab ini bisa kamu kembangkan nanti.")