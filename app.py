import streamlit as st
import pandas as pd

# Konfigurasi awal halaman
st.set_page_config(page_title="Aplikasi Biomekanik & EMG", layout="wide")

st.title("Aplikasi Pemrosesan Data Biomekanik")

# 1. Membaca File
uploaded_file = st.file_uploader("Unggah file data (.txt)", type=["txt"])

if uploaded_file is not None:
    # Definisi 15 nama kolom sesuai dengan instruksi
    column_names = [
        "time", "heel", "toe", "hip", "knee", "ankle", 
        "gluteus maximus", "biceps femoris short", "biceps femoris long", 
        "vastus medialis", "vastus lateralis", "rectus femoris", 
        "soleus", "gastrocnemius", "tibialis anterior"
    ]
    
    # Membaca file txt
    df = pd.read_csv(uploaded_file, sep='\s+', header=None, names=column_names)
    
    st.subheader("Cuplikan Data Asli")
    st.dataframe(df.head())
    
    # 2. Membuat Tab
    tab1, tab2 = st.tabs(["Grafik EMG", "Tab Lainnya"])
    
    with tab1:
        st.header("Sinyal EMG (Full Wave Rectification - Manual)")
        st.write("Berikut adalah grafik masing-masing otot. Nilai negatif sudah diubah menjadi positif secara manual.")
        
        # Mengambil daftar kolom EMG saja (indeks 6 sampai 14)
        emg_columns = column_names[6:15]
        
        # Looping untuk membuat grafik terpisah tiap kolom EMG
        for col in emg_columns:
            # Membuat subheader untuk nama otot agar rapi
            st.subheader(f"Otot: {col.title()}")
            
            # Melakukan rektifikasi manual hanya untuk kolom yang sedang di-loop
            rectified_data = [val if val >= 0 else -val for val in df[col]]
            
            # Membuat DataFrame sementara khusus untuk 1 plot ini (waktu vs data otot)
            df_single_plot = pd.DataFrame({
                'time': df['time'],
                col: rectified_data
            })
            
            # Set index ke 'time' agar Streamlit otomatis menjadikannya sumbu X
            df_single_plot = df_single_plot.set_index('time')
            
            # Plot grafik untuk otot ini saja
            st.line_chart(df_single_plot, height=300)
            
    with tab2:
        st.header("Ruang Kosong untuk Analisis Lanjut")
        st.write("Tab ini bisa kamu kembangkan nanti untuk memplot data sendi (hip, knee, ankle) atau data gait (heel, toe).")