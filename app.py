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
    
    # Menampilkan sedikit cuplikan data asli
    st.subheader("Cuplikan Data Asli")
    st.dataframe(df.head())
    
    # 2. Membuat Tab
    tab1, tab2 = st.tabs(["Grafik EMG", "Tab Lainnya"])
    
    with tab1:
        st.header("Sinyal EMG (Full Wave Rectification - Manual)")
        st.write("Grafik di bawah ini memplot data otot (kolom 7-15) yang sudah direktifikasi menggunakan logika Python murni tanpa fungsi matematika dari library.")
        
        # Mengambil daftar kolom EMG saja (indeks 6 sampai 14)
        emg_columns = column_names[6:15]
        
        # Membuat DataFrame baru untuk menampung hasil rektifikasi manual
        df_emg_rectified = pd.DataFrame()
        
        # Melakukan Full Wave Rectification secara manual dengan looping & list comprehension murni Python
        for col in emg_columns:
            # Logika manual: jika val >= 0 maka tetap val, jika kurang dari 0 maka diubah jadi -val (positif)
            df_emg_rectified[col] = [val if val >= 0 else -val for val in df[col]]
        
        # Memasukkan kembali kolom 'time' untuk digunakan sebagai sumbu X (waktu)
        df_emg_rectified['time'] = df['time']
        
        # Menjadikan kolom time sebagai index agar st.line_chart memplot sumbu X dengan benar
        df_plot = df_emg_rectified.set_index('time')
        
        # Plotting interaktif ke Streamlit
        st.line_chart(df_plot, height=500)
        
    with tab2:
        st.header("Ruang Kosong untuk Analisis Lanjut")
        st.write("Tab ini bisa kamu kembangkan nanti untuk memplot data sendi (hip, knee, ankle) atau data gait (heel, toe).")
