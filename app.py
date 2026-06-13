import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

# ==========================================
# 1. FUNGSI PENGOLAHAN SINYAL (MANUAL)
# ==========================================
def apply_manual_lpf(data, dt, cutoff_freq, order=2):
    """
    Fungsi untuk mengaplikasikan Low Pass Filter (LPF) secara manual.
    """
    RC = 1.0 / (2 * math.pi * cutoff_freq)
    alpha = dt / (RC + dt)
    
    filtered_data = data.copy()
    for _ in range(order):
        temp_filtered = [filtered_data[0]]
        for i in range(1, len(filtered_data)):
            val = alpha * filtered_data[i] + (1 - alpha) * temp_filtered[i-1]
            temp_filtered.append(val)
        filtered_data = temp_filtered
        
    return filtered_data

# ==========================================
# 2. PENGATURAN HALAMAN STREAMLIT
# ==========================================
st.set_page_config(page_title="Analisis Biomekanik", layout="wide")
st.title("Aplikasi Pemrosesan Data Biomekanik")

# Upload file .txt
uploaded_file = st.file_uploader("Unggah file data (.txt)", type="txt")

if uploaded_file is not None:
    try:
        # Membaca data dari file teks
        df = pd.read_csv(uploaded_file, sep='\t')
        
        # Asumsi kolom pertama adalah waktu, sisanya adalah data otot
        time_col = df.columns[0]
        emg_columns = df.columns[1:].tolist()
        
        # Ubah nama kolom waktu menjadi 'time' agar seragam
        df.rename(columns={time_col: 'time'}, inplace=True)
        
        # Menghitung delta time (dt)
        dt = df['time'].iloc[1] - df['time'].iloc[0]
        
        # Menyearahkan sinyal (Rectification) -> absolute value
        rect_dict = {}
        for col in emg_columns:
            rect_dict[col] = df[col].abs().tolist()
            
        # ==========================================
        # 3. TAMPILAN TAB STREAMLIT
        # ==========================================
        tab1, tab2 = st.tabs(["Grafik EMG", "Tab Lainnya"])
        
        # ------------------------------------------
        # TAB 1: PENGOLAHAN EMG & PETA AKTIVASI
        # ------------------------------------------
        with tab1:
            st.header("Analisis Sinyal EMG & Aktivasi Otot")
            
            # Kontrol Global untuk LPF
            col1, col2 = st.columns(2)
            with col1:
                cutoff_freq = st.slider("Cutoff Frequency LPF (Hz)", min_value=0.5, max_value=20.0, value=5.0, step=0.5)
            with col2:
                filter_order = st.slider("Orde Filter LPF", min_value=1, max_value=5, value=2, step=1)
                
            st.markdown("---")
            
            # BAGIAN A: PETA AKTIVASI SEMUA OTOT (MENGGUNAKAN MATPLOTLIB)
            st.subheader("📊 Peta Aktivasi Semua Otot (Threshold 5% Max)")
            st.write("Grafik horizontal di bawah menunjukkan kapan setiap otot aktif (ON) dan istirahat (OFF) secara bersamaan.")
            
            step = 10 # Mengambil data tiap 10 langkah agar grafik lebih ringan
            time_steps = df['time'][::step].tolist()
            
            # Membuat kanvas (figure) matplotlib
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Looping untuk memproses dan menggambar setiap otot
            for idx, muscle in enumerate(emg_columns):
                r_data = rect_dict[muscle]
                # Aplikasikan LPF
                l_data = apply_manual_lpf(r_data, dt, cutoff_freq, filter_order)
                
                # Deteksi Threshold (5% dari nilai maksimum LPF otot tersebut)
                max_lpf = max(l_data)
                auto_threshold = 0.05 * max_lpf
                
                # Slicing data biar sesuai langkah step
                l_data_sampled = l_data[::step]
                
                # Bikin nilai untuk sumbu Y (jika ON maka ditaruh di posisinya, jika OFF ditaruh NaN agar bolong)
                y_vals = []
                for val in l_data_sampled:
                    if val >= auto_threshold:
                        y_vals.append(idx)
                    else:
                        y_vals.append(np.nan)
                
                # Gambar balok horizontalnya
                ax.plot(time_steps, y_vals, linewidth=8, solid_capstyle='butt', color='#1f77b4')
                
            # Merapikan tampilan grafik
            ax.set_yticks(range(len(emg_columns)))
            ax.set_yticklabels([m.title() for m in emg_columns], fontsize=10)
            ax.set_xlabel("Time (seconds)", fontsize=11)
            ax.set_ylabel("Muscles", fontsize=11)
            ax.set_title("Muscle Activation Profile Each Cycle", fontsize=12, fontweight='bold')
            ax.grid(axis='x', linestyle='--', alpha=0.5)
            ax.set_ylim(-0.5, len(emg_columns) - 0.5)
            
            # Tampilkan grafik dari matplotlib ke Streamlit
            st.pyplot(fig)
            
            # ------------------------------------------
            # BAGIAN B: LACI VALIDASI DETAIL PER OTOT
            # ------------------------------------------
            st.markdown("---")
            with st.expander("🔍 Lihat Validasi & Detail Sinyal per Otot (Klik untuk Membuka)"):
                st.write("Gunakan menu ini jika ingin memeriksa proses pemisahan sinyal mentah ke sinyal envelope.")
                selected_muscle = st.selectbox("Pilih otot yang ingin divalidasi:", emg_columns)
                
                # Tarik data dari otot yang dipilih
                raw_data = df[selected_muscle].tolist()
                rectified_data = rect_dict[selected_muscle]
                lpf_data = apply_manual_lpf(rectified_data, dt, cutoff_freq, filter_order)
                
                # 1. GRAFIK KHUSUS RAW EMG
                st.markdown(f"#### 1️⃣ Sinyal Mentah (Raw EMG): {selected_muscle.title()}")
                df_raw = pd.DataFrame({
                    'time': df['time'][::step],
                    'Raw EMG': raw_data[::step]
                }).set_index('time')
                st.line_chart(df_raw, height=200)
                
                # 2. GRAFIK RECTIFIED & LPF
                st.markdown("#### 2️⃣ Hasil Penyearahan (Rectified) & Envelope (LPF Filtered)")
                df_processed = pd.DataFrame({
                    'time': df['time'][::step],
                    'Rectified': rectified_data[::step],
                    'LPF Envelope': lpf_data[::step]
                }).set_index('time')
                st.line_chart(df_processed, height=250)

        # ------------------------------------------
        # TAB 2: TEMPAT UNTUK KINEMATIK (SUDUT SENDI NANTI)
        # ------------------------------------------
        with tab2:
            st.header("Analisis Sudut Sendi (Kinematik)")
            st.info("Area ini disiapkan untuk data Hip, Knee, dan Ankle.")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses file: {e}")
