import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# CONSTANTS & CONFIGURATION
# ==========================================
st.set_page_config(page_title="Gait Extraction & STFT Analysis", layout="wide")

MUSCLES = [
    "Gluteus Maximus", "Biceps Femoris Short", "Biceps Femoris Long", 
    "Vastus Medialis", "Vastus Lateralis", "Rectus Femoris", 
    "Medial Gastrocnemius", "Tibialis Anterior", "Soleus"
]

# ==========================================
# MATHEMATICAL DSP FUNCTIONS (SCRATCH CODING)
# ==========================================

def apply_manual_lpf(signal, fc, fs, order=4):
    """
    Filter Low-Pass IIR Orde 1 yang di-cascade sebanyak 'order' kali 
    untuk meniru karakteristik filter Butterworth.
    """
    w0 = 2 * np.pi * fc / fs
    alpha = w0 / (w0 + 1)
    
    y = np.array(signal, dtype=float)
    for _ in range(order):
        y_new = np.zeros_like(y)
        y_new[0] = y[0]
        for n in range(1, len(y)):
            y_new[n] = alpha * y[n] + (1 - alpha) * y_new[n-1]
        y = y_new
    return y

def detect_crossings_detailed(signal, time_arr, threshold):
    """
    Mendeteksi kapan sinyal melewati threshold ke arah atas (Rise) dan bawah (Fall)
    menggunakan prinsip interpolasi linier presisi tinggi.
    """
    cross_up = []
    cross_down = []
    for i in range(1, len(signal)):
        # Deteksi Naik (Rise)
        if signal[i-1] < threshold <= signal[i]:
            t_cross = time_arr[i-1] + (threshold - signal[i-1]) * (time_arr[i] - time_arr[i-1]) / (signal[i] - signal[i-1])
            cross_up.append(t_cross)
        # Deteksi Turun (Fall)
        elif signal[i-1] >= threshold > signal[i]:
            t_cross = time_arr[i-1] + (threshold - signal[i-1]) * (time_arr[i] - time_arr[i-1]) / (signal[i] - signal[i-1])
            cross_down.append(t_cross)
    return cross_up, cross_down

def compute_stft_manual(signal, fs, nperseg=128, noverlap=64):
    """
    Menghitung matriks STFT secara manual menggunakan fungsi jendela Hamming
    dan konversi ke Power Spektrogram.
    """
    step = nperseg - noverlap
    # Fungsi jendela Hamming manual
    window = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(nperseg) / (nperseg - 1))
    
    stft_matrix = []
    time_slots = []
    
    for start in range(0, len(signal) - nperseg + 1, step):
        segment = signal[start : start + nperseg]
        windowed = segment * window
        # Perhitungan FFT internal Radix-2
        fft_res = np.fft.fft(windowed)
        half_len = nperseg // 2 + 1
        power_spectrum = np.abs(fft_res[:half_len]) ** 2
        stft_matrix.append(power_spectrum)
        time_slots.append((start + nperseg / 2) / fs)
        
    stft_matrix = np.array(stft_matrix).T
    freqs = np.arange(half_len) * (fs / nperseg)
    return freqs, np.array(time_slots), stft_matrix

# ==========================================
# SIDEBAR PANEL KONTROL
# ==========================================
st.sidebar.header("Panel Kontrol")
uploaded_file = st.sidebar.file_uploader("LOAD DATA (TXT)", type=["txt"])

fc_input = st.sidebar.slider("Cutoff Frequency LPF (Hz) EMG", min_value=1.0, max_value=20.0, value=6.30, step=0.1)

# ==========================================
# MAIN APPLICATION LOGIC
# ==========================================
st.title("Gait Parameter Extraction & STFT Analysis")

if uploaded_file is not None:
    # Membaca data log file
    try:
        df = pd.read_csv(uploaded_file, sep=r'\s+', header=None)
    except Exception:
        df = pd.read_csv(uploaded_file, sep='\t', header=None)
        
    total_rows = len(df)
    st.sidebar.success(f"Jumlah Data : {total_rows}")
    
    # Menghitung Frekuensi Sampling Berdasarkan Kolom Waktu Asli
    raw_time = df.iloc[:, 0].values
    dt = np.mean(np.diff(raw_time))
    fs = 1.0 / dt if dt > 0 else 1000.0
    
    # Normalisasi vektor waktu agar seragam mulai dari 0 detik
    waktu = np.arange(total_rows) / fs
    
    # Ekstraksi Kolom Sinyal Kinematika dan FSR
    hip_angle = df.iloc[:, 1].values
    knee_angle = df.iloc[:, 2].values
    ankle_angle = df.iloc[:, 3].values
    heel_raw = df.iloc[:, 4].values
    toe_raw = df.iloc[:, 5].values
    
    # Proses Filter LPF pada Sinyal FSR Kontrol Langkah
    heel_filtered = apply_manual_lpf(heel_raw, fc_input, fs, order=4)
    toe_filtered = apply_manual_lpf(toe_raw, fc_input, fs, order=4)
    
    # Deteksi Titik Nilai Crossing Fase Berjalan
    gait_threshold = 0.15
    heel_up, heel_down = detect_crossings_detailed(heel_filtered, waktu, gait_threshold)
    toe_up, toe_down = detect_crossings_detailed(toe_filtered, waktu, gait_threshold)
    
    # ==========================================
    # PROSES DATA PARAMETER TEMPORAL & RATA-RATA
    # ==========================================
    cycles_data = []
    for i in range(len(heel_up) - 1):
        hs1 = heel_up[i]
        hs2 = heel_up[i+1]
        valid_to = [t for t in toe_down if hs1 < t < hs2]
        
        if not valid_to:
            continue
        to1 = valid_to[0]
        
        cycle_time = hs2 - hs1
        stance_time = to1 - hs1
        swing_time = hs2 - to1
        
        cycles_data.append({
            'cycle': str(i + 1),
            'start_time': cycle_time, # Menyimpan data dasar sebelum pembulatan untuk mean
            'toe_off_time': to1,
            'end_time': hs2,
            'gait_cycle_time': cycle_time,
            'stance_time': stance_time,
            'swing_time': swing_time,
            'stance_percent': (stance_time / cycle_time) * 100,
            'swing_percent': (swing_time / cycle_time) * 100
        })
        
    df_temporal = pd.DataFrame(cycles_data)
    
    if not df_temporal.empty:
        # Hitung Nilai Statistik Rata-rata Sebelum Diformat String
        avg_vals = df_temporal.mean(numeric_only=True)
        
        # Format Data Row Utama
        df_display = pd.DataFrame()
        df_display['cycle'] = df_temporal['cycle']
        df_display['start_time'] = df_temporal['start_time'].round(3)
        df_display['toe_off_time'] = df_temporal['toe_off_time'].round(3)
        df_display['end_time'] = df_temporal['end_time'].round(3)
        df_display['gait_cycle_time'] = df_temporal['gait_cycle_time'].round(3)
        df_display['stance_time'] = df_temporal['stance_time'].round(3)
        df_display['swing_time'] = df_temporal['swing_time'].round(3)
        df_display['stance_percent'] = df_temporal['stance_percent'].round(2)
        df_display['swing_percent'] = df_temporal['swing_percent'].round(2)
        
        # Susun Baris Rata-rata
        avg_row = {
            'cycle': 'Rata-rata',
            'start_time': round(avg_vals['start_time'], 3),
            'toe_off_time': round(avg_vals['toe_off_time'], 3),
            'end_time': round(avg_vals['end_time'], 3),
            'gait_cycle_time': round(avg_vals['gait_cycle_time'], 3),
            'stance_time': round(avg_vals['stance_time'], 3),
            'swing_time': round(avg_vals['swing_time'], 3),
            'stance_percent': round(avg_vals['stance_percent'], 2),
            'swing_percent': round(avg_vals['swing_percent'], 2)
        }
        df_display = pd.concat([df_display, pd.DataFrame([avg_row])], ignore_index=True)
        
        # Tampilkan Indikator Global di Sidebar
        mean_c_time = avg_vals['gait_cycle_time']
        cadence_calc = (60.0 / mean_c_time) if mean_c_time > 0 else 0.0
        st.sidebar.markdown("### Temporal Parameters:")
        st.sidebar.write(f"* Rata-rata Cycle: {mean_c_time:.3f} s")
        st.sidebar.write(f"* Cadence: {cadence_calc:.2f} step/min")
        st.sidebar.write(f"* Jumlah Cycle: {len(df_temporal)}")
    else:
        df_display = pd.DataFrame(columns=['cycle', 'start_time', 'toe_off_time', 'end_time', 'gait_cycle_time', 'stance_time', 'swing_time', 'stance_percent', 'swing_percent'])
        st.sidebar.write("* Data siklus tidak valid.")

    # ==========================================
    # ANTARMUKA TABS UTAMA streamlit
    # ==========================================
    t1, t2, t3, t4, t5 = st.tabs(["GAIT PARAMETERS", "DYNAMIC EMG", "EMG PREPROCESSING", "PARAMETER (TABEL)", "STFT ANALYSIS"])
    
    # ------------------------------------------
    # TAB 1: GAIT PARAMETERS
    # ------------------------------------------
    with t1:
        st.subheader("Hasil Evaluasi Filter FSR & Analisis Fase Berjalan")
        
        # REVISI 1: Grafik Input vs Output Menggunakan Sumbu X Berbasis Waktu (Detik)
        fig_io, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
        ax1.plot(waktu, heel_raw, color='blue', alpha=0.7, label='Heel Mentah')
        ax1.plot(waktu, toe_raw, color='red', alpha=0.7, label='Toe Mentah')
        ax1.set_title("Sinyal Masukan Mentah (Input)")
        ax1.set_ylabel("Amplitudo")
        ax1.grid(True)
        ax1.legend()
        
        ax2.plot(waktu, heel_filtered, color='blue', label='Heel Filtering')
        ax2.plot(waktu, toe_filtered, color='red', label='Toe Filtering')
        ax2.set_title(f"OUTPUT / HASIL FILTERING (Cutoff: {fc_input} Hz)")
        ax2.set_xlabel("Waktu (s)")
        ax2.set_ylabel("Amplitudo")
        ax2.grid(True)
        ax2.legend()
        st.pyplot(fig_io)
        
        # REVISI 2: Grafik Fase dengan Garis Putus-putus Warna-warni Berbeda Tiap Peristiwa
        st.markdown("---")
        fig_phase, ax3 = plt.subplots(figsize=(11, 4))
        ax3.plot(waktu, heel_filtered, color='blue', linewidth=1.5, label='Heel')
        ax3.plot(waktu, toe_filtered, color='red', linewidth=1.5, label='Toe')
        ax3.axhline(gait_threshold, color='black', linestyle='--', alpha=0.5, label='Threshold')
        
        for idx, t in enumerate(heel_up):
            ax3.axvline(x=t, color='green', linestyle=':', linewidth=1.2, label='Heel Strike (Naik)' if idx==0 else "")
        for idx, t in enumerate(heel_down):
            ax3.axvline(x=t, color='purple', linestyle=':', linewidth=1.2, label='Heel Off (Turun)' if idx==0 else "")
        for idx, t in enumerate(toe_up):
            ax3.axvline(x=t, color='cyan', linestyle=':', linewidth=1.2, label='Toe Strike (Naik)' if idx==0 else "")
        for idx, t in enumerate(toe_down):
            ax3.axvline(x=t, color='orange', linestyle=':', linewidth=1.2, label='Toe Off (Turun)' if idx==0 else "")
            
        ax3.set_title("HEEL dan TOE PHASE DETECTION WITH MULTICOLORED EVENTS")
        ax3.set_xlabel("Waktu (s)")
        ax3.set_ylabel("Amplitudo")
        ax3.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
        ax3.grid(True)
        st.pyplot(fig_phase)
        
        # Grafik Kinematika Sendi 3-Kanal
        st.markdown("---")
        fig_ang, ax4 = plt.subplots(figsize=(11, 3.5))
        ax4.plot(waktu, hip_angle, label='Hip Joint')
        ax4.plot(waktu, knee_angle, label='Knee Joint')
        ax4.plot(waktu, ankle_angle, label='Ankle Joint')
        ax4.set_title("JOINT ANGLE PARAMETERS")
        ax4.set_xlabel("Waktu (s)")
        ax4.set_ylabel("Sudut (Derajat)")
        ax4.legend()
        ax4.grid(True)
        st.pyplot(fig_ang)

    # ------------------------------------------
    # TAB 2: DYNAMIC EMG (MATRIKS AKTIVASI)
    # ------------------------------------------
    with t2:
        st.subheader("Muscle Activation Pattern (All 9 Muscles)")
        fig_act, ax_act = plt.subplots(figsize=(11, 5))
        
        for i, m_name in enumerate(MUSCLES):
            emg_raw = df.iloc[:, 6 + i].values
            emg_rect = np.abs(emg_raw)
            emg_env = apply_manual_lpf(emg_rect, fc_input, fs, order=4)
            
            # Normalisasi Ambang Batas Biner On/Off Otot
            max_val = np.max(emg_env) if np.max(emg_env) > 0 else 1.0
            binary_activation = np.where((emg_env / max_val) > 0.05, 1, 0)
            
            # Membuat plotting balok koordinasi horizontal
            ax_act.fill_between(waktu, i - 0.3, i + 0.3, where=(binary_activation == 1), color='C0', alpha=0.9)
            
        ax_act.set_yticks(range(len(MUSCLES)))
        ax_act.set_yticklabels(MUSCLES)
        ax_act.set_title("Muscle Activation Map Each Cycle")
        ax_act.set_xlabel("Waktu (s)")
        ax_act.grid(axis='x', linestyle='--')
        st.pyplot(fig_act)

    # ------------------------------------------
    # TAB 3: EMG PREPROCESSING (DROPDOWN SELECTION)
    # ------------------------------------------
    with t3:
        st.subheader("Preprocessing Analisis Linear Envelope Sinyal Otot")
        selected_m1 = st.selectbox("Pilih Otot untuk Dianalisis (Pre-processing):", MUSCLES, key="pre_m")
        m_idx1 = MUSCLES.index(selected_m1)
        
        emg_raw_sel = df.iloc[:, 6 + m_idx1].values
        emg_rect_sel = np.abs(emg_raw_sel)
        emg_env_sel = apply_manual_lpf(emg_rect_sel, fc_input, fs, order=4)
        
        fig_emg, (ax_r, ax_e) = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
        ax_r.plot(waktu, emg_raw_sel, color='black', linewidth=0.6)
        ax_r.set_title(f"RAW EMG SIGNAL - {selected_m1}")
        ax_r.set_ylabel("Amplitudo (mV)")
        ax_r.grid(True)
        
        ax_e.plot(waktu, emg_env_sel, color='crimson', linewidth=1.5)
        ax_e.set_title(f"LINEAR ENVELOPE (Rectified + Low Pass Filter {fc_input} Hz)")
        ax_e.set_xlabel("Waktu (s)")
        ax_e.set_ylabel("Amplitudo (mV)")
        ax_e.grid(True)
        st.pyplot(fig_emg)

    # ------------------------------------------
    # TAB 4: PARAMETER (TABEL DENGAN BARIS RATA-RATA)
    # ------------------------------------------
    with t4:
        st.subheader("Tabel Hasil Ekstraksi Parameter Temporal Langkah")
        # REVISI 3: Menampilkan data tabular lengkap dengan row "Rata-rata" di bagian dasar layar
        st.dataframe(df_display, use_container_width=True)

    # ------------------------------------------
    # TAB 5: STFT ANALYSIS (TIME-FREQUENCY HEATMAP)
    # ------------------------------------------
    with t5:
        st.subheader("Analisis Spektrogram Waktu-Frekuensi Menggunakan STFT")
        selected_m2 = st.selectbox("Pilih Isyarat Data untuk Grafik Spektrogram:", ["Sensor Heel", "Sensor Toe"] + MUSCLES, key="stft_m")
        
        # Seleksi target sinyal berdasarkan opsi dropdown
        if selected_m2 == "Sensor Heel":
            target_signal = heel_filtered
        elif selected_m2 == "Sensor Toe":
            target_signal = toe_filtered
        else:
            m_idx2 = MUSCLES.index(selected_m2)
            target_signal = df.iloc[:, 6 + m_idx2].values
            
        freqs, times, spectrogram_matrix = compute_stft_manual(target_signal, fs, nperseg=128, noverlap=64)
        
        # Visualisasi Spektrogram Heatmap 2D
        fig_spec, ax_s = plt.subplots(figsize=(11, 4.5))
        mesh = ax_s.pcolormesh(times, freqs, spectrogram_matrix, shading='gouraud', cmap='viridis')
        ax_s.set_title(f"STFT Spectrogram Matrix - {selected_m2}")
        ax_s.set_xlabel("Waktu (s)")
        ax_s.set_ylabel("Frekuensi (Hz)")
        
        # Batasi visualisasi sumbu frekuensi agar fokus pada area penting
        if "Sensor" in selected_m2:
            ax_s.set_ylim(0, 15)
        else:
            ax_s.set_ylim(0, 50) # Tampilan frekuensi otot yang lebih tinggi
            
        cbar = fig_spec.colorbar(mesh, ax=ax_s)
        cbar.set_label("Kerapatan Daya (Power)")
        st.pyplot(fig_spec)

else:
    st.info("Silakan unggah file data berekstensi .TXT terlebih dahulu melalui panel kontrol sebelah kiri.")
