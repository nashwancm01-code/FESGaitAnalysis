import streamlit as st
import math
import cmath # Untuk operasi bilangan kompleks pada FFT manual
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ==========================================
# --- 1. FUNGSI MATEMATIKA & DSP MANUAL ---
# ==========================================

def get_mean(data):
    if not data: return 0
    return sum(data) / len(data)

def get_diff(data):
    return [data[i] - data[i-1] for i in range(1, len(data))]

def normalize_signal(signal):
    signal = np.asarray(signal)
    min_val = np.min(signal)
    max_val = np.max(signal)
    if max_val - min_val == 0:
        return np.zeros_like(signal)
    return (signal - min_val) / (max_val - min_val)

@st.cache_data
def apply_manual_lpf(data, fs, cutoff=6, order=4):
    """Low Pass Filter IIR manual"""
    dt = 1.0 / fs
    if cutoff <= 0 or order < 1: return data
    fc_adj = cutoff / math.sqrt(2**(1.0 / order) - 1.0)
    tau = 1.0 / (2.0 * math.pi * fc_adj)
    alpha = dt / (tau + dt)
    
    y = list(data)
    for _ in range(order):
        y_new = [y[0]]
        for i in range(1, len(y)):
            curr_y = alpha * y[i] + (1.0 - alpha) * y_new[-1]
            y_new.append(curr_y)
        y = y_new
    return np.array(y) # Return sebagai numpy array agar mudah diolah

def detect_activation_segments_manual(time, signal, threshold):
    """Mendeteksi segmen mulai dan akhir saat sinyal melebihi threshold."""
    segments = []
    start = None
    for i in range(1, len(signal)):
        is_active_now = signal[i] >= threshold
        was_active_before = signal[i-1] >= threshold
        
        if is_active_now and not was_active_before:
            start = time[i]
        elif not is_active_now and was_active_before and start is not None:
            segments.append((start, time[i]))
            start = None
    if start is not None:
        segments.append((start, time[-1]))
    return segments

# --- FFT MANUAL UNTUK STFT ---
def radix2_fft(x):
    """Algoritma Fast Fourier Transform murni Python."""
    N = len(x)
    if N <= 1: return x
    even = radix2_fft(x[0::2])
    odd = radix2_fft(x[1::2])
    T = [cmath.exp(-2j * cmath.pi * k / N) * odd[k] for k in range(N // 2)]
    return [even[k] + T[k] for k in range(N // 2)] + [even[k] - T[k] for k in range(N // 2)]

@st.cache_data
def compute_stft_manual(signal, fs, nperseg=128):
    """STFT (Short-Time Fourier Transform) manual menggantikan scipy stft."""
    step = nperseg // 2
    power_matrix = []
    time_bins = []
    
    # Hamming window manual
    window = [0.54 - 0.46 * math.cos(2 * math.pi * j / (nperseg - 1)) for j in range(nperseg)]
    
    for i in range(0, len(signal) - nperseg, step):
        segment = signal[i:i+nperseg]
        windowed = [segment[j] * window[j] for j in range(nperseg)]
        
        # Hitung FFT
        X = radix2_fft(windowed)
        
        # Ambil magnitudo setengah (karena simetris)
        mag = [abs(X[k]) for k in range(nperseg // 2)]
        power_matrix.append(mag)
        time_bins.append(i / fs)
        
    freq_bins = [k * fs / nperseg for k in range(nperseg // 2)]
    
    # Transpose matriks untuk plotting (baris=frekuensi, kolom=waktu)
    transposed_power = [[power_matrix[col][row] for col in range(len(power_matrix))] for row in range(len(power_matrix[0]))]
    return freq_bins, time_bins, transposed_power

# ==========================================
# --- 2. PARSER DATA ---
# ==========================================

@st.cache_data
def load_and_process_data(file_bytes):
    raw_text = file_bytes.decode('utf-8').splitlines()
    data = []
    for line in raw_text:
        try:
            parts = [float(x) for x in line.split()]
            if len(parts) >= 15:
                data.append(parts)
        except ValueError: continue
        
    if not data: return None, "Data tidak valid."
    
    # Ekstrak per kolom
    t = [row[0] for row in data]
    heel = [row[1] for row in data]
    toe = [row[2] for row in data]
    hip = [row[3] for row in data]
    knee = [row[4] for row in data]
    ankle = [row[5] for row in data]
    
    # Transpose EMG (9 otot)
    emg = [[row[col] for row in data] for col in range(6, 15)]
    
    diff_t = get_diff(t)
    fs = 1.0 / get_mean(diff_t) if diff_t else 1000.0
    
    return {
        "t": np.array(t), "heel": np.array(heel), "toe": np.array(toe), "hip": np.array(hip), 
        "knee": np.array(knee), "ankle": np.array(ankle), "emg": emg, "fs": fs
    }, None

# ==========================================
# --- 3. UI APLIKASI STREAMLIT ---
# ==========================================

st.set_page_config(page_title="FP PSB - Gait & STFT", layout="wide")
st.title("Gait Parameter Extraction & STFT Analysis")

muscle_names = ["Gluteus Maximus", "Biceps Femoris Short", "Biceps Femoris Long", 
                "Vastus Medialis", "Vastus Lateralis", "Rectus Femoris", 
                "Medial Gastrocnemius", "Tibialis Anterior", "Soleus"]

# Sidebar untuk Load Data & Slider Filter
with st.sidebar:
    st.header("Panel Kontrol")
    uploaded_file = st.file_uploader("LOAD DATA (TXT)", type=["txt"])
    
    cutoff_val = st.slider(
        label="Cutoff Frequency LPF (Hz) EMG",
        min_value=1.0,
        max_value=20.0,
        value=6.0,
        step=0.1
    )

if uploaded_file is not None:
    data_dict, err = load_and_process_data(uploaded_file.getvalue())
    
    if err:
        st.error(err)
    else:
        st.sidebar.success(f"Jumlah Data : {len(data_dict['t'])}")
        
        # Ekstraksi variabel utama
        t = data_dict['t']
        fs = data_dict['fs']
        
        # --- Pre-processing Data FSR & Normalisasi ---
        heel_filt = apply_manual_lpf(data_dict['heel'], fs, cutoff=cutoff_val, order=4)
        toe_filt = apply_manual_lpf(data_dict['toe'], fs, cutoff=cutoff_val, order=4)
        
        heel_norm = normalize_signal(heel_filt)
        toe_norm = normalize_signal(toe_filt)
        
        # --- Deteksi Fase 4 Titik (Menggunakan np.where) ---
        threshold_val = 0.05
        
        heel_rise = np.where((heel_norm[:-1] < threshold_val) & (heel_norm[1:] >= threshold_val))[0] + 1
        heel_fall = np.where((heel_norm[:-1] >= threshold_val) & (heel_norm[1:] < threshold_val))[0] + 1
        toe_rise = np.where((toe_norm[:-1] < threshold_val) & (toe_norm[1:] >= threshold_val))[0] + 1
        toe_fall = np.where((toe_norm[:-1] >= threshold_val) & (toe_norm[1:] < threshold_val))[0] + 1
        
        # --- Perhitungan Parameter Temporal ---
        gait_cycles = []
        for i in range(len(heel_rise) - 1):
            start_idx = int(heel_rise[i])
            end_idx = int(heel_rise[i + 1])
            start_time = float(t[start_idx])
            end_time = float(t[end_idx])
            duration = float(end_time - start_time)

            gait_cycles.append({
                "cycle": i + 1,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "start_time": start_time,
                "end_time": end_time,
                "duration": duration
            })

        temporal_parameters = []
        for cycle in gait_cycles:
            cycle_num = cycle["cycle"]
            start_idx = cycle["start_idx"]
            end_idx = cycle["end_idx"]
            start_time = cycle["start_time"]
            end_time = cycle["end_time"]
            gait_cycle_time = cycle["duration"]

            toe_fall_in_cycle = toe_fall[(toe_fall > start_idx) & (toe_fall < end_idx)]
            if len(toe_fall_in_cycle) == 0:
                continue

            toe_off_idx = int(toe_fall_in_cycle[0])
            toe_off_time = float(t[toe_off_idx])

            stance_time = toe_off_time - start_time
            swing_time = end_time - toe_off_time
            stance_percent = (stance_time / gait_cycle_time) * 100
            swing_percent = (swing_time / gait_cycle_time) * 100

            temporal_parameters.append({
                "cycle": str(cycle_num),
                "start_time": round(start_time, 3),
                "toe_off_time": round(toe_off_time, 3),
                "end_time": round(end_time, 3),
                "gait_cycle_time": round(gait_cycle_time, 3),
                "stance_time": round(stance_time, 3),
                "swing_time": round(swing_time, 3),
                "stance_percent": round(stance_percent, 2),
                "swing_percent": round(swing_percent, 2)
            })

        df_temporal = pd.DataFrame(temporal_parameters)
        
        if not df_temporal.empty:
            avg_vals = df_temporal.mean(numeric_only=True)
            avg_row = {
                "cycle": "Rata-rata",
                "start_time": round(avg_vals["start_time"], 3),
                "toe_off_time": round(avg_vals["toe_off_time"], 3),
                "end_time": round(avg_vals["end_time"], 3),
                "gait_cycle_time": round(avg_vals["gait_cycle_time"], 3),
                "stance_time": round(avg_vals["stance_time"], 3),
                "swing_time": round(avg_vals["swing_time"], 3),
                "stance_percent": round(avg_vals["stance_percent"], 2),
                "swing_percent": round(avg_vals["swing_percent"], 2)
            }
            df_display = pd.concat([df_temporal, pd.DataFrame([avg_row])], ignore_index=True)
            
            mean_cycle = avg_vals['gait_cycle_time']
            cadence = (60.0 / mean_cycle) if mean_cycle > 0 else 0.0
        else:
            df_display = pd.DataFrame()
            mean_cycle = 0
            cadence = 0
            
        st.sidebar.markdown(f"""
        **Temporal Parameters:**
        - Rata-rata Cycle: {mean_cycle:.3f} s
        - Cadence: {cadence:.2f} step/min
        - Jumlah Cycle: {len(gait_cycles)}
        """)

        # TABS SESUAI PYQT
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "GAIT PARAMETERS", "DYNAMIC EMG", "EMG PREPROCESSING", 
            "PARAMETER (TABEL)", "STFT ANALYSIS"
        ])
        
        # ---------------------------------------------------------
        # TAB 1: GAIT PARAMETERS
        # ---------------------------------------------------------
        with tab1:
            # 1. Plot Input (Sumbu X pakai Waktu)
            fig_in, ax_in = plt.subplots(figsize=(10, 2.5))
            ax_in.plot(t, data_dict['heel'], 'b-', label="Heel / FSR Biru", linewidth=1)
            ax_in.plot(t, data_dict['toe'], 'r-', label="Toe / FSR Merah", linewidth=1)
            ax_in.set_title("INPUT (RAW DATA)")
            ax_in.set_xlabel("Waktu (s)"); ax_in.set_ylabel("Amplitude")
            ax_in.legend(); ax_in.grid(True)
            st.pyplot(fig_in, use_container_width=True)
            
            # 2. Plot Output Filter (Sumbu X pakai Waktu)
            fig_out, ax_out = plt.subplots(figsize=(10, 2.5))
            ax_out.plot(t, heel_filt, 'b-', label="Heel Filtering", linewidth=1)
            ax_out.plot(t, toe_filt, 'r-', label="Toe Filtering", linewidth=1)
            ax_out.set_title(f"OUTPUT / HASIL FILTERING (Cutoff: {cutoff_val} Hz)")
            ax_out.set_xlabel("Waktu (s)"); ax_out.set_ylabel("Amplitude")
            ax_out.legend(); ax_out.grid(True)
            st.pyplot(fig_out, use_container_width=True)
            
            # 3. Plot Segmentasi (4 WARNA FASE BERBEDA)
            fig_seg, ax_seg = plt.subplots(figsize=(10, 3.5))
            ax_seg.plot(t, heel_norm, 'purple', label="Heel Normalized", linewidth=1.5)
            ax_seg.plot(t, toe_norm, 'blue', label="Toe Normalized", linewidth=1.5)
            ax_seg.axhline(threshold_val, color='black', linestyle='-', label="Threshold 0.05", alpha=0.5)
            
            for i, idx in enumerate(heel_rise):
                ax_seg.axvline(x=t[idx], color='green', linestyle='--', linewidth=1.2, label='Heel Strike (Naik)' if i==0 else "")
            for i, idx in enumerate(heel_fall):
                ax_seg.axvline(x=t[idx], color='red', linestyle='--', linewidth=1.2, label='Heel Off (Turun)' if i==0 else "")
            for i, idx in enumerate(toe_rise):
                ax_seg.axvline(x=t[idx], color='cyan', linestyle=':', linewidth=1.2, label='Toe Strike (Naik)' if i==0 else "")
            for i, idx in enumerate(toe_fall):
                ax_seg.axvline(x=t[idx], color='orange', linestyle=':', linewidth=1.2, label='Toe Off (Turun)' if i==0 else "")

            ax_seg.set_title("HEEL dan TOE PHASE DETECTION")
            ax_seg.set_xlabel("Waktu (s)"); ax_seg.set_ylabel("Amplitudo")
            ax_seg.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
            ax_seg.grid(True)
            st.pyplot(fig_seg, use_container_width=True)
            
            # 4. Plot Joint Angles (Sumbu X pakai Waktu)
            fig_joint, ax_joint = plt.subplots(figsize=(10, 2.5))
            ax_joint.plot(t, data_dict['hip'], 'r-', label="Hip", linewidth=1)
            ax_joint.plot(t, data_dict['knee'], 'g-', label="Knee", linewidth=1)
            ax_joint.plot(t, data_dict['ankle'], 'b-', label="Ankle", linewidth=1)
            ax_joint.set_title("JOINT ANGLE PARAMETERS")
            ax_joint.set_xlabel("Waktu (s)"); ax_joint.set_ylabel("Degree")
            ax_joint.legend(); ax_joint.grid(True)
            st.pyplot(fig_joint, use_container_width=True)

        # ---------------------------------------------------------
        # TAB 2: DYNAMIC EMG
        # ---------------------------------------------------------
        with tab2:
            offset_raw = 2.0
            offset_env = 1.2
            
            emg_raw = data_dict['emg']
            emg_rect = [[abs(val) for val in m_data] for m_data in emg_raw]
            emg_env = [normalize_signal(apply_manual_lpf(m, fs, cutoff=cutoff_val)) for m in emg_rect]
            
            fig_emg1, ax_emg1 = plt.subplots(figsize=(10, 5))
            for i in range(9):
                ax_emg1.plot(t, [val + i * offset_raw for val in emg_raw[i]], 'r-', linewidth=0.8)
                ax_emg1.text(t[-1], i * offset_raw, muscle_names[i], fontsize=8, verticalalignment='center')
            ax_emg1.set_title("Raw EMG dan LE Software")
            ax_emg1.set_xlabel("Waktu (s)")
            st.pyplot(fig_emg1, use_container_width=True)

            fig_emg2, ax_emg2 = plt.subplots(figsize=(10, 5))
            for i in range(9):
                ax_emg2.plot(t, [val + i * offset_raw for val in emg_rect[i]], 'r-', linewidth=0.8)
            ax_emg2.set_title("Rectification")
            ax_emg2.set_xlabel("Waktu (s)")
            st.pyplot(fig_emg2, use_container_width=True)

            fig_emg3, ax_emg3 = plt.subplots(figsize=(10, 5))
            for i in range(9):
                ax_emg3.plot(t, [val + i * offset_env for val in emg_env[i]], 'g-', linewidth=1.5)
            ax_emg3.set_title(f"Enveloped Filter (Cutoff: {cutoff_val} Hz)")
            ax_emg3.set_xlabel("Waktu (s)")
            st.pyplot(fig_emg3, use_container_width=True)
            
            fig_act, ax_act = plt.subplots(figsize=(10, 5))
            for i in range(9):
                segments = detect_activation_segments_manual(t, emg_env[i], 0.05)
                bar_data = [(start, end - start) for start, end in segments]
                y_pos = 8 - i
                ax_act.broken_barh(bar_data, (y_pos - 0.25, 0.5), facecolors='#1f77b4')
                
            ax_act.set_yticks(list(range(9)))
            ax_act.set_yticklabels(muscle_names[::-1])
            ax_act.set_title("Muscle activation each cycle")
            ax_act.set_xlabel("Waktu (s)")
            ax_act.grid(axis='x', linestyle=':')
            st.pyplot(fig_act, use_container_width=True)

        # ---------------------------------------------------------
        # TAB 3: EMG PREPROCESSING
        # ---------------------------------------------------------
        with tab3:
            otot_pilihan = st.selectbox("Pilih Otot untuk dianalisis pada proses Preprocessing:", muscle_names)
            idx_otot = muscle_names.index(otot_pilihan)
            
            st.markdown(f"Menampilkan komparasi untuk otot **{otot_pilihan}**")
            
            fig_pre_raw, ax_pre_raw = plt.subplots(figsize=(10, 3))
            ax_pre_raw.plot(t, emg_raw[idx_otot], 'k-', linewidth=1)
            ax_pre_raw.set_title("RAW EMG SIGNAL")
            ax_pre_raw.set_ylabel("EMG (mv)"); ax_pre_raw.set_xlabel("Waktu (s)")
            st.pyplot(fig_pre_raw, use_container_width=True)
            
            fig_pre_res, ax_pre_res = plt.subplots(figsize=(10, 3))
            ax_pre_res.plot(t, emg_rect[idx_otot], color='gray', label="Rectified", alpha=0.5)
            ax_pre_res.plot(t, apply_manual_lpf(emg_rect[idx_otot], fs, cutoff=cutoff_val), color='red', label="Low-pass Filtered", linewidth=2)
            ax_pre_res.set_title(f"PREPROCESSED EMG (RECTIFIED & LPF {cutoff_val} Hz)")
            ax_pre_res.set_ylabel("Processed EMG (mv)"); ax_pre_res.set_xlabel("Waktu (s)")
            ax_pre_res.legend()
            st.pyplot(fig_pre_res, use_container_width=True)

        # ---------------------------------------------------------
        # TAB 4: PARAMETER (TABEL)
        # ---------------------------------------------------------
        with tab4:
            st.subheader("Temporal Parameters (Detailed per Cycle)")
            # Menampilkan tabel baru yang ada rata-ratanya
            st.dataframe(df_display, use_container_width=True)
            
            st.markdown("---")
            
            # Tetap mempertahankan tabel Joint Angle aslinya
            st.subheader("Joint Angle Parameters")
            joint_sel = st.selectbox("Pilih Joint:", ["hip", "knee", "ankle"])
            sig_j = data_dict[joint_sel]
            max_v, min_v = max(sig_j), min(sig_j)
            rom_v = max_v - min_v
            
            st.table({
                "Parameter": ["Max (deg)", "Min (deg)", "ROM (deg)"],
                "Nilai": [f"{max_v:.2f}", f"{min_v:.2f}", f"{rom_v:.2f}"]
            })

        # ---------------------------------------------------------
        # TAB 5: STFT ANALYSIS
        # ---------------------------------------------------------
        with tab5:
            stft_opts = ["heel", "toe", "hip", "knee", "ankle"] + muscle_names
            sel_stft = st.selectbox("Pilih Sinyal untuk Spectrogram:", stft_opts)
            
            if sel_stft in ["heel", "toe", "hip", "knee", "ankle"]:
                sig_stft = data_dict[sel_stft]
            else:
                idx = muscle_names.index(sel_stft)
                sig_stft = emg_raw[idx]
            
            freqs, times_stft, power_matrix = compute_stft_manual(sig_stft, fs, nperseg=128)
            
            fig_stft, ax_stft = plt.subplots(figsize=(10, 4))
            c = ax_stft.pcolormesh(times_stft, freqs, power_matrix, shading='gouraud', cmap='viridis')
            fig_stft.colorbar(c, ax=ax_stft, label="Power")
            ax_stft.set_title("STFT Spectrogram")
            ax_stft.set_ylabel("Frequency (Hz)")
            ax_stft.set_xlabel("Waktu (s)")
            ax_stft.set_ylim(0, 11)
            st.pyplot(fig_stft, use_container_width=True)
else:
    st.info("Silakan unggah file data .txt dari menu sebelah kiri untuk memulai analisis.")
