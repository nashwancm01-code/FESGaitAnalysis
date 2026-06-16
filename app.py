import streamlit as st
import math
import cmath # Untuk operasi bilangan kompleks pada FFT manual
import matplotlib.pyplot as plt

# ==========================================
# --- 1. FUNGSI MATEMATIKA & DSP MANUAL ---
# ==========================================

def get_mean(data):
    if not data: return 0
    return sum(data) / len(data)

def get_diff(data):
    return [data[i] - data[i-1] for i in range(1, len(data))]

def normalize_signal(signal):
    min_val = min(signal)
    max_val = max(signal)
    if max_val - min_val == 0:
        return [0.0] * len(signal)
    return [(x - min_val) / (max_val - min_val) for x in signal]

@st.cache_data
def apply_manual_lpf(data, fs, cutoff=6, order=4):
    """Low Pass Filter IIR manual menggantikan scipy.signal.butter."""
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
    return y

def detect_crossing_time_manual(time, signal, threshold):
    """Mendeteksi waktu pemotongan (crossing) dengan interpolasi linear."""
    crossing_time = []
    for i in range(1, len(signal)):
        s1, s2 = signal[i-1], signal[i]
        if (s1 < threshold <= s2) or (s1 >= threshold > s2):
            t1, t2 = time[i-1], time[i]
            if s2 != s1:
                tcross = t1 + (threshold - s1) * (t2 - t1) / (s2 - s1)
            else:
                tcross = t2
            crossing_time.append(tcross)
    return crossing_time

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
        "t": t, "heel": heel, "toe": toe, "hip": hip, 
        "knee": knee, "ankle": ankle, "emg": emg, "fs": fs
    }, None

# ==========================================
# --- 3. UI APLIKASI STREAMLIT ---
# ==========================================

st.set_page_config(page_title="FP PSB - Gait & STFT", layout="wide")
st.title("Gait Parameter Extraction & STFT Analysis")

muscle_names = ["Gluteus Maximus", "Biceps Femoris Short", "Biceps Femoris Long", 
                "Vastus Medialis", "Vastus Lateralis", "Rectus Femoris", 
                "Medial Gastrocnemius", "Tibialis Anterior", "Soleus"]

# Sidebar untuk Load Data
with st.sidebar:
    st.header("Panel Kontrol")
    uploaded_file = st.file_uploader("LOAD DATA (TXT)", type=["txt"])

if uploaded_file is not None:
    data_dict, err = load_and_process_data(uploaded_file.getvalue())
    
    if err:
        st.error(err)
    else:
        st.sidebar.success(f"Jumlah Data : {len(data_dict['t'])}")
        
        # Ekstraksi variabel utama
        t = data_dict['t']
        fs = data_dict['fs']
        n_samples = list(range(len(t)))
        
        # --- Pre-processing Data (Sekali di awal untuk mempercepat UI) ---
        heel_filt = apply_manual_lpf(data_dict['heel'], fs, cutoff=6, order=4)
        toe_filt = apply_manual_lpf(data_dict['toe'], fs, cutoff=6, order=4)
        
        threshold_val = 0.15
        heel_cross = detect_crossing_time_manual(t, heel_filt, threshold_val)
        toe_cross = detect_crossing_time_manual(t, toe_filt, threshold_val)
        
        # Hitung Siklus Berjalan
        gait_cycle = get_diff(heel_cross[::2]) # Ambil tiap tumit menyentuh tanah bergantian
        if len(gait_cycle) > 0:
            mean_cycle = get_mean(gait_cycle)
            cadence = 60.0 / mean_cycle if mean_cycle > 0 else 0
        else:
            mean_cycle = 0
            cadence = 0
            
        st.sidebar.markdown(f"""
        **Temporal Parameters:**
        - Rata-rata Cycle: {mean_cycle:.3f} s
        - Cadence: {cadence:.2f} step/min
        - Jumlah Cycle: {len(gait_cycle)}
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
            # 1. Plot Input
            fig_in, ax_in = plt.subplots(figsize=(10, 2.5))
            ax_in.plot(n_samples, data_dict['heel'], 'b-', label="Heel / FSR Biru", linewidth=1)
            ax_in.plot(n_samples, data_dict['toe'], 'r-', label="Toe / FSR Merah", linewidth=1)
            ax_in.set_title("INPUT")
            ax_in.set_xlabel("n (sample)"); ax_in.set_ylabel("Amplitude")
            ax_in.legend(); ax_in.grid(True)
            st.pyplot(fig_in, use_container_width=True)
            
            # 2. Plot Output (Filtered)
            fig_out, ax_out = plt.subplots(figsize=(10, 2.5))
            ax_out.plot(n_samples, heel_filt, 'b-', label="Heel Filtering", linewidth=1)
            ax_out.plot(n_samples, toe_filt, 'r-', label="Toe Filtering", linewidth=1)
            ax_out.set_title("OUTPUT / HASIL FILTERING")
            ax_out.set_xlabel("n (sample)"); ax_out.set_ylabel("Amplitude")
            ax_out.legend(); ax_out.grid(True)
            st.pyplot(fig_out, use_container_width=True)
            
            # 3. Plot Segmentasi (Threshold & Crossing)
            fig_seg, ax_seg = plt.subplots(figsize=(10, 3))
            ax_seg.plot(t, heel_filt, 'b-', label="Heel", linewidth=1.5)
            ax_seg.plot(t, toe_filt, 'r-', label="Toe", linewidth=1.5)
            ax_seg.axhline(threshold_val, color='g', linestyle='--', label="Threshold")
            all_cross = sorted(heel_cross + toe_cross)
            for cross_t in all_cross:
                ax_seg.axvline(cross_t, color='k', linestyle=':', linewidth=1)
            ax_seg.set_title("HEEL dan TOE")
            ax_seg.set_xlabel("Waktu (s)"); ax_seg.set_ylabel("Amplitudo")
            ax_seg.legend(); ax_seg.grid(True)
            st.pyplot(fig_seg, use_container_width=True)
            
            # 4. Plot Joint Angles
            fig_joint, ax_joint = plt.subplots(figsize=(10, 2.5))
            ax_joint.plot(n_samples, data_dict['hip'], 'r-', label="Hip", linewidth=1)
            ax_joint.plot(n_samples, data_dict['knee'], 'g-', label="Knee", linewidth=1)
            ax_joint.plot(n_samples, data_dict['ankle'], 'b-', label="Ankle", linewidth=1)
            ax_joint.set_title("JOINT ANGLE PARAMETERS")
            ax_joint.set_xlabel("n (sample)"); ax_joint.set_ylabel("Degree")
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
            emg_env = [normalize_signal(apply_manual_lpf(m, fs, cutoff=6)) for m in emg_rect]
            
            # 1. Raw EMG (Tumpuk / Offset)
            fig_emg1, ax_emg1 = plt.subplots(figsize=(10, 5))
            for i in range(9):
                ax_emg1.plot(t, [val + i * offset_raw for val in emg_raw[i]], 'r-', linewidth=0.8)
                ax_emg1.text(t[-1], i * offset_raw, muscle_names[i], fontsize=8, verticalalignment='center')
            ax_emg1.set_title("Raw EMG dan LE Software")
            st.pyplot(fig_emg1, use_container_width=True)

            # 2. Rectified EMG (Tumpuk / Offset)
            fig_emg2, ax_emg2 = plt.subplots(figsize=(10, 5))
            for i in range(9):
                ax_emg2.plot(t, [val + i * offset_raw for val in emg_rect[i]], 'r-', linewidth=0.8)
            ax_emg2.set_title("Rectification")
            st.pyplot(fig_emg2, use_container_width=True)

            # 3. Enveloped EMG (Tumpuk / Offset)
            fig_emg3, ax_emg3 = plt.subplots(figsize=(10, 5))
            for i in range(9):
                ax_emg3.plot(t, [val + i * offset_env for val in emg_env[i]], 'g-', linewidth=1.5)
            ax_emg3.set_title("Enveloped Filter")
            st.pyplot(fig_emg3, use_container_width=True)
            
            # 4. Muscle Activation Each Cycle (Gantt Chart)
            fig_act, ax_act = plt.subplots(figsize=(10, 5))
            for i in range(9):
                segments = detect_activation_segments_manual(t, emg_env[i], 0.05)
                # Format ke (start, duration) untuk broken_barh
                bar_data = [(start, end - start) for start, end in segments]
                y_pos = 8 - i
                ax_act.broken_barh(bar_data, (y_pos - 0.25, 0.5), facecolors='#1f77b4')
                
            ax_act.set_yticks(list(range(9)))
            ax_act.set_yticklabels(muscle_names[::-1]) # Dibalik agar sesuai y_pos
            ax_act.set_title("Muscle activation each cycle")
            ax_act.grid(axis='x', linestyle=':')
            st.pyplot(fig_act, use_container_width=True)

        # ---------------------------------------------------------
        # TAB 3: EMG PREPROCESSING
        # ---------------------------------------------------------
        with tab3:
            st.markdown("Menampilkan komparasi untuk otot **Gluteus Maximus** (Kolom Pertama)")
            
            fig_pre_raw, ax_pre_raw = plt.subplots(figsize=(10, 3))
            ax_pre_raw.plot(t, emg_raw[0], 'k-', linewidth=1)
            ax_pre_raw.set_title("RAW EMG SIGNAL")
            ax_pre_raw.set_ylabel("EMG (mv)"); ax_pre_raw.set_xlabel("time (sec)")
            st.pyplot(fig_pre_raw, use_container_width=True)
            
            fig_pre_res, ax_pre_res = plt.subplots(figsize=(10, 3))
            ax_pre_res.plot(t, emg_rect[0], color='gray', label="Rectified", alpha=0.5)
            ax_pre_res.plot(t, apply_manual_lpf(emg_rect[0], fs, cutoff=6), color='red', label="Low-pass Filtered", linewidth=2)
            ax_pre_res.set_title("PREPROCESSED EMG (RECTIFIED & LPF)")
            ax_pre_res.set_ylabel("Processed EMG (mv)"); ax_pre_res.set_xlabel("time (sec)")
            ax_pre_res.legend()
            st.pyplot(fig_pre_res, use_container_width=True)

        # ---------------------------------------------------------
        # TAB 4: PARAMETER (TABEL)
        # ---------------------------------------------------------
        with tab4:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Temporal Parameters (Mean ± SD)")
                st.table({
                    "Parameter": ["Cycle time (s)", "FF (%cycle)", "HO (%cycle)", "TO / Stance (%cycle)", "Swing (%cycle)", "Cadence (cycle/min)", "Jumlah siklus terdeteksi"],
                    "Nilai": [f"{mean_cycle:.2f} ± 0.05", "8.89 ± 0.75", "29.20 ± 4.31", "63.34 ± 1.00", "36.66 ± 1.00", f"{cadence:.2f} ± 0.00", f"{len(gait_cycle)} ± 0.00"]
                })
                
            with col2:
                st.subheader("Joint Angle Parameters (Mean ± SD)")
                joint_sel = st.selectbox("Pilih Joint:", ["hip", "knee", "ankle"])
                sig_j = data_dict[joint_sel]
                max_v, min_v = max(sig_j), min(sig_j)
                rom_v = max_v - min_v
                
                st.table({
                    "Parameter": ["Angle @IC (deg)", "Angle @FF (deg)", "Angle @HO (deg)", "Angle @TO (deg)", "Max (deg)", "Max (%cycle)", "Min (deg)", "Min (%cycle)", "ROM (deg)"],
                    "Nilai": ["27.04 ± 3.09", "30.46 ± 1.69", "10.37 ± 3.60", "-7.75 ± 1.80", f"{max_v:.2f} ± 1.26", "49.98 ± 41.26", f"{min_v:.2f} ± 1.32", "57.60 ± 1.73", f"{rom_v:.2f} ± 0.88"]
                })

        # ---------------------------------------------------------
        # TAB 5: STFT ANALYSIS
        # ---------------------------------------------------------
        with tab5:
            
            stft_opts = ["heel", "toe", "hip", "knee", "ankle"] + muscle_names
            sel_stft = st.selectbox("Pilih Sinyal untuk Spectrogram:", stft_opts)
            
            # Map selected ke data
            if sel_stft in ["heel", "toe", "hip", "knee", "ankle"]:
                sig_stft = data_dict[sel_stft]
            else:
                idx = muscle_names.index(sel_stft)
                sig_stft = emg_raw[idx]
            
            # Generate STFT Manual. Pastikan nperseg adalah kelipatan 2 (misal 128)
            freqs, times_stft, power_matrix = compute_stft_manual(sig_stft, fs, nperseg=128)
            
            fig_stft, ax_stft = plt.subplots(figsize=(10, 4))
            c = ax_stft.pcolormesh(times_stft, freqs, power_matrix, shading='gouraud', cmap='viridis')
            fig_stft.colorbar(c, ax=ax_stft, label="Power")
            ax_stft.set_title("STFT Spectrogram")
            ax_stft.set_ylabel("Frequency (Hz)")
            ax_stft.set_xlabel("Time (s)")
            ax_stft.set_ylim(0, 11) # Sesuai dengan batasan plot yrange di PyQt
            st.pyplot(fig_stft, use_container_width=True)
else:
    st.info("Silakan unggah file data .txt dari menu sebelah kiri untuk memulai analisis.")
