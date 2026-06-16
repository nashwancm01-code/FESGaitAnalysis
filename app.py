import streamlit as st
import math
import matplotlib.pyplot as plt
import pandas as pd # Hanya untuk st.dataframe visualisasi tabel, tidak untuk kalkulasi

# ==========================================
# --- 1. FUNGSI MATEMATIKA & DSP MANUAL ---
# ==========================================

@st.cache_data
def apply_manual_lpf(data, dt, cutoff, order):
    """Low Pass Filter IIR manual tanpa scipy/numpy."""
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

def get_max_value(data_list):
    """Mencari nilai maksimum manual."""
    max_val = data_list[0]
    for val in data_list:
        if val > max_val:
            max_val = val
    return max_val

def normalize_to_100_percent(data_segment):
    """Linear interpolation manual ke 101 titik (0% - 100%) tanpa numpy.interp."""
    if not data_segment: return []
    n = len(data_segment)
    if n == 1: return [data_segment[0]] * 101
    
    res = []
    for i in range(101):
        pos = i * (n - 1) / 100.0
        idx = int(pos)
        frac = pos - idx
        if idx >= n - 1:
            res.append(data_segment[-1])
        else:
            res.append(data_segment[idx] + frac * (data_segment[idx+1] - data_segment[idx]))
    return res

def find_threshold_crossings_up(data, threshold):
    """Mendeteksi indeks saat sinyal naik melewati threshold (untuk Heel Strike)."""
    crossings = []
    for i in range(1, len(data)):
        if data[i-1] <= threshold < data[i]:
            crossings.append(i)
    return crossings

# ==========================================
# --- 2. FUNGSI LOAD DATA ---
# ==========================================

@st.cache_data
def load_and_process_data(file_bytes):
    column_names = [
        "time", "heel", "toe", "hip", "knee", "ankle", 
        "gluteus maximus", "biceps femoris short", "biceps femoris long", 
        "vastus medialis", "vastus lateralis", "rectus femoris", 
        "soleus", "gastrocnemius", "tibialis anterior"
    ]
    raw_text = file_bytes.decode('utf-8').splitlines()
    parsed_data = {col: [] for col in column_names}
    
    for line in raw_text:
        parts = line.split()
        if len(parts) >= len(column_names):
            try:
                for idx, col in enumerate(column_names):
                    parsed_data[col].append(float(parts[idx]))
            except ValueError: continue
            
    dt = (parsed_data["time"][1] - parsed_data["time"][0]) if len(parsed_data["time"]) > 1 else 0.001
    emg_cols = column_names[6:15]
    
    # Penyearahan gelombang (Rectification) manual (Absolute)
    rect_dict = {col: [abs(x) for x in parsed_data[col]] for col in emg_cols}
    return parsed_data, dt, emg_cols, rect_dict

# ==========================================
# --- 3. UI APLIKASI STREAMLIT ---
# ==========================================

st.set_page_config(page_title="Gait & EMG Analysis", layout="wide")
st.title("Aplikasi Pemrosesan Sinyal Biomekanika (Pure Python)")

uploaded_file = st.file_uploader("Unggah file data (.txt)", type=["txt"])

if uploaded_file is not None:
    parsed_data, dt, emg_cols, rect_dict = load_and_process_data(uploaded_file.getvalue())
    
    with st.expander("Lihat Data Mentah (Preview)"):
        st.dataframe(pd.DataFrame(parsed_data).head(10))

    tab1, tab2 = st.tabs(["Grafik EMG & Aktivasi", "Gait Parameters & Kinematics"])
    
    # ---------------------------------------------------------
    # TAB 1: EMG ANALYSIS
    # ---------------------------------------------------------
    with tab1:
        st.header("Analisis Sinyal EMG")
        col1, col2 = st.columns([1, 3])
        with col1:
            cutoff_emg = st.slider("Cutoff Frequency LPF (Hz) EMG", 0.5, 20.0, 5.0, key='emg_cutoff')
            muscle = st.selectbox("Pilih Otot untuk Divisualisasikan:", emg_cols)
        
        # Proses Data EMG
        raw_emg = parsed_data[muscle]
        rect_emg = rect_dict[muscle]
        lpf_emg = apply_manual_lpf(rect_emg, dt, cutoff_emg, 2)
        
        # Thresholding 5% Manual
        max_emg = get_max_value(lpf_emg)
        threshold_emg = max_emg * 0.05
        activation_emg = [max_emg if val > threshold_emg else 0 for val in lpf_emg] # Square wave untuk visualisasi
        
        # Plot 1: Raw Signal
        st.subheader(f"1. Raw Signal: {muscle.title()}")
        fig_raw, ax_raw = plt.subplots(figsize=(10, 3))
        ax_raw.plot(parsed_data['time'], raw_emg, color='gray', linewidth=0.8)
        ax_raw.set_ylabel("Amplitudo")
        st.pyplot(fig_raw)
        
        # Plot 2: Rectified & LPF
        st.subheader("2. Rectified & LPF Signal")
        fig_lpf, ax_lpf = plt.subplots(figsize=(10, 3))
        ax_lpf.plot(parsed_data['time'], rect_emg, color='lightblue', alpha=0.6, label="Rectified")
        ax_lpf.plot(parsed_data['time'], lpf_emg, color='red', linewidth=1.5, label="LPF (Envelope)")
        ax_lpf.legend()
        ax_lpf.set_ylabel("Amplitudo")
        st.pyplot(fig_lpf)
        
        # Plot 3: Thresholding 5%
        st.subheader("3. Muscle Activation (Threshold 5%)")
        fig_thresh, ax_thresh = plt.subplots(figsize=(10, 3))
        ax_thresh.plot(parsed_data['time'], lpf_emg, color='red', label="Filtered Signal")
        ax_thresh.axhline(threshold_emg, color='green', linestyle='--', label=f"Threshold 5% ({threshold_emg:.3f})")
        ax_thresh.fill_between(parsed_data['time'], 0, activation_emg, color='orange', alpha=0.3, label="Active Phase")
        ax_thresh.legend()
        ax_thresh.set_xlabel("Waktu (s)")
        st.pyplot(fig_thresh)

    # ---------------------------------------------------------
    # TAB 2: GAIT ANALYSIS
    # ---------------------------------------------------------
    with tab2:
        st.header("Analisis Gait & Kinematika")
        cutoff_gait = st.slider("Cutoff Frequency LPF (Hz) Gait", 1.0, 50.0, 10.0, key='gait_cutoff')
        
        time_data = parsed_data['time']
        raw_heel = parsed_data['heel']
        raw_toe = parsed_data['toe']
        
        # Proses LPF Gait Manual
        filt_heel = apply_manual_lpf(raw_heel, dt, cutoff_gait, 2)
        filt_toe = apply_manual_lpf(raw_toe, dt, cutoff_gait, 2)
        
        # Plot 1: Input Signal (Raw)
        st.subheader("1. Input Signal (Raw Heel & Toe)")
        fig_g1, ax_g1 = plt.subplots(figsize=(12, 3))
        ax_g1.plot(time_data, raw_heel, label='Heel Raw', color='blue', alpha=0.5)
        ax_g1.plot(time_data, raw_toe, label='Toe Raw', color='red', alpha=0.5)
        ax_g1.legend()
        st.pyplot(fig_g1)
        
        # Plot 2: Filtered Output
        st.subheader("2. Filtered Output")
        fig_g2, ax_g2 = plt.subplots(figsize=(12, 3))
        ax_g2.plot(time_data, filt_heel, label='Heel Filtered', color='blue')
        ax_g2.plot(time_data, filt_toe, label='Toe Filtered', color='red')
        ax_g2.legend()
        st.pyplot(fig_g2)
        
        # Plot 3: Threshold 5% & Phases Detection
        st.subheader("3. Normalisasi Threshold 5% & Phase Detection")
        thresh_heel_val = get_max_value(filt_heel) * 0.05
        thresh_toe_val = get_max_value(filt_toe) * 0.05
        
        fig_g3, ax_g3 = plt.subplots(figsize=(12, 4))
        ax_g3.plot(time_data, filt_heel, label='Heel', color='blue')
        ax_g3.plot(time_data, filt_toe, label='Toe', color='red')
        ax_g3.axhline(thresh_heel_val, color='green', linestyle='--', linewidth=1, label='Threshold Heel')
        ax_g3.axhline(thresh_toe_val, color='lightgreen', linestyle='--', linewidth=1, label='Threshold Toe')
        
        # Mencari titik Heel Strike (Fase Gait)
        heel_strikes_idx = find_threshold_crossings_up(filt_heel, thresh_heel_val)
        for idx in heel_strikes_idx:
            ax_g3.axvline(time_data[idx], color='black', linestyle=':', linewidth=1)
            
        ax_g3.legend()
        ax_g3.grid(True)
        st.pyplot(fig_g3)
        
        # SEGMENTASI KE 0-100% CYCLES
        if len(heel_strikes_idx) >= 2:
            st.markdown("---")
            st.subheader("4. Segmentasi Tiap Siklus (Joint Angles 0-100%)")
            
            # Ambil satu siklus (dari Heel Strike 1 ke Heel Strike 2)
            start_idx = heel_strikes_idx[0]
            end_idx = heel_strikes_idx[1]
            
            hip_seg = parsed_data['hip'][start_idx:end_idx]
            knee_seg = parsed_data['knee'][start_idx:end_idx]
            ankle_seg = parsed_data['ankle'][start_idx:end_idx]
            heel_seg = filt_heel[start_idx:end_idx]
            toe_seg = filt_toe[start_idx:end_idx]
            
            # Interpolasi manual ke 100%
            hip_norm = normalize_to_100_percent(hip_seg)
            knee_norm = normalize_to_100_percent(knee_seg)
            ankle_norm = normalize_to_100_percent(ankle_seg)
            heel_norm = normalize_to_100_percent(heel_seg)
            toe_norm = normalize_to_100_percent(toe_seg)
            percent_axis = list(range(101))
            
            # Plot Joint Angles
            fig_j, ax_j = plt.subplots(1, 3, figsize=(15, 4))
            ax_j[0].plot(percent_axis, hip_norm, color='purple'); ax_j[0].set_title("Hip Joint Angle"); ax_j[0].grid(True)
            ax_j[1].plot(percent_axis, knee_norm, color='teal'); ax_j[1].set_title("Knee Joint Angle"); ax_j[1].grid(True)
            ax_j[2].plot(percent_axis, ankle_norm, color='darkorange'); ax_j[2].set_title("Ankle Joint Angle"); ax_j[2].grid(True)
            for ax in ax_j: ax.set_xlabel("% Gait Cycle")
            st.pyplot(fig_j)
            
            # Plot Akhir: Gait Phase Diagram
            st.subheader("5. Gait Phase Summary")
            fig_phase, ax_phase = plt.subplots(figsize=(10, 3))
            ax_phase.plot(percent_axis, heel_norm, label='Heel', color='blue')
            ax_phase.plot(percent_axis, toe_norm, label='Toe', color='red')
            ax_phase.axhline(thresh_heel_val, color='green', linestyle='--', linewidth=1)
            ax_phase.set_title("Satu Siklus Penuh Gait Berjalan (0 - 100%)")
            ax_phase.set_xlabel("% Gait Cycle")
            ax_phase.legend()
            ax_phase.grid(True)
            st.pyplot(fig_phase)
            
        else:
            st.warning("Data terlalu pendek untuk mendeteksi siklus berjalan (butuh minimal 2 ketukan tumit/Heel Strikes).")
