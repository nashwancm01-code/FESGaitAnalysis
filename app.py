import streamlit as st
import pandas as pd
import math
import matplotlib.pyplot as plt

st.set_page_config(page_title="Biomechanics Analysis", layout="wide")

# ==========================================
# FUNGSI PEMROSESAN MANUAL (PURE PYTHON)
# ==========================================

@st.cache_data
def manual_lowpass_filter(data, dt, cutoff, order=2):
    """Filter LPF Rekursif Manual tanpa Scipy"""
    if cutoff <= 0 or order < 1 or not data: return data
    fc_adj = cutoff / math.sqrt(2**(1.0 / order) - 1.0)
    tau = 1.0 / (2.0 * math.pi * fc_adj)
    alpha = dt / (tau + dt)
    
    y = list(data)
    for _ in range(order):
        y_new = [y[0]]
        for i in range(1, len(y)):
            y_new.append(alpha * y[i] + (1.0 - alpha) * y_new[-1])
        y = y_new
    return y

def manual_normalize_max(data):
    """Normalisasi manual ke rentang 0-1 berdasarkan nilai mutlak maksimum"""
    max_val = max([abs(x) for x in data])
    if max_val == 0: return data
    return [x / max_val for x in data]

def detect_gait_cycles_manual(heel_norm, threshold, dt, cooldown=0.4):
    """Deteksi fasa rising edge (Heel Strike) dengan pengunci waktu manual"""
    hs_indices = []
    for i in range(1, len(heel_norm)):
        if heel_norm[i-1] < threshold and heel_norm[i] >= threshold:
            # Gunakan cooldown agar tidak double-detect noise
            if not hs_indices or (i - hs_indices[-1]) * dt > cooldown:
                hs_indices.append(i)
    return hs_indices

def manual_interpolate_100(vector):
    """Interpolasi linear manual untuk mengubah siklus menjadi persis 100 poin"""
    N = len(vector)
    if N < 2: return [0.0] * 100
    out = []
    for i in range(100):
        frac_idx = (i / 99.0) * (N - 1)
        idx_low = int(math.floor(frac_idx))
        idx_high = int(math.ceil(frac_idx))
        weight = frac_idx - idx_low
        val = (1 - weight) * vector[idx_low] + weight * vector[idx_high]
        out.append(val)
    return out

def find_column(columns, keywords):
    """Pencari nama kolom fleksibel"""
    for col in columns:
        if any(kw.lower() in col.lower() for kw in keywords):
            return col
    return None

# ==========================================
# UI APLIKASI
# ==========================================
st.title("🏃‍♀️ Integrated Biomechanics Analysis App")
st.markdown("Membaca **satu file data flat** yang berisi gabungan EMG dan Gait Sensor.")

uploaded_file = st.file_uploader("Upload File Data (.txt atau .csv)", type=["txt", "csv"])

if uploaded_file is not None:
    # Membaca data menggunakan Pandas hanya untuk data frame, prosesnya murni Python
    df = pd.read_csv(uploaded_file, sep='\t')
    if len(df.columns) < 3:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, sep=',')
        
    st.success("Data berhasil dimuat!")
    columns = list(df.columns)
    
    tab1, tab2 = st.tabs(["📊 Tugas 1: Analisis EMG", "🚶‍♀️ Tugas 2: Ekstraksi Parameter Gait"])
    
    # ==========================================
    # TAB 1: EMG
    # ==========================================
    with tab1:
        st.header("Analisis Sinyal EMG")
        time_col = find_column(columns, ['time', 'waktu', 'detik'])
        if time_col is None:
            st.warning("Tidak menemukan kolom waktu.")
        else:
            t = df[time_col].tolist()
            exclude_kws = ['time', 'waktu', 'detik', 'hip', 'knee', 'ankle', 'heel', 'toe', 'fsr']
            muscle_cols = [c for c in columns if not any(kw in c.lower() for kw in exclude_kws)]
            
            if muscle_cols:
                selected_muscle = st.selectbox("Pilih Otot untuk Dilihat:", muscle_cols)
                raw_muscle = df[selected_muscle].tolist()
                
                fig_emg, ax_emg = plt.subplots(figsize=(10, 3))
                ax_emg.plot(t, raw_muscle, color='#333333', linewidth=0.7)
                ax_emg.set_title(f"RAW EMG: {selected_muscle}")
                st.pyplot(fig_emg)
            else:
                st.info("Tidak ada kolom otot terdeteksi.")

    # ==========================================
    # TAB 2: GAIT ANALYSIS MANUAL
    # ==========================================
    with tab2:
        st.header("Ekstraksi Parameter Gait (Manual Processing)")
        
        time_col = find_column(columns, ['time', 'waktu', 'detik'])
        heel_col = find_column(columns, ['heel'])
        toe_col = find_column(columns, ['toe'])
        hip_col = find_column(columns, ['hip'])
        knee_col = find_column(columns, ['knee'])
        ankle_col = find_column(columns, ['ankle'])
        
        if not all([time_col, heel_col, toe_col, hip_col, knee_col, ankle_col]):
            st.error("Kolom (Time, Heel, Toe, Hip, Knee, Ankle) tidak lengkap.")
        else:
            # Konversi ke standard Python List untuk diolah manual
            t = df[time_col].tolist()
            raw_heel = df[heel_col].tolist()
            raw_toe = df[toe_col].tolist()
            raw_hip = df[hip_col].tolist()
            raw_knee = df[knee_col].tolist()
            raw_ankle = df[ankle_col].tolist()
            
            dt = t[1] - t[0] if len(t) > 1 else 0.01
            fs = 1.0 / dt
            
            st.divider()
            
            # --- STEP 1: RAW SIGNAL ---
            st.subheader("1. Grafik Raw Input Signal")
            fig1, ax1 = plt.subplots(figsize=(12, 3))
            ax1.plot(t, raw_heel, label='Raw Heel', color='blue', alpha=0.7)
            ax1.plot(t, raw_toe, label='Raw Toe', color='red', alpha=0.7)
            ax1.set_xlabel('Time (s)')
            ax1.legend()
            st.pyplot(fig1)
            
            # --- STEP 2: FILTERED SIGNAL ---
            st.subheader("2. Grafik Filtered Output Signal (Manual LPF)")
            # Menggunakan LPF manual buatan sendiri, bukan scipy
            filt_heel = manual_lowpass_filter(raw_heel, dt, cutoff=5.0, order=2)
            filt_toe = manual_lowpass_filter(raw_toe, dt, cutoff=5.0, order=2)
            
            fig2, ax2 = plt.subplots(figsize=(12, 3))
            ax2.plot(t, filt_heel, label='Filtered Heel', color='blue')
            ax2.plot(t, filt_toe, label='Filtered Toe', color='red')
            ax2.set_xlabel('Time (s)')
            ax2.legend()
            st.pyplot(fig2)
            
            # --- STEP 3: NORMALIZATION & THRESHOLD 5% ---
            st.subheader("3. Normalisasi → Threshold 5% → Phases Detection")
            norm_heel = manual_normalize_max(filt_heel)
            norm_toe = manual_normalize_max(filt_toe)
            
            threshold = 0.05
            hs_indices = detect_gait_cycles_manual(norm_heel, threshold, dt)
            
            fig3, ax3 = plt.subplots(figsize=(12, 3))
            ax3.plot(t, norm_heel, label='Normalized Heel', color='blue')
            ax3.plot(t, norm_toe, label='Normalized Toe', color='red')
            ax3.axhline(y=threshold, color='green', linestyle='--', label='5% Threshold')
            
            # Titik-titik Heel Strike Manual
            t_hs = [t[i] for i in hs_indices]
            y_hs = [norm_heel[i] for i in hs_indices]
            ax3.scatter(t_hs, y_hs, color='black', zorder=5, label='Heel Strike Detected')
            ax3.legend()
            st.pyplot(fig3)
            
            # --- STEP 4: SEGMENTASI PER CYCLE ---
            st.subheader("4. Segmentasi Tiap Cycle")
            fig4, ax4 = plt.subplots(figsize=(12, 3))
            ax4.plot(t, norm_heel, color='blue', alpha=0.5)
            ax4.plot(t, norm_toe, color='red', alpha=0.5)
            for idx in hs_indices:
                ax4.axvline(x=t[idx], color='black', linestyle='-.', alpha=0.7)
            ax4.set_title("Gait Cycles Segmented by Vertical Lines")
            st.pyplot(fig4)
            
            # --- STEP 5: TEMPORAL PARAMETERS ---
            st.subheader("5. Temporal Parameters")
            num_cycles = len(hs_indices) - 1
            if num_cycles > 0:
                cycle_times = [(t[hs_indices[i+1]] - t[hs_indices[i]]) for i in range(num_cycles)]
                avg_gait_cycle = sum(cycle_times) / len(cycle_times)
                cadence = (60.0 / avg_gait_cycle) * 2 # step per menit
                
                temp_data = {
                    "Parameter": ["Total Baris Data", "Gait Cycle Rata-Rata (s)", "Cadence (step/min)", "Jumlah Siklus"],
                    "Nilai": [len(t), round(avg_gait_cycle, 3), round(cadence, 2), num_cycles]
                }
                st.table(pd.DataFrame(temp_data))
                
                # --- STEP 6: JOINT ANGLE PARAMETERS ---
                st.subheader("6. Joint Angle Parameters (Averaged 0-100% Stride)")
                
                hip_cycles, knee_cycles, ankle_cycles = [], [], []
                
                for i in range(num_cycles):
                    start_idx = hs_indices[i]
                    end_idx = hs_indices[i+1]
                    
                    # Pemotongan manual dan interpolasi ke 100%
                    hip_cycles.append(manual_interpolate_100(raw_hip[start_idx:end_idx]))
                    knee_cycles.append(manual_interpolate_100(raw_knee[start_idx:end_idx]))
                    ankle_cycles.append(manual_interpolate_100(raw_ankle[start_idx:end_idx]))
                
                # Menghitung Rata-rata dari multiple
