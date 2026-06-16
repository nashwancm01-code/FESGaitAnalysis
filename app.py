import streamlit as st
import pandas as pd
import math
import matplotlib.pyplot as plt
import io

st.set_page_config(page_title="Biomechanics Analysis", layout="wide")

# ==========================================
# FUNGSI PEMROSESAN MANUAL (PURE PYTHON)
# ==========================================

@st.cache_data
def manual_lowpass_filter(data, dt, cutoff, order=2):
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
    max_val = max([abs(x) for x in data])
    if max_val == 0: return data
    return [x / max_val for x in data]

def detect_gait_cycles_manual(heel_norm, threshold, dt, cooldown=0.4):
    hs_indices = []
    for i in range(1, len(heel_norm)):
        if heel_norm[i-1] < threshold and heel_norm[i] >= threshold:
            if not hs_indices or (i - hs_indices[-1]) * dt > cooldown:
                hs_indices.append(i)
    return hs_indices

def manual_interpolate_100(vector):
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
    for col in columns:
        if any(kw.lower() in col.lower() for kw in keywords):
            return col
    return None

# ==========================================
# UI APLIKASI & PEMBACAAN DATA CERDAS
# ==========================================
st.title("🏃‍♀️ Integrated Biomechanics Analysis App")
st.markdown("Membaca **satu file data flat** yang berisi gabungan EMG dan Gait Sensor.")

uploaded_file = st.file_uploader("Upload File Data (.txt atau .csv)", type=["txt", "csv"])

if uploaded_file is not None:
    # 1. BACA FILE DENGAN CERDAS (Abaikan Metadata di atas)
    content = uploaded_file.getvalue().decode("utf-8")
    lines = content.splitlines()
    
    header_idx = 0
    for i, line in enumerate(lines):
        # Cari baris yang punya kata 'time' atau 'waktu' sebagai baris kolom sebenarnya
        if 'time' in line.lower() or 'waktu' in line.lower():
            header_idx = i
            break
            
    # Gunakan separator whitespace dinamis (\s+) untuk membaca spasi/tab yang berantakan
    df = pd.read_csv(io.StringIO(content), sep=r'\s+', skiprows=header_idx)
    if len(df.columns) < 3: # Fallback kalau ternyata pakai koma
        df = pd.read_csv(io.StringIO(content), sep=',', skiprows=header_idx)
        
    st.success("Data berhasil dimuat dan dibaca kolomnya!")
    columns = list(df.columns)
    
    tab1, tab2 = st.tabs(["📊 Tugas 1: Analisis EMG", "🚶‍♀️ Tugas 2: Ekstraksi Parameter Gait"])
    
    # ==========================================
    # TAB 1: EMG
    # ==========================================
    with tab1:
        st.header("Analisis Sinyal EMG")
        time_col = find_column(columns, ['time', 'waktu', 'detik'])
        if not time_col:
            st.warning(f"Tidak menemukan kolom waktu. Kolom yang terbaca: {columns[:5]}...")
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
                st.info("Tidak ada kolom otot terdeteksi di file ini.")

    # ==========================================
    # TAB 2: GAIT ANALYSIS MANUAL (FLEXIBLE)
    # ==========================================
    with tab2:
        st.header("Ekstraksi Parameter Gait (Manual Processing)")
        
        time_col = find_column(columns, ['time', 'waktu', 'detik'])
        heel_col = find_column(columns, ['heel', 'fsr1'])
        toe_col = find_column(columns, ['toe', 'fsr2'])
        
        # Cek ketersediaan parameter utama
        if not time_col:
            st.error("Kolom Time tidak terdeteksi.")
        elif not heel_col or not toe_col:
            st.error(f"Kolom Heel/Toe tidak lengkap. Kolom yang ada: {columns}")
        else:
            t = df[time_col].tolist()
            raw_heel = df[heel_col].tolist()
            raw_toe = df[toe_col].tolist()
            
            dt = t[1] - t[0] if len(t) > 1 else 0.01
            
            # --- STEP 1: RAW SIGNAL ---
            st.subheader("1. Grafik Raw Input Signal")
            fig1, ax1 = plt.subplots(figsize=(12, 3))
            ax1.plot(t, raw_heel, label='Raw Heel', color='blue', alpha=0.7)
            ax1.plot(t, raw_toe, label='Raw Toe', color='red', alpha=0.7)
            ax1.set_xlabel('Time (s)')
            ax1.legend()
            st.pyplot(fig1)
            
            # --- STEP 2 & 3: FILTER & NORMALIZATION ---
            st.subheader("2. Filter LPF & Normalisasi (Threshold 5%)")
            filt_heel = manual_lowpass_filter(raw_heel, dt, cutoff=5.0, order=2)
            filt_toe = manual_lowpass_filter(raw_toe, dt, cutoff=5.0, order=2)
            
            norm_heel = manual_normalize_max(filt_heel)
            norm_toe = manual_normalize_max(filt_toe)
            
            threshold = 0.05
            hs_indices = detect_gait_cycles_manual(norm_heel, threshold, dt)
            
            fig3, ax3 = plt.subplots(figsize=(12, 3))
            ax3.plot(t, norm_heel, label='Normalized Heel', color='blue')
            ax3.plot(t, norm_toe, label='Normalized Toe', color='red')
            ax3.axhline(y=threshold, color='green', linestyle='--', label='5% Threshold')
            
            t_hs = [t[i] for i in hs_indices]
            y_hs = [norm_heel[i] for i in hs_indices]
            ax3.scatter(t_hs, y_hs, color='black', zorder=5, label='Heel Strike')
            ax3.legend()
            st.pyplot(fig3)
            
            # --- STEP 4: TEMPORAL PARAMETERS ---
            st.subheader("3. Temporal Parameters")
            num_cycles = len(hs_indices) - 1
            if num_cycles > 0:
                cycle_times = [(t[hs_indices[i+1]] - t[hs_indices[i]]) for i in range(num_cycles)]
                avg_gait_cycle = sum(cycle_times) / len(cycle_times)
                cadence = (60.0 / avg_gait_cycle) * 2
                
                temp_data = {
                    "Parameter": ["Total Baris Data", "Gait Cycle Rata-Rata (s)", "Cadence (step/min)", "Jumlah Siklus"],
                    "Nilai": [len(t), round(avg_gait_cycle, 3), round(cadence, 2), num_cycles]
                }
                st.table(pd.DataFrame(temp_data))
                
                # --- STEP 5: JOINT ANGLE PARAMETERS (OPSIONAL) ---
                hip_col = find_column(columns, ['hip'])
                knee_col = find_column(columns, ['knee'])
                ankle_col = find_column(columns, ['ankle'])
                
                if hip_col and knee_col and ankle_col:
                    st.subheader("4. Joint Angle Parameters (0-100% Stride)")
                    raw_hip = df[hip_col].tolist()
                    raw_knee = df[knee_col].tolist()
                    raw_ankle = df[ankle_col].tolist()
                    
                    hip_cycles, knee_cycles, ankle_cycles = [], [], []
                    
                    for i in range(num_cycles):
                        start_idx = hs_indices[i]
                        end_idx = hs_indices[i+1]
                        
                        hip_cycles.append(manual_interpolate_100(raw_hip[start_idx:end_idx]))
                        knee_cycles.append(manual_interpolate_100(raw_knee[start_idx:end_idx]))
                        ankle_cycles.append(manual_interpolate_100(raw_ankle[start_idx:end_idx]))
                    
                    avg_hip = [sum(col) / len(col) for col in zip(*hip_cycles)]
                    avg_knee = [sum(col) / len(col) for col in zip(*knee_cycles)]
                    avg_ankle = [sum(col) / len(col) for col in zip(*ankle_cycles)]
                    
                    x_percent = list(range(100))
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        fig_h, ax_h = plt.subplots(figsize=(5, 4))
                        ax_h.plot(x_percent, avg_hip, color='blue', linewidth=2)
                        ax_h.set_title("Hip Joint")
                        st.pyplot(fig_h)
                    with col2:
                        fig_k, ax_k = plt.subplots(figsize=(5, 4))
                        ax_k.plot(x_percent, avg_knee, color='green', linewidth=2)
                        ax_k.set_title("Knee Joint")
                        st.pyplot(fig_k)
                    with col3:
                        fig_a, ax_a = plt.subplots(figsize=(5, 4))
                        ax_a.plot(x_percent, avg_ankle, color='red', linewidth=2)
                        ax_a.set_title("Ankle Joint")
                        st.pyplot(fig_a)
                else:
                    st.info("💡 Grafik sudut sendi (Kinematika) tidak ditampilkan karena kolom Hip/Knee/Ankle tidak ada di file ini.")
            else:
                st.warning("Siklus tidak cukup terdeteksi.")
