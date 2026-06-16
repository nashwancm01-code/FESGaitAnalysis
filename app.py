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

# ==========================================
# UI APLIKASI & PEMBACAAN DATA CERDAS
# ==========================================
st.title("🏃‍♀️ Integrated Biomechanics Analysis App")

uploaded_file = st.file_uploader("Upload File Data (.txt atau .csv)", type=["txt", "csv"])

if uploaded_file is not None:
    content = uploaded_file.getvalue().decode("utf-8")
    
    # 1. CEK APAKAH FILE PUNYA JUDUL ATAU CUMA ANGKA
    try:
        # Coba baca 5 baris pertama tanpa menganggap ada judul
        df_test = pd.read_csv(io.StringIO(content), sep=r'\s+', nrows=5, header=None)
        first_row_val = str(df_test.iloc[0, 0])
        
        # Kalau baris pertama isinya bisa diubah ke angka (float), berarti file gak punya header
        is_no_header = False
        try:
            float(first_row_val)
            is_no_header = True
        except ValueError:
            is_no_header = False

        if is_no_header:
            # Baca data sebagai kumpulan angka dan beri nama generik
            df = pd.read_csv(io.StringIO(content), sep=r'\s+', header=None)
            df.columns = [f"Kolom_{i}" for i in range(len(df.columns))]
            st.warning("⚠️ File ini hanya berisi angka tanpa judul kolom. Gunakan pengaturan di bawah untuk memetakan kolomnya.")
        else:
            # File normal (ada judul/metadata)
            lines = content.splitlines()
            header_idx = 0
            for i, line in enumerate(lines[:20]): 
                if 'time' in line.lower() or 'waktu' in line.lower() or 'fsr' in line.lower():
                    header_idx = i
                    break
            df = pd.read_csv(io.StringIO(content), sep=r'\s+', skiprows=header_idx)
            if len(df.columns) < 3: 
                df = pd.read_csv(io.StringIO(content), sep=',', skiprows=header_idx)
                
        st.success("Data berhasil dimuat!")
        columns = list(df.columns)

        # 2. PENGATURAN KOLOM MANUAL (Mencegah error salah deteksi kolom)
        st.markdown("### ⚙️ Pengaturan Kolom Data")
        st.write("Silakan pilih kolom mana yang sesuai dengan data berikut. (Biasanya Waktu di Kolom_0, FSR 1 di Kolom_1, dll).")
        
        col_t, col_h, col_to = st.columns(3)
        with col_t:
            time_col = st.selectbox("Pilih Kolom Waktu (Time):", columns, index=0)
        with col_h:
            heel_col = st.selectbox("Pilih Kolom FSR Heel:", columns, index=1 if len(columns)>1 else 0)
        with col_to:
            toe_col = st.selectbox("Pilih Kolom FSR Toe:", columns, index=2 if len(columns)>2 else 0)
            
        st.divider()

        tab1, tab2 = st.tabs(["📊 Tugas 1: Analisis EMG", "🚶‍♀️ Tugas 2: Ekstraksi Parameter Gait"])
        
        # ==========================================
        # TAB 1: EMG
        # ==========================================
        with tab1:
            st.header("Analisis Sinyal EMG")
            t = df[time_col].tolist()
            
            # Sisa kolom yang bukan waktu/heel/toe bisa dianggap sebagai otot (EMG)
            used_cols = [time_col, heel_col, toe_col]
            muscle_cols = [c for c in columns if c not in used_cols]
            
            if muscle_cols:
                selected_muscle = st.selectbox("Pilih Kolom Otot (EMG) untuk Dilihat:", muscle_cols)
                raw_muscle = df[selected_muscle].tolist()
                
                fig_emg, ax_emg = plt.subplots(figsize=(10, 3))
                ax_emg.plot(t, raw_muscle, color='#333333', linewidth=0.7)
                ax_emg.set_title(f"RAW EMG: {selected_muscle}")
                st.pyplot(fig_emg)
            else:
                st.info("Tidak ada sisa kolom untuk data EMG.")

        # ==========================================
        # TAB 2: GAIT ANALYSIS
        # ==========================================
        with tab2:
            st.header("Ekstraksi Parameter Gait (Manual Processing)")
            
            t = df[time_col].tolist()
            raw_heel = df[heel_col].tolist()
            raw_toe = df[toe_col].tolist()
            
            # Hitung dt (selisih waktu antar baris)
            dt = t[1] - t[0] if len(t) > 1 else 0.01
            if dt == 0: dt = 0.01 # Cegah error division by zero
            
            # --- STEP 1: RAW SIGNAL ---
            st.subheader("1. Grafik Raw Input Signal")
            fig1, ax1 = plt.subplots(figsize=(12, 3))
            ax1.plot(t, raw_heel, label=f'Raw Heel ({heel_col})', color='blue', alpha=0.7)
            ax1.plot(t, raw_toe, label=f'Raw Toe ({toe_col})', color='red', alpha=0.7)
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
            else:
                st.warning("Siklus tidak cukup terdeteksi. Silakan periksa grafik apakah ada lonjakan sinyal yang jelas.")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses data: {e}")
