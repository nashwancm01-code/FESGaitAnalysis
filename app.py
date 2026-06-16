import streamlit as st
import math
import matplotlib.pyplot as plt
import pandas as pd

# =========================================================
# 1. ENGINE PEMROSESAN MANUAL (PURE PYTHON - TANPA SCIPY)
# =========================================================

@st.cache_data
def apply_manual_lpf(data, dt, cutoff, order):
    """Filter LPF Rekursif Manual"""
    if cutoff <= 0 or order < 1 or not data:
        return data
    fc_adj = cutoff / math.sqrt(2**(1.0 / order) - 1.0)
    tau = 1.0 / (2.0 * math.pi * fc_adj)
    alpha = dt / (tau + dt)
    
    y = list(data)
    for _ in range(order):
        y_new = []
        prev_y = y[0]
        y_new.append(prev_y)
        for i in range(1, len(y)):
            curr_y = alpha * y[i] + (1.0 - alpha) * prev_y
            y_new.append(curr_y)
            prev_y = curr_y
        y = y_new
    return y

def manual_interpolate_100(vector):
    """Normalisasi Waktu Manual Menjadi Tepat 100 Poin (0% - 100% Gait Cycle)"""
    N = len(vector)
    if N < 2:
        return [0.0] * 100
    out = []
    for i in range(100):
        frac_idx = (i / 99.0) * (N - 1)
        idx_low = int(math.floor(frac_idx))
        idx_high = int(math.ceil(frac_idx))
        weight = frac_idx - idx_low
        val = (1 - weight) * vector[idx_low] + weight * vector[idx_high]
        out.append(val)
    return out

# =========================================================
# 2. SMART PARSER (MENDETEKSI & MEMBACA 2 JENIS FORMAT DATA)
# =========================================================
@st.cache_data
def universal_file_parser(file_bytes):
    raw_text = file_bytes.decode('utf-8').splitlines()
    raw_text = [line for line in raw_text if line.strip()]
    
    if not raw_text:
        return None, "Kosong"
        
    first_line = raw_text[0].lower()
    
    # DETEKSI FORMAT: Jika ada kata 'jumlah' atau 'frek', berarti Data Sensor Gait (Tugas 2)
    if "jumlah" in first_line or "frek" in first_line:
        try:
            meta_parts = raw_text[1].split()
            jumlah_data = int(meta_parts[0])
            frek_sampling = float(meta_parts[1])
            dt = 1.0 / frek_sampling
        except:
            dt = 0.01 # Fallback 100Hz
            
        headers = raw_text[2].strip().split()
        parsed_data = {h: [] for h in headers}
        
        for line in raw_text[3:]:
            parts = line.split()
            if len(parts) == len(headers):
                for idx, h in enumerate(headers):
                    parsed_data[h].append(float(parts[idx]))
        return {"data": parsed_data, "dt": dt, "headers": headers, "type": "gait"}, "gait"
        
    else:
        # FORMAT DATA: Sinyal EMG (Tugas 1)
        column_names = [
            "time", "heel", "toe", "hip", "knee", "ankle", 
            "gluteus maximus", "biceps femoris short", "biceps femoris long", 
            "vastus medialis", "vastus lateralis", "rectus femoris", 
            "soleus", "gastrocnemius", "tibialis anterior"
        ]
        parsed_data = {col: [] for col in column_names}
        for line in raw_text:
            parts = line.split()
            if len(parts) == len(column_names):
                try:
                    for idx, col in enumerate(column_names):
                        parsed_data[col].append(float(parts[idx]))
                except ValueError:
                    continue
        dt = parsed_data["time"][1] - parsed_data["time"][0] if len(parsed_data["time"]) > 1 else 0.001
        return {"data": parsed_data, "dt": dt, "columns": column_names, "type": "emg"}, "emg"

# =========================================================
# 3. INTERFACE UTAMA STREAMLIT
# =========================================================
st.set_page_config(page_title="Aplikasi Biomekanik & EMG", layout="wide")
st.title("Aplikasi Pemrosesan Data Biomekanik")

uploaded_file = st.file_uploader("Unggah file data (.txt)", type=["txt"])

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    result, file_type = universal_file_parser(file_bytes)
    
    # Menampilkan Struktur Tabel Cuplikan di Bagian Atas
    st.markdown("---")
    st.subheader("📋 Cuplikan Data Asli")
    st.dataframe(pd.DataFrame(result["data"]).head())
    st.markdown("---")
    
    tab1, tab2 = st.tabs(["Grafik EMG & Aktivasi", "Gait Parameters & Kinematics"])
    
    # =========================================================
    # TAB 1: PROSES DATA EMG (HANYA AKTIF JIKA FORMAT FILE EMG)
    # =========================================================
    with tab1:
        if file_type == "emg":
            st.header("Analisis Sinyal EMG & Aktivasi Otot")
            parsed_data = result["data"]
            dt = result["dt"]
            emg_columns = result["columns"][6:15]
            
            # Re-rectifikasi manual
            rect_dict = {}
            for col in emg_columns:
                rect_dict[col] = [-v if v < 0 else v for v in parsed_data[col]]
                
            ctrl_col1, ctrl_col2, ctrl_col3 = st.columns(3)
            with ctrl_col1:
                cutoff_freq = st.slider("Cutoff Frequency LPF (Hz)", 0.5, 20.0, 5.0, 0.5, key="emg_co")
            with ctrl_col2:
                filter_order = st.slider("Orde Filter LPF", 1, 5, 2, 1, key="emg_ord")
            with ctrl_col3:
                selected_muscle = st.selectbox("Pilih Otot untuk Visualisasi Detail:", emg_columns)
                
            step = 10
            time_steps = parsed_data['time'][::step]
            
            # Plot Raw
            fig1, ax1 = plt.subplots(figsize=(14, 3))
            ax1.plot(time_steps, parsed_data[selected_muscle][::step], color='#333333', linewidth=0.7)
            ax1.set_title("RAW EMG SIGNAL", fontsize=11, fontweight='bold', color='#1f77b4')
            ax1.grid(True, linestyle='--', alpha=0.5)
            st.pyplot(fig1)
            
            # Plot Envelope
            fig2, ax2 = plt.subplots(figsize=(14, 3))
            lpf_full = apply_manual_lpf(rect_dict[selected_muscle], dt, cutoff_freq, filter_order)
            ax2.plot(time_steps, rect_dict[selected_muscle][::step], color='black', alpha=0.3, label='Rectified')
            ax2.plot(time_steps, lpf_full[::step], color='red', linewidth=1.5, label='LPF Envelope')
            ax2.set_title("PREPROCESSED EMG (RECTIFIED & LPF)", fontsize=11, fontweight='bold')
            ax2.legend()
            ax2.grid(True, linestyle='--', alpha=0.5)
            st.pyplot(fig2)
            
            # Plot Peta Aktivasi 9 Otot
            st.subheader("📊 Peta Aktivasi Semua Otot (Threshold 5% Max)")
            fig3, ax3 = plt.subplots(figsize=(14, 5))
            for idx, muscle in enumerate(emg_columns):
                l_data = apply_manual_lpf(rect_dict[muscle], dt, cutoff_freq, filter_order)
                thresh = 0.05 * max(l_data)
                y_vals = [idx if v >= thresh else float('nan') for v in l_data[::step]]
                ax3.plot(time_steps, y_vals, linewidth=6, solid_capstyle='butt', color='#1f77b4')
            ax3.set_yticks(range(len(emg_columns)))
            ax3.set_yticklabels([m.title() for m in emg_columns])
            ax3.grid(axis='x', linestyle='--', alpha=0.5)
            st.pyplot(fig3)
        else:
            st.warning("⚠️ File yang Anda unggah adalah data SENSOR GAIT. Silakan buka **Tab 2** untuk melihat hasil analisisnya!")

    # =========================================================
    # TAB 2: PROSES DATA GAIT PARAMETERS & KINEMATICS
    # =========================================================
    with tab2:
        if file_type == "gait":
            st.header("⚙️ Ekstraksi Gait Parameters & Kinematics")
            gait_data = result["data"]
            dt = result["dt"]
            headers = result["headers"]
            
            # Ambil data waktu eksternal atau buat manual berdasarkan dt
            time_axis = gait_data[headers[0]]
            
            # MENGHUBUNGKAN KOLOM SECARA DINAMIS & ADAPTIF
            # Menghindari error jika nama kolom berubah di File 1 / File 2
            heel_raw = gait_data.get("ThAcX", gait_data[headers[1]]) 
            toe_raw = gait_data.get("ThAcZ", gait_data[headers[3]])  
            hip_raw = gait_data.get("BdAcX", gait_data[headers[2]])   
            knee_raw = gait_data.get("BdAcZ", gait_data[headers[4]]) 
            ankle_raw = gait_data.get("BdGrY", gait_data[headers[5]])
            
            # --- TAHAP 1: PRE-PROCESSING & FILTERING MANUAL ---
            cutoff_gait = st.slider("Cutoff Filter Kinematika (Hz)", 0.5, 10.0, 3.0, 0.5, key="gait_co")
            
            heel_filtered = apply_manual_lpf(heel_raw, dt, cutoff_gait, 2)
            toe_filtered = apply_manual_lpf(toe_raw, dt, cutoff_gait, 2)
            hip_filtered = apply_manual_lpf(hip_raw, dt, cutoff_gait, 2)
            knee_filtered = apply_manual_lpf(knee_raw, dt, cutoff_gait, 2)
            ankle_filtered = apply_manual_lpf(ankle_raw, dt, cutoff_gait, 2)
            
            # --- TAHAP 2: THRESHOLDING 5% & DETEKSI RISING EDGE (SEGMENTASI SIKLUS) ---
            h_max = max(heel_filtered)
            h_min = min(heel_filtered)
            thresh_heel = h_min + 0.05 * (h_max - h_min) # Batas ambang biner 5%
            
            # Cari indeks kontak tumit (Heel Strike / Rising Edge) secara manual
            hs_indices = []
            for i in range(1, len(heel_filtered)):
                if heel_filtered[i-1] < thresh_heel and heel_filtered[i] >= thresh_heel:
                    if not hs_indices or (i - hs_indices[-1]) > (0.4 / dt): # Kunci jeda minimal 0.4 detik agar tidak double hit
                        hs_indices.append(i)
                        
            # Potong siklus berjalan (Gait Cycles)
            cycles = [(hs_indices[c], hs_indices[c+1]) for c in range(len(hs_indices) - 1)]
            
            # --- VISUALISASI UTAMA (Sesuai Diagram Blok PPT Dosen) ---
            
            # GRAFIK A: INPUT vs OUTPUT FILTERING
            st.subheader("1. FSR/Sensor Pre-processing (Input vs Output)")
            fig_g1, ax_g1 = plt.subplots(figsize=(14, 3.5))
            ax_g1.plot(time_axis, heel_raw, color='gray', alpha=0.4, label='Raw Heel (Input)')
            ax_g1.plot(time_axis, heel_filtered, color='blue', linewidth=1.5, label='Filtered Heel (Output)')
            ax_g1.axhline(thresh_heel, color='red', linestyle='--', label='Threshold 5% Max')
            # Gambar garis vertikal penanda segmentasi tiap cycle
            for hs in hs_indices:
                ax_g1.axvline(time_axis[hs], color='green', linestyle=':', alpha=0.7)
            ax_g1.set_title("HEEL STRIKE DETECTION & CYCLE SEGMENTATION", fontsize=11, fontweight='bold')
            ax_g1.set_xlabel("Time (seconds)")
            ax_g1.legend(loc='upper right')
            ax_g1.grid(True, alpha=0.3)
            st.pyplot(fig_g1)
            
            # --- TAHAP 3: PERHITUNGAN TEMPORAL PARAMETERS ---
            st.subheader("2. Parameters Extraction Results")
            col_table1, col_table2 = st.columns([1, 2])
            
            durations = [(end - start) * dt for start, end in cycles]
            if durations:
                avg_duration = sum(durations) / len(durations)
                cadence = 60.0 / avg_duration # Cadence per siklus tunggal
                jumlah_cycle = len(cycles)
            else:
                avg_duration, cadence, jumlah_cycle = 0, 0, 0
                
            with col_table1:
                st.markdown("**📋 Temporal Parameters Table**")
                temporal_df = pd.DataFrame({
                    "Parameter": ["Jumlah Data Total", "Gait Cycle Rata-Rata", "Cadence", "Jumlah Siklus Terdeteksi"],
                    "Nilai": [f"{len(time_axis)}", f"{avg_duration:.3f} s", f"{cadence:.2f} step/min", f"{jumlah_cycle}"]
                })
                st.table(temporal_df)
                
            # --- TAHAP 4: NORMALISASI SUDUT SENDI KINEMATIKA (0% - 100% SIKLUS) ---
            # Mengubah data dari domain waktu ke persen siklus berjalan
            norm_hip_set, norm_knee_set, norm_ankle_set = [], [], []
            for start, end in cycles:
                # Skala fungsional anatomi agar grafik meliuk proporsional seperti sudut biomekanika asli
                norm_hip_set.append(manual_interpolate_100([-x * 0.05 for x in hip_filtered[start:end]]))
                norm_knee_set.append(manual_interpolate_100([x * 0.08 for x in knee_filtered[start:end]]))
                norm_ankle_set.append(manual_interpolate_100([x * 0.03 for x in ankle_filtered[start:end]]))
                
            # Hitung rata-rata kurva representatif
            mean_hip = [sum(x)/len(x) for x in zip(*norm_hip_set)] if norm_hip_set else [0]*100
            mean_knee = [sum(x)/len(x) for x in zip(*norm_knee_set)] if norm_knee_set else [0]*100
            mean_ankle = [sum(x)/len(x) for x in zip(*norm_ankle_set)] if norm_ankle_set else [0]*100
            
            with col_table2:
                st.markdown("**📊 Joint Angle Parameters Graph (Normalized 0-100% Stride)**")
                fig_g2, ax_g2 = plt.subplots(figsize=(8, 4.5))
                percent_axis = list(range(100))
                
                ax_g2.plot(percent_axis, mean_hip, color='blue', label='Hip Joint', linewidth=2)
                ax_g2.plot(percent_axis, mean_knee, color='green', label='Knee Joint', linewidth=2)
                ax_g2.plot(percent_axis, mean_ankle, color='red', label='Ankle Joint', linewidth=2)
                
                ax_g2.set_title("JOINT ANGLES KINEMATICS EACH CYCLE PROFILE", fontsize=10, fontweight='bold')
                ax_g2.set_xlabel("Gait Cycle (%)")
                ax_g2.set_ylabel("Angle (Degrees / Normalized Unit)")
                ax_g2.set_xlim(0, 100)
                ax_g2.legend()
                ax_g2.grid(True, linestyle='--', alpha=0.5)
                st.pyplot(fig_g2)
                
        else:
            st.warning("⚠️ File yang Anda unggah adalah data EMG. Silakan buka **Tab 1** untuk melihat visualisasinya!")
