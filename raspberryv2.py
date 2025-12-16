#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#################################################################################################
# CPU/RAM SAFETY: limit BLAS threads (set BEFORE numpy import)
#################################################################################################
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

#################################################################################################
# PLOTTING SAFETY: headless backend (set BEFORE pyplot import)
#################################################################################################
import matplotlib
matplotlib.use("Agg")  # avoid GUI / plt.show() on Raspberry Pi OS
import matplotlib.pyplot as plt

#################################################################################################
# IMPORTS
#################################################################################################
import gc
import psutil
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split
import wfdb
import neurokit2 as nk
from scipy.signal import butter, filtfilt

#################################################################################################
# GLOBAL PARAMETERS
#################################################################################################
FS_DEFAULT = 200         # ECG sampling rate (Hz)
WO = 300                 # Window size (s)
S = 110                  # Step (s)
PRE_ICTAL_SEC = 1800     # Pre-ictal duration (s)

# Debug/plots (KEEP OFF on low-RAM boards)
SAVE_PLOTS = True
OUT_DIR = "./out_pi"
os.makedirs(OUT_DIR, exist_ok=True)

#################################################################################################
# ECG PROCESSING
#################################################################################################
def butterworth_filter(ecg_signal, fs, lowcut=0.5, highcut=40.0, order=2):
    ecg_signal = np.asarray(ecg_signal, dtype=np.float32)
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    # filtfilt allocates temporaries; float32 helps
    return filtfilt(b, a, ecg_signal).astype(np.float32)

def detect_r_peaks_fast(ecg_segment, fs):
    """
    Much lighter than CWT. Uses NeuroKit peak detector.
    """
    # ecg_clean is optional; your Butterworth already helps, but cleaning improves robustness
    cleaned = nk.ecg_clean(ecg_segment, sampling_rate=fs, method="neurokit")
    _, peaks = nk.ecg_peaks(cleaned, sampling_rate=fs)
    return np.asarray(peaks.get("ECG_R_Peaks", []), dtype=np.int32)

def sample_entropy_slow(data, m=2):
    """
    Simple SampEn implementation. OK for ~300 beats/window.
    """
    data = np.asarray(data, dtype=np.float64)
    N = len(data)
    if N < m + 2:
        return 0.0
    r = 0.2 * np.std(data)
    if r == 0:
        return 0.0

    # O(N^2) but small N in HRV windows
    count_m = 0
    count_m1 = 0
    for i in range(N - m):
        template_m = data[i:i+m]
        template_m1 = data[i:i+m+1]
        for j in range(i + 1, N - m):
            if np.all(np.abs(template_m - data[j:j+m]) <= r):
                count_m += 1
                if j < N - m - 1 and np.all(np.abs(template_m1 - data[j:j+m+1]) <= r):
                    count_m1 += 1

    if count_m == 0 or count_m1 == 0:
        return 0.0
    return float(-np.log(count_m1 / count_m))

def calculate_hrv_metrics_from_rpeaks(rpeaks, fs):
    """
    Returns 7 features: [sdnn, rmssd, lf, hf, sampen, csi, cvi]
    """
    if rpeaks is None or len(rpeaks) < 3:
        return np.zeros(7, dtype=np.float32)

    rr = np.diff(rpeaks) / float(fs)  # seconds
    if len(rr) < 2:
        return np.zeros(7, dtype=np.float32)

    sdnn = float(np.std(rr))
    rmssd = float(np.sqrt(np.mean(np.diff(rr) ** 2)))

    # LF/HF (can fail depending on window content; keep safe defaults)
    lf = 0.0
    hf = 0.0
    try:
        hrv_freq = nk.hrv_frequency(rpeaks, sampling_rate=fs, show=False)
        if "HRV_LF" in hrv_freq:
            lf = float(hrv_freq["HRV_LF"].iloc[0])
        if "HRV_HF" in hrv_freq:
            hf = float(hrv_freq["HRV_HF"].iloc[0])
    except Exception:
        lf, hf = 0.0, 0.0

    sampen = float(sample_entropy_slow(rr, m=2))
    csi = float(lf / hf) if hf != 0 else 0.0
    cvi = float(hf)

    return np.array([sdnn, rmssd, lf, hf, sampen, csi, cvi], dtype=np.float32)

def extract_hrv_parameters(ecg_segment, fs):
    rpeaks = detect_r_peaks_fast(ecg_segment, fs)
    return calculate_hrv_metrics_from_rpeaks(rpeaks, fs)

#################################################################################################
# WINDOWING + LABELING (NO per-sample label arrays -> saves a lot of RAM)
#################################################################################################
def sliding_window_features_and_labels(ecg, fs, seizure_onsets_sec,
                                      wo_sec=WO, step_sec=S, pre_ictal_sec=PRE_ICTAL_SEC):
    wo_samp = int(wo_sec * fs)
    step_samp = int(step_sec * fs)

    if len(ecg) < wo_samp + 1:
        return np.zeros((0, 7), dtype=np.float32), np.zeros((0,), dtype=np.int8)

    # number of windows
    n_win = 1 + (len(ecg) - wo_samp) // step_samp
    X = np.zeros((n_win, 7), dtype=np.float32)
    y = np.full((n_win,), -1, dtype=np.int8)

    onsets = np.asarray(seizure_onsets_sec, dtype=np.float64)
    pre_starts = onsets - float(pre_ictal_sec)
    pre_ends = onsets

    w = 0
    for start in range(0, len(ecg) - wo_samp + 1, step_samp):
        seg = ecg[start:start + wo_samp]
        X[w, :] = extract_hrv_parameters(seg, fs)

        # label window by interval overlap with any pre-ictal region
        start_s = start / float(fs)
        end_s = (start + wo_samp) / float(fs)
        if onsets.size > 0 and np.any((end_s > pre_starts) & (start_s < pre_ends)):
            y[w] = 1
        w += 1

    return X, y

#################################################################################################
# PATIENT LOADER (still aggregates, but much safer now)
#################################################################################################
def carregar_todos_pacientes(base_path, n_pacientes=None):
    registros = sorted([f.replace(".hea", "") for f in os.listdir(base_path) if f.endswith(".hea")])
    if n_pacientes is not None:
        registros = registros[:n_pacientes]

    all_X = []
    all_y = []

    for i, rec in enumerate(registros):
        print(f"\n[DATA] Processando paciente {i+1}/{len(registros)} → {rec}")

        path_record = os.path.join(base_path, rec)
        record = wfdb.rdrecord(path_record)  # reads full record
        fs = int(getattr(record, "fs", FS_DEFAULT))
        ecg = np.asarray(record.p_signal[:, 0], dtype=np.float32)

        # filter
        ecg = butterworth_filter(ecg, fs)

        # load seizure onsets (seconds)
        path_seiz = os.path.join(base_path, rec + ".seizures")
        onsets = []
        if os.path.exists(path_seiz):
            with open(path_seiz, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    onsets.append(float(line.split()[0]))

        X, y = sliding_window_features_and_labels(ecg, fs, onsets)

        print(f"[HRV] Janelas: {len(y)} (fs={fs} Hz)")
        all_X.append(X)
        all_y.append(y)

        # free big arrays ASAP
        del record, ecg, X, y
        gc.collect()

    if len(all_X) == 0:
        return np.zeros((0, 7), dtype=np.float32), np.zeros((0,), dtype=np.int8)

    return np.vstack(all_X), np.hstack(all_y)

#################################################################################################
# METRICS + PLOTS
#################################################################################################
def calcular_metricas(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[-1, 1])
    fp = (cm[0, 1] / np.sum(cm[0, :])) if np.sum(cm[0, :]) > 0 else 0.0
    sensibilidade = recall[1] if len(recall) > 1 else 0.0
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fp_rate": fp,
        "sensibilidade": sensibilidade,
        "confusion_matrix": cm,
    }

def save_confusion_matrix(cm, out_path):
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Matriz de Confusão")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Interictal", "Pré-ictal"])
    plt.yticks(tick_marks, ["Interictal", "Pré-ictal"])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.xlabel("Predito")
    plt.ylabel("Verdadeiro")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

#################################################################################################
# MAIN
#################################################################################################
if __name__ == "__main__":
    print("\n================ INICIALIZAÇÃO DO SISTEMA =================")
    print("Sistema iniciado com sucesso na Raspberry Pi OS (Linux).")

    base_path = "/home/pi/MicproRaspberry/synthetic_hrv_bigsep"
    n_pacientes = 2

    print("\n[1/6] Carregando modelo treinado (SVM + Scaler + PCA)...")
    data = joblib.load("modelo_svm.joblib")
    modelo_svm = data["modelo"]
    scaler = data["scaler"]
    pca = data["pca"]
    print("Modelo carregado com sucesso.")

    print("\n[2/6] Carregando e processando dados ECG (modo seguro, sem CWT)...")
    X_raw, y = carregar_todos_pacientes(base_path, n_pacientes)
    print(f"Total de janelas processadas: {len(y)}")

    if len(y) == 0:
        raise SystemExit("Nenhuma janela gerada. Verifique base_path / arquivos .hea/.seizures")

    # If only one class exists, stratify will crash
    unique = np.unique(y)
    if unique.size < 2:
        print(f"[WARN] Apenas 1 classe encontrada ({unique}). Avaliando sem split.")
        X_test, y_test = X_raw, y
    else:
        print("\n[3/6] Divisão treino/teste (70/30)...")
        _, X_test, _, y_test = train_test_split(
            X_raw, y, test_size=0.30, random_state=42, stratify=y
        )

    print("\n[4/6] Aplicando normalização e PCA...")
    X_test_scaled = scaler.transform(X_test)
    X_test_pca = pca.transform(X_test_scaled)
    print(f"Número de componentes PCA: {X_test_pca.shape[1]}")

    print("\n[5/6] Executando predição com SVM...")
    y_pred = modelo_svm.predict(X_test_pca)
    print("Predição finalizada.")

    print("\n[6/6] Métricas + memória...")
    metricas = calcular_metricas(y_test, y_pred)
    print("\n=== Métricas do Modelo ===")
    print(f"Acurácia: {metricas['accuracy']:.3f}")
    print(f"Precision por classe: {metricas['precision']}")
    print(f"Recall por classe: {metricas['recall']}")
    print(f"F1-score por classe: {metricas['f1']}")
    print(f"Sensibilidade: {metricas['sensibilidade']:.3f}")
    print(f"Taxa de falsos positivos: {metricas['fp_rate']:.3f}")

    if SAVE_PLOTS:
        cm_path = os.path.join(OUT_DIR, "confusion_matrix.png")
        save_confusion_matrix(metricas["confusion_matrix"], cm_path)
        print(f"[PLOT] Confusion matrix salva em: {cm_path}")

    process = psutil.Process()
    mem = process.memory_info().rss / 1024**2
    print(f"[MEM] RSS do processo: {mem:.2f} MB")
