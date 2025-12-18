#################################################################################################
# IMPORTS
#################################################################################################
import os, psutil, joblib, time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import wfdb
import neurokit2 as nk
import scipy.signal
from scipy.signal import butter, filtfilt
from scipy.interpolate import CubicSpline
import pywt

#################################################################################################
# PARÂMETROS GLOBAIS
#################################################################################################
fs = 200                # Frequência de amostragem do ECG
Wo = 300                # Tamanho da janela (s)
S = 110                 # Passo de sobreposição (s)
pre_ictal_sec = 1800    # Duração pré-ictal (s)

#################################################################################################
# FUNÇÕES DE PROCESSAMENTO DE ECG
#################################################################################################

def butterworth_filter(ecg_signal, fs, lowcut=0.5, highcut=40, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, ecg_signal)

def detect_r_peaks_cwt(ecg_signal, sampling_rate=fs, wavelet='mexh', scale_range=(1,20)):
    scales = np.arange(scale_range[0], scale_range[1])
    coeffs,_ = pywt.cwt(ecg_signal, scales, wavelet)
    cwt_sum = np.sum(np.abs(coeffs), axis=0)
    peaks,_ = scipy.signal.find_peaks(cwt_sum, distance=sampling_rate/2.5)
    return peaks

def fill_gaps_with_spline(rpeaks, fs):
    time_vector = np.arange(len(rpeaks))/fs
    cs = CubicSpline(time_vector, rpeaks)
    new_time_vector = np.linspace(0,time_vector[-1],len(rpeaks))
    interpolated_rpeaks = cs(new_time_vector)
    return interpolated_rpeaks

def check_ecg_quality(rpeaks):
    return len(rpeaks) >= 3

def calculate_hrv_metrics(rpeaks, fs=fs):
    rr_intervals = np.diff(rpeaks)/fs
    if len(rr_intervals)<2:
        return [0]*7
    sdnn = np.std(rr_intervals)
    rmssd = np.sqrt(np.mean(np.diff(rr_intervals)**2))
    try:
        hrv = nk.hrv(rpeaks, sampling_rate=fs)
        lf = hrv.get('HRV_LF',[0])[0]
        hf = hrv.get('HRV_HF',[0])[0]
    except:
        lf = hf = 0
    def sample_entropy(data,m=2):
        N = len(data)
        if N < m+2: return 0
        r = 0.2*np.std(data)
        phi_m = np.sum([np.all(np.abs(data[i:i+m]-data[j:j+m])<=r)
                        for i in range(N-m) for j in range(i+1,N-m)]) / ((N-m)*(N-m-1))
        phi_m1 = np.sum([np.all(np.abs(data[i:i+m+1]-data[j:j+m+1])<=r)
                         for i in range(N-m-1) for j in range(i+1,N-m-1)]) / ((N-m-1)*(N-m-2))
        return -np.log(phi_m1/phi_m) if phi_m>0 and phi_m1>0 else 0
    sampen = sample_entropy(rr_intervals)
    csi = lf/hf if hf!=0 else 0
    cvi = hf
    return [sdnn, rmssd, lf, hf, sampen, csi, cvi]

def extract_hrv_parameters(ecg_segment, fs=fs):
    rpeaks = detect_r_peaks_cwt(ecg_segment, sampling_rate=fs)
    if not check_ecg_quality(rpeaks):
        return [0]*7
    cleaned = fill_gaps_with_spline(rpeaks, fs)
    return calculate_hrv_metrics(cleaned, fs)

def sliding_window_hrv(ecg, fs=fs, Wo=Wo, S=S):
    Wo_s = int(Wo*fs)
    S_s = int(S*fs)
    X = []
    for start in range(0, len(ecg)-Wo_s, S_s):
        seg = ecg[start:start+Wo_s]
        X.append(extract_hrv_parameters(seg, fs))
    return np.array(X)
# print(f"[HRV] Total de janelas processadas: {len(X)}")

#################################################################################################
# LABELS
#################################################################################################

def gerar_labels_inter_pre(ecg_len, fs, seizure_onsets, pre_ictal_sec=pre_ictal_sec):
    labels = np.full(ecg_len, -1)
    for onset in seizure_onsets:
        onset_samp = int(onset * fs)
        pre_start = max(0, onset_samp - int(pre_ictal_sec * fs))
        labels[pre_start:onset_samp] = 1
    return labels

def gerar_labels_por_janela(labels, Wo=Wo, S=S, fs=fs):
    Wo_s = int(Wo*fs)
    S_s = int(S*fs)
    y = []
    for start in range(0, len(labels)-Wo_s, S_s):
        janela = labels[start:start+Wo_s]
        label_janela = 1 if np.any(janela==1) else -1
        y.append(label_janela)
    return np.array(y)

#################################################################################################
# CARREGAMENTO DE PACIENTES
#################################################################################################

def carregar_todos_pacientes(base_path, n_pacientes=None):
    registros = sorted([f.replace(".hea","") for f in os.listdir(base_path) if f.endswith(".hea")])
    if n_pacientes is not None:
        registros = registros[:n_pacientes]
    all_X, all_y = [], []
    for rec in registros:
        # print(f"\n[DATA] Processando paciente {i+1}/{len(registros)} → {rec}")
        path_record = os.path.join(base_path, rec)
        record = wfdb.rdrecord(path_record)
        ecg = record.p_signal[:,0]
        fs = record.fs
        ecg = butterworth_filter(ecg, fs)
        path_seiz = os.path.join(base_path, rec+".seizures")
        with open(path_seiz,"r") as f:
            onsets = [float(line.strip().split()[0]) for line in f]
        labels_amostra = gerar_labels_inter_pre(len(ecg), fs, onsets)
        X = sliding_window_hrv(ecg, fs)
        y = gerar_labels_por_janela(labels_amostra)
        all_X.append(X)
        all_y.append(y)
    return np.vstack(all_X), np.hstack(all_y)

#################################################################################################
# PCA (para teste)
#################################################################################################

def aplicar_pca_com_cov(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    pca = PCA(n_components=0.85)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    return X_train_pca, X_test_pca, scaler, pca

#################################################################################################
# MÉTRICAS
#################################################################################################

def calcular_metricas(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    fp = cm[0,1]/np.sum(cm[0,:]) if np.sum(cm[0,:])>0 else 0
    sensibilidade = recall[1] if len(recall)>1 else 0
    return {'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'fp_rate': fp,
            'sensibilidade': sensibilidade,
            'confusion_matrix': cm}

#################################################################################################
# PREDITOR CONTÍNUO
#################################################################################################

def preditor_continuo(X_raw, modelo, scaler, pca, limiar_alerta=0.65, save_fig=False, fig_name='predicao_continua.png'):
    X_scaled = scaler.transform(X_raw)
    X_pca = pca.transform(X_scaled)
    probs = modelo.predict_proba(X_pca)[:, 1]
    alertas = [(i, p) for i, p in enumerate(probs) if p >= limiar_alerta]

    plt.figure(figsize=(12,4))
    plt.plot(probs, marker='o')
    plt.axhline(y=limiar_alerta, linestyle='--', linewidth=1.5, label=f'Limiar alerta = {limiar_alerta}')
    plt.xlabel('Índice da Janela')
    plt.ylabel('Probabilidade de Pré-ictal')
    plt.title('Predição Contínua - Probabilidade de Pré-ictal por Janela')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save_fig:
        plt.savefig(fig_name)
    plt.show()

    return probs, alertas

#################################################################################################
# EXECUÇÃO PRINCIPAL
#################################################################################################

if __name__ == '__main__':
    print("\n================ INICIALIZAÇÃO DO SISTEMA =================")
    print("Sistema iniciado com sucesso na Raspberry Pi.")
    # base_path = "/home/pi/MicproRaspberry/synthetic_hrv_bigsep"  # ajuste para Raspberry Pi
    # n_pacientes = 2
    
    
    base_path = "/home/pi/MicproRaspberry/synthetic_hrv_bigsep"  # ajuste para Raspberry Pi
    # base_path = "C:/Nova_tentativa_tcc/synthetic_hrv_bigsep"

    PACIENTE_ID = "synth_02"
    registros = [PACIENTE_ID]
    
    # # Validação silenciosa
    # arquivo_hea = os.path.join(base_path, PACIENTE_ID + ".hea")
    # arquivo_seiz = os.path.join(base_path, PACIENTE_ID + ".seizures")

    # if not os.path.exists(arquivo_hea):
    # raise FileNotFoundError(f"Arquivo não encontrado: {arquivo_hea}")

    # if not os.path.exists(arquivo_seiz):
    # raise FileNotFoundError(f"Arquivo não encontrado: {arquivo_seiz}")
    
    n_pacientes = 1
    
    print("\n[1/7] Carregando modelo treinado (SVM + Scaler + PCA)...")
    # Carregar modelo treinado
    data = joblib.load("modelo_svm.joblib")
    modelo_svm = data['modelo']
    scaler = data['scaler']
    pca = data['pca']
    print("Modelo carregado com sucesso.")

    print("\n[2/7] Carregando e processando dados ECG...")
    # Carregar dados de teste
    X_raw, y = carregar_todos_pacientes(base_path, n_pacientes)
   

    print("\n[3/7] Realizando divisão treino/teste (70/30)...")
    # Split 70/30
    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.30, random_state=42, stratify=y
    )
    print("Divisão concluída.")
    
    print("\n[4/7] Aplicando normalização e PCA (modelo treinado)...")
    # PCA (usar scaler e pca do modelo)
    X_test_scaled = scaler.transform(X_test)
    X_test_pca = pca.transform(X_test_scaled)
    print(f"Número de componentes PCA: {X_test_pca.shape[1]}")

    print("\n[5/7] Executando predição com SVM...")
    # Predição
    y_pred = modelo_svm.predict(X_test_pca)
    print("Predição finalizada.")

    print("\n[6/7] Calculando métricas de desempenho...")
    # Métricas
    metricas = calcular_metricas(y_test, y_pred)
    print("\n=== Métricas do Modelo ===")
    print(f"Acurácia: {metricas['accuracy']:.3f}")
    print(f"Precision por classe: {metricas['precision']}")
    print(f"Recall por classe: {metricas['recall']}")
    print(f"F1-score por classe: {metricas['f1']}")
    print(f"Sensibilidade: {metricas['sensibilidade']:.3f}")
    print(f"Taxa de falsos positivos: {metricas['fp_rate']:.3f}")

    print("\nGerando matriz de confusão...")
    # Matriz de confusão
    cm = metricas['confusion_matrix']
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Interictal','Pré-ictal'], 
                yticklabels=['Interictal','Pré-ictal'])
    plt.xlabel('Predito')
    plt.ylabel('Verdadeiro')
    plt.title('Matriz de Confusão')
    plt.show()

    print("\n[7/7] Monitorando uso de memória...")
    # Memória total usada
    process = psutil.Process()
    mem_info = process.memory_full_info()
    print(f"\nMemória total usada (USS + compartilhada): {mem_info.uss / 1024**2:.2f} MB")
    print(f"Memória virtual total: {mem_info.vms / 1024**2:.2f} MB")
    
#     # ---------- PCA 2D - após processamento (CWT incluso) ----------
# X_scaled = scaler.transform(X_raw)  # Usa o scaler do treinamento
# X_pca_all = pca.transform(X_scaled)  # Usa o PCA do treinamento

# plt.figure(figsize=(8,6))
# for label_val, label_name in zip([-1,1], ['Interictal','Pré-ictal']):
#     idx = y==label_val
#     plt.scatter(X_pca_all[idx,0], X_pca_all[idx,1], label=label_name, alpha=0.6)
# plt.xlabel('Componente Principal 1')
# plt.ylabel('Componente Principal 2')
# plt.title('PCA 2D - Todas as janelas (pós-CWT)')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# # ---------- SINAL DO ÚLTIMO PACIENTE ----------
# ultimo_ecg = ecg  # último paciente processado
# tempo = np.arange(len(ultimo_ecg)) / fs

# # ---------- CWT ----------
# scales = np.arange(1, 20)
# coeffs, freqs = pywt.cwt(ultimo_ecg, scales, 'mexh')

# # ---------- PLOT LADO A LADO ----------
# plt.figure(figsize=(14,5))

# # Sinal original
# plt.subplot(1,2,1)
# plt.plot(tempo, ultimo_ecg)
# plt.xlabel('Tempo (s)')
# plt.ylabel('Amplitude ECG')
# plt.title('Sinal ECG Original')
# plt.grid(True)

# # CWT
# plt.subplot(1,2,2)
# plt.imshow(np.abs(coeffs), extent=[0, tempo[-1], scales[-1], scales[0]], cmap='jet', aspect='auto')
# plt.colorbar(label='Amplitude CWT')
# plt.xlabel('Tempo (s)')
# plt.ylabel('Escalas')
# plt.title('CWT do ECG')
# plt.tight_layout()
# plt.show()


# # ECG filtrado
# ecg_filtrado = butterworth_filter(ultimo_ecg, fs)

# # Sinal filtrado
# plt.subplot(1,2,2)
# plt.plot(tempo, ecg_filtrado)
# plt.xlabel('Tempo (s)')
# plt.ylabel('Amplitude ECG')
# plt.title('ECG Filtrado (Butterworth)')
# plt.grid(True)

# plt.tight_layout()
# plt.show()

