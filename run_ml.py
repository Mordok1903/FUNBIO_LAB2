import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

FS = 256
DATA_DIR = Path("data")

def bandpass_filter(x, fs=256, lowcut=0.1, highcut=30.0, order=4):
    x = np.asarray(x, dtype=float)
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut/nyq, highcut/nyq], btype="band")
    return filtfilt(b, a, x)

def extract_features(x, prefix):
    x = np.asarray(x, dtype=float)
    return {
        f"{prefix}_mean": np.mean(x),
        f"{prefix}_mav": np.mean(np.abs(x)),
        f"{prefix}_rms": np.sqrt(np.mean(x**2)),
        f"{prefix}_std": np.std(x),
        f"{prefix}_max": np.max(x),
        f"{prefix}_min": np.min(x),
        f"{prefix}_ptp": np.ptp(x),
        f"{prefix}_wl": np.sum(np.abs(np.diff(x))),
    }

def process_subject(subject_id):
    print(f"Procesando {subject_id}...")
    eog_path = DATA_DIR / f"{subject_id}-EOG.csv"
    cs_path = DATA_DIR / f"{subject_id}-ControlSignal.csv"
    
    df_eog = pd.read_csv(eog_path)
    df_cs = pd.read_csv(cs_path)
    
    df_eog = df_eog.loc[:, ~df_eog.columns.str.contains(r"^Unnamed")].copy()
    df_cs = df_cs.loc[:, ~df_cs.columns.str.contains(r"^Unnamed")].copy()
    df_eog.columns = [c.strip() for c in df_eog.columns]
    df_cs.columns = [c.strip() for c in df_cs.columns]
    if "ControlSignal" not in df_cs.columns and len(df_cs.columns) == 1:
        df_cs.columns = ["ControlSignal"]
        
    n = min(len(df_eog), len(df_cs))
    df = pd.concat([df_eog.iloc[:n].reset_index(drop=True), df_cs[["ControlSignal"]].iloc[:n].reset_index(drop=True)], axis=1)
    df["EOGh"] = df["V3"] - df["V4"]
    df["EOGv"] = df["V1"] - df["V2"]
    
    # Preprocesamiento
    df["EOGh_filt"] = bandpass_filter(df["EOGh"], fs=FS)
    df["EOGv_filt"] = bandpass_filter(df["EOGv"], fs=FS)
    
    # Segmentacion
    df["block_id"] = (df["ControlSignal"] != df["ControlSignal"].shift()).cumsum()
    segments = df.groupby("block_id").agg(
        label=("ControlSignal", "first"),
        start_idx=("ControlSignal", lambda x: x.index[0]),
        end_idx=("ControlSignal", lambda x: x.index[-1]),
    ).reset_index(drop=True)
    
    WINDOW_SEC = 1
    OVERLAP = 0.5
    WIN_SIZE = int(WINDOW_SEC * FS)
    STEP = int(WIN_SIZE * (1 - OVERLAP))
    
    records = []
    for _, seg in segments.iterrows():
        label = int(seg["label"])
        start_seg = int(seg["start_idx"])
        end_seg = int(seg["end_idx"]) + 1
        if end_seg - start_seg < WIN_SIZE: continue
        for start in range(start_seg, end_seg - WIN_SIZE + 1, STEP):
            end = start + WIN_SIZE
            if df.loc[start:end-1, "ControlSignal"].nunique() != 1: continue
            records.append({
                "subject_id": subject_id,
                "label": label,
                "start_idx": start,
                "end_idx": end - 1,
                "EOGh_window": df.loc[start:end-1, "EOGh_filt"].to_numpy(),
                "EOGv_window": df.loc[start:end-1, "EOGv_filt"].to_numpy(),
            })
    windows_df = pd.DataFrame(records)
    
    # Features
    feature_rows = []
    for _, row in windows_df.iterrows():
        feats = {}
        feats.update(extract_features(row["EOGh_window"], "EOGh"))
        feats.update(extract_features(row["EOGv_window"], "EOGv"))
        feats["subject_id"] = row["subject_id"]
        feats["label"] = row["label"]
        feature_rows.append(feats)
        
    return pd.DataFrame(feature_rows)

subjects = [f"S{i}" for i in range(1, 8)]
all_features = pd.concat([process_subject(s) for s in subjects], ignore_index=True)

print("\nClases detectadas:")
print(all_features['label'].value_counts())

# Use S7 for testing
test_subject = "S7"
train_df = all_features[all_features['subject_id'] != test_subject]
test_df = all_features[all_features['subject_id'] == test_subject]

features_cols = [c for c in all_features.columns if c not in ["subject_id", "label", "start_idx", "end_idx"]]

X_train = train_df[features_cols]
y_train = train_df['label']

X_test = test_df[features_cols]
y_test = test_df['label']

print("\n--- Entrenamiento y comparación de modelos (2.5 puntos) ---")
print("Sujeto reservado para prueba:", test_subject)

# Model 1: Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_cv_scores = cross_val_score(rf, X_train, y_train, cv=5)
print(f"\nModelo 1: Random Forest Classifier")
print(f"Accuracy en Cross-Validation (5-fold): {rf_cv_scores.mean():.4f} +/- {rf_cv_scores.std():.4f}")

# Model 2: SVM
svm = SVC(kernel='rbf', probability=True, random_state=42)
svm_cv_scores = cross_val_score(svm, X_train, y_train, cv=5)
print(f"\nModelo 2: Support Vector Machine (RBF kernel)")
print(f"Accuracy en Cross-Validation (5-fold): {svm_cv_scores.mean():.4f} +/- {svm_cv_scores.std():.4f}")

print("\n--- Selección y evaluación del modelo final (1.5 puntos) ---")
best_model = rf if rf_cv_scores.mean() > svm_cv_scores.mean() else svm
best_model_name = "Random Forest" if rf_cv_scores.mean() > svm_cv_scores.mean() else "Support Vector Machine"
print(f"Modelo seleccionado: {best_model_name}")

best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy en Sujeto de Prueba ({test_subject}): {acc:.4f}")

print("\nReporte de Clasificación (Sujeto de prueba):")
print(classification_report(y_test, y_pred))

print("\nMatriz de Confusión (Sujeto de prueba):")
cm = confusion_matrix(y_test, y_pred)
print(cm)

joblib.dump(best_model, "modelo_grupo.joblib")
print("\nModelo guardado como modelo_grupo.joblib")

sample_features = test_df.iloc[:20].drop(columns=['subject_id'])
sample_features.to_csv("archivo_features.csv", index=False)
print("Archivo de prueba guardado como archivo_features.csv")
