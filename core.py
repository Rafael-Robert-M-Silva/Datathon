"""Dados + modelos preditivos — Passos Magicos (sem dependencia de Streamlit)."""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.inspection import permutation_importance

FEATURES = ["Defasagem", "IPP", "IPV", "IDA", "IPS", "IEG", "IAA"]
FEATURE_LABELS = {
    "Defasagem": "Defasagem atual (anos)",
    "IPP": "IPP · Psicopedagogico",
    "IPV": "IPV · Ponto de Virada",
    "IDA": "IDA · Desempenho",
    "IPS": "IPS · Psicossocial",
    "IEG": "IEG · Engajamento",
    "IAA": "IAA · Autoavaliacao",
}

# Aluno "constante" = presente em todos os anos do ciclo (2022-2024).
# Esta e a definicao usada no notebook (df_modelagem) e reproduz seus numeros.
CONSTANTE_MIN_ANOS = 3


def load_data(path="dados_consolidados.csv"):
    """Carrega o dataset consolidado e classifica cada registro em Novo/Constante (por ano)."""
    df = pd.read_csv(path)
    if "Ano ingresso" in df.columns:
        df["tipo_aluno"] = np.where(df["Ano"] <= df["Ano ingresso"], "Novo", "Constante")
    else:
        n_anos = df.groupby("RA")["Ano"].transform("nunique")
        df["tipo_aluno"] = np.where(n_anos >= 3, "Constante", "Novo")
    return df


def alunos_constantes(df):
    """Retorna apenas os registros de alunos presentes em todos os anos do ciclo."""
    n_ano = df.groupby("RA")["Ano"].transform("nunique")
    return df[n_ano >= CONSTANTE_MIN_ANOS].copy()


def build_training_frame(df):
    """Alvo de alerta precoce: o aluno estara defasado no ANO SEGUINTE?"""
    d = df.sort_values(["RA", "Ano"]).copy()
    d["def_next"] = d.groupby("RA")["Defasagem"].shift(-1)
    d = d.dropna(subset=["def_next"]).copy()
    d["risco"] = (d["def_next"] < 0).astype(int)
    return d


def train_models(df, apenas_constantes=True):
    """Treina Random Forest e Rede Neural (MLP com normalizacao).

    apenas_constantes=True -> o modelo aprende somente com alunos que
    permaneceram no programa por todo o ciclo (trajetoria longitudinal real).
    """
    base = alunos_constantes(df) if apenas_constantes else df.copy()
    d = build_training_frame(base)
    d[FEATURES] = d[FEATURES].apply(pd.to_numeric, errors="coerce")
    data = d.dropna(subset=["risco"]).copy()
    X, y = data[FEATURES], data["risco"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    rf = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(n_estimators=400, max_depth=8, min_samples_leaf=8,
                                       class_weight="balanced", random_state=42)),
    ])
    nn = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("clf", MLPClassifier(hidden_layer_sizes=(16, 8), activation="relu", alpha=1e-3,
                              max_iter=2000, early_stopping=True, n_iter_no_change=25,
                              random_state=42)),
    ])
    rf.fit(Xtr, ytr)
    nn.fit(Xtr, ytr)

    prob_rf = rf.predict_proba(Xte)[:, 1]
    prob_nn = nn.predict_proba(Xte)[:, 1]
    auc_rf = roc_auc_score(yte, prob_rf)
    auc_nn = roc_auc_score(yte, prob_nn)

    perm = permutation_importance(rf, Xte, yte, n_repeats=15, random_state=42, scoring="roc_auc")
    imp = pd.Series(perm.importances_mean, index=FEATURES).clip(lower=0).sort_values(ascending=False)

    fpr_rf, tpr_rf, _ = roc_curve(yte, prob_rf)
    fpr_nn, tpr_nn, _ = roc_curve(yte, prob_nn)

    meta = {
        "auc_rf": round(float(auc_rf), 3), "auc_nn": round(float(auc_nn), 3),
        "n": int(len(y)), "base_rate": round(float(y.mean()) * 100, 1),
        "importances": imp,
        "roc_rf": (fpr_rf.tolist(), tpr_rf.tolist()),
        "roc_nn": (fpr_nn.tolist(), tpr_nn.tolist()),
        "apenas_constantes": apenas_constantes,
        "n_ras_treino": int(base["RA"].nunique()),
    }
    return {"Random Forest": rf, "Rede Neural": nn}, meta


def predict_risk(model, values):
    row = pd.DataFrame([{f: values.get(f, np.nan) for f in FEATURES}])
    return float(model.predict_proba(row)[:, 1][0])


def risk_band(prob):
    if prob >= 0.66:
        return "Risco alto", "#E86A5C"
    if prob >= 0.40:
        return "Risco moderado", "#F2B14A"
    return "Risco baixo", "#35C08A"


from pathlib import Path
if __name__ == "__main__":
    # Caminho dinâmico baseado na localização atual do arquivo script
    BASE_DIR = Path(__file__).resolve().parent
    csv_path = BASE_DIR / "dados_consolidados.csv"

    df = load_data(csv_path)
    dc = alunos_constantes(df)
    print("Registros totais:", len(df), "| constantes (3 anos):", len(dc), "| RAs constantes:", dc.RA.nunique())
    models, meta = train_models(df)
    print("AUC RF:", meta["auc_rf"], "| AUC Rede Neural:", meta["auc_nn"], "| n pares:", meta["n"],
          "| RAs treino:", meta["n_ras_treino"], "| apenas_constantes:", meta["apenas_constantes"])
    print("Importancias RF:\n", meta["importances"].round(3))
