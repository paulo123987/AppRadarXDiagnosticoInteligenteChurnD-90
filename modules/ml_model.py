"""
modules/ml_model.py
Módulo de Machine Learning preditivo de churn.
Modelos: RandomForest, XGBoost, Logistic Regression.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, accuracy_score,
)
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


COLORS = {
    "primary": "#E63946",
    "secondary": "#457B9D",
    "accent": "#F4A261",
}


def build_features(df_d90: pd.DataFrame, df_classified: pd.DataFrame = None) -> pd.DataFrame:
    """
    Constrói feature matrix para o modelo preditivo.
    Features numéricas derivadas da jornada D-90.
    """
    feat = df_d90[["ID_CLIENTE", "N_LIGACOES_D90", "SPAN_DIAS",
                   "INTERVALO_MEDIO_DIAS", "TOKENS_TOTAL", "TOKENS_MEDIO"]].copy()

    # Feature: n_categorias_distintas e repeticao_motivo (se análise de agentes disponível)
    if df_classified is not None and "N_CATEGORIAS_DISTINTAS" in df_classified.columns:
        merge_cols = ["ID_CLIENTE", "N_CATEGORIAS_DISTINTAS", "REPETICAO_MOTIVO"]
        feat = feat.merge(
            df_classified[merge_cols],
            on="ID_CLIENTE", how="left"
        )
        feat["N_CATEGORIAS_DISTINTAS"] = feat["N_CATEGORIAS_DISTINTAS"].fillna(1)
        feat["REPETICAO_MOTIVO"] = feat["REPETICAO_MOTIVO"].fillna(0.5)
    else:
        feat["N_CATEGORIAS_DISTINTAS"] = 1
        feat["REPETICAO_MOTIVO"] = 0.5

    # Feature derivadas
    feat["TAXA_LIGACOES_DIA"] = feat["N_LIGACOES_D90"] / feat["SPAN_DIAS"].replace(0, 1)
    feat["SCORE_INTENSIDADE"] = (
        feat["N_LIGACOES_D90"] * 0.3
        + feat["TOKENS_TOTAL"] / 1000 * 0.2
        + feat["N_CATEGORIAS_DISTINTAS"] * 0.2
        + feat["REPETICAO_MOTIVO"] * 0.3
    )

    # Target: 1 = chrun de alto risco (> percentil 60 no score de intensidade)
    threshold = feat["SCORE_INTENSIDADE"].quantile(0.4)
    feat["TARGET_CHURN"] = (feat["SCORE_INTENSIDADE"] >= threshold).astype(int)

    return feat.set_index("ID_CLIENTE")


def train_models(feat: pd.DataFrame) -> dict:
    """
    Treina RandomForest, XGBoost (se disponível) e Logistic Regression.
    Retorna dicionário com resultados de cada modelo.
    """
    feature_cols = [
        "N_LIGACOES_D90", "SPAN_DIAS", "INTERVALO_MEDIO_DIAS",
        "TOKENS_TOTAL", "TOKENS_MEDIO", "N_CATEGORIAS_DISTINTAS",
        "REPETICAO_MOTIVO", "TAXA_LIGACOES_DIA", "SCORE_INTENSIDADE",
    ]
    X = feat[feature_cols].fillna(0)
    y = feat["TARGET_CHURN"]

    if X.shape[0] < 10:
        return {}

    # Split
    test_size = min(0.3, max(0.2, 5 / len(X)))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y if y.nunique() > 1 else None
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    models_def = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=5, random_state=42, class_weight="balanced"
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=500, random_state=42, class_weight="balanced"
        ),
    }
    if HAS_XGB:
        models_def["XGBoost"] = XGBClassifier(
            n_estimators=100, max_depth=4, random_state=42,
            use_label_encoder=False, eval_metric="logloss",
        )

    results = {}
    for name, mdl in models_def.items():
        try:
            use_scaled = name != "Random Forest" and not name.startswith("XGB")
            Xtr = X_train_s if use_scaled else X_train.values
            Xte = X_test_s if use_scaled else X_test.values

            mdl.fit(Xtr, y_train)
            preds = mdl.predict(Xte)
            proba = mdl.predict_proba(Xte)[:, 1] if hasattr(mdl, "predict_proba") else preds

            acc = accuracy_score(y_test, preds)
            auc = roc_auc_score(y_test, proba) if y_test.nunique() > 1 else 0.5
            report = classification_report(y_test, preds, output_dict=True, zero_division=0)

            # Feature importances
            importances = {}
            if hasattr(mdl, "feature_importances_"):
                importances = dict(zip(feature_cols, mdl.feature_importances_.tolist()))
            elif hasattr(mdl, "coef_"):
                importances = dict(zip(feature_cols, abs(mdl.coef_[0]).tolist()))

            # Probabilidades para todos os registros
            all_proba = mdl.predict_proba(
                scaler.transform(X.values) if use_scaled else X.values
            )[:, 1]

            results[name] = {
                "model": mdl,
                "accuracy": round(acc, 4),
                "auc": round(auc, 4),
                "report": report,
                "importance": importances,
                "all_prob": dict(zip(feat.index, all_proba.tolist())),
                "feature_cols": feature_cols,
            }
        except Exception as e:
            results[name] = {"error": str(e)}

    return results


def chart_feature_importance(importance_dict: dict, model_name: str):
    """Gráfico de importância de features."""
    if not importance_dict:
        return None
    df_imp = pd.DataFrame(
        {"Feature": list(importance_dict.keys()), "Importância": list(importance_dict.values())}
    ).sort_values("Importância", ascending=True)

    fig = px.bar(
        df_imp, x="Importância", y="Feature", orientation="h",
        title=f"🔍 Importância das Features — {model_name}",
        color="Importância",
        color_continuous_scale=[[0, "#457B9D"], [1, "#E63946"]],
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#FFFFFF"),
        title_font_size=16,
        xaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
        showlegend=False,
    )
    return fig


def chart_risk_scores(all_prob: dict, title: str = "Score de Risco por Cliente"):
    """Gráfico de score de risco por cliente."""
    df_r = pd.DataFrame(
        {"ID_CLIENTE": list(all_prob.keys()), "SCORE_RISCO": list(all_prob.values())}
    ).sort_values("SCORE_RISCO", ascending=False)

    df_r["COR"] = df_r["SCORE_RISCO"].apply(
        lambda x: "#E63946" if x > 0.7 else ("#F4A261" if x > 0.4 else "#2A9D8F")
    )

    fig = go.Figure(go.Bar(
        x=df_r["ID_CLIENTE"],
        y=df_r["SCORE_RISCO"],
        marker_color=df_r["COR"].tolist(),
        text=[f"{v:.0%}" for v in df_r["SCORE_RISCO"]],
        textposition="outside",
        textfont=dict(color="#FFFFFF"),
        hovertemplate="<b>%{x}</b><br>Score: %{y:.1%}<extra></extra>",
    ))
    fig.update_layout(
        title=f"🎯 {title}",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#FFFFFF"),
        title_font_size=16,
        xaxis=dict(tickangle=-45, gridcolor="rgba(255,255,255,0.1)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.1)", tickformat=".0%", range=[0, 1.15]),
    )
    return fig
