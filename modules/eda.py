"""
modules/eda.py
Análise Exploratória de Dados (EDA) com Plotly.
Versão Light: Branco, Cinza Claro, Preto e Vermelho.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


# ─── Paleta de cores (Light) ──────────────────────────────────────────────────
COLORS = {
    "primary": "#E63946",  # Vermelho
    "secondary": "#457B9D", # Azul (usado como secundário discreto)
    "accent": "#F4A261",    # Laranja
    "bg": "#FFFFFF",        # Branco
    "surface": "#F0F2F6",   # Cinza Claro
    "text": "#000000",      # Preto
}

PALETTE = [
    "#E63946", "#1D3557", "#457B9D", "#A8DADC",
    "#F4A261", "#2A9D8F", "#264653", "#6D597A",
]


def update_fig_layout(fig, title: str):
    """Padroniza o layout das figuras para o tema claro."""
    fig.update_layout(
        title=f"<b>{title}</b>",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["text"]),
        title_font_size=18,
        xaxis=dict(gridcolor="rgba(0,0,0,0.1)", tickfont=dict(color=COLORS["text"])),
        yaxis=dict(gridcolor="rgba(0,0,0,0.1)", tickfont=dict(color=COLORS["text"])),
        legend=dict(font=dict(color=COLORS["text"])),
    )
    return fig


def chart_volume_por_mes(df: pd.DataFrame):
    """Volume de transcrições por mês."""
    agg = df.groupby("MES_ANO").size().reset_index(name="VOLUME")
    agg = agg.sort_values("MES_ANO")
    fig = px.bar(
        agg, x="MES_ANO", y="VOLUME",
        color_discrete_sequence=[COLORS["primary"]],
        labels={"MES_ANO": "Mês/Ano", "VOLUME": "Qtd. Ligações"},
    )
    return update_fig_layout(fig, "📅 Volume de Ligações por Mês")


def chart_ligacoes_por_cliente(df: pd.DataFrame):
    """Quantidade de ligações por cliente."""
    agg = df.groupby("ID_CLIENTE").size().reset_index(name="N_LIGACOES")
    agg = agg.sort_values("N_LIGACOES", ascending=False)
    fig = px.bar(
        agg, x="ID_CLIENTE", y="N_LIGACOES",
        color="N_LIGACOES",
        color_continuous_scale=[[0, COLORS["secondary"]], [1, COLORS["primary"]]],
        labels={"ID_CLIENTE": "Cliente", "N_LIGACOES": "Ligações"},
    )
    fig = update_fig_layout(fig, "👤 Ligações por Cliente")
    fig.update_coloraxes(showscale=False)
    return fig


def chart_distribuicao_motivos(df_classified: pd.DataFrame, title="🎯 Distribuição de Macro Motivos"):
    """Distribuição de macro motivos classificados."""
    if df_classified is None or "MACRO_MOTIVO" not in df_classified.columns:
        return None
    agg = df_classified["MACRO_MOTIVO"].value_counts().reset_index()
    agg.columns = ["MACRO_MOTIVO", "COUNT"]
    fig = px.pie(
        agg, names="MACRO_MOTIVO", values="COUNT",
        color_discrete_sequence=PALETTE,
        hole=0.45,
    )
    return update_fig_layout(fig, title)


def chart_heatmap_dia_hora(df: pd.DataFrame):
    """Heatmap de ligações por dia da semana x hora."""
    df2 = df.copy()
    df2["HORA"] = df2["DATETIME_TRANSCRICAO_LIGACAO"].dt.hour
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    pt = df2.pivot_table(index="DIA_SEMANA", columns="HORA", aggfunc="size", fill_value=0)
    pt = pt.reindex([d for d in order if d in pt.index])

    fig = go.Figure(data=go.Heatmap(
        z=pt.values,
        x=[f"{h}h" for h in pt.columns],
        y=pt.index.tolist(),
        colorscale=[[0, "#F0F2F6"], [0.5, "#457B9D"], [1, "#E63946"]],
        showscale=True,
    ))
    return update_fig_layout(fig, "🕐 Concentração de Ligações (Dia × Hora)")


def chart_sankey_jornada(df_classified: pd.DataFrame):
    """Visualização Sankey do fluxo de motivos na jornada."""
    if df_classified is None or "MACRO_MOTIVO" not in df_classified.columns:
        return None
    
    # Gerar pares Origem -> Destino na sequência do cliente
    df_sorted = df_classified.sort_values(["ID_CLIENTE", "DATETIME_TRANSCRICAO_LIGACAO"])
    df_sorted["NEXT_MOTIVO"] = df_sorted.groupby("ID_CLIENTE")["MACRO_MOTIVO"].shift(-1)
    
    # Adicionar evento final de Churn para a última ligação
    df_sorted["NEXT_MOTIVO"] = df_sorted["NEXT_MOTIVO"].fillna("CHURN")
    
    flows = df_sorted.groupby(["MACRO_MOTIVO", "NEXT_MOTIVO"]).size().reset_index(name="VALUE")
    
    all_nodes = list(set(flows["MACRO_MOTIVO"]) | set(flows["NEXT_MOTIVO"]))
    node_map = {node: i for i, node in enumerate(all_nodes)}
    
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15, thickness=20, line=dict(color="black", width=0.5),
            label=all_nodes,
            color=[COLORS["primary"] if n == "CHURN" else COLORS["secondary"] for n in all_nodes]
        ),
        link=dict(
            source=flows["MACRO_MOTIVO"].map(node_map),
            target=flows["NEXT_MOTIVO"].map(node_map),
            value=flows["VALUE"]
        )
    )])
    return update_fig_layout(fig, "🌊 Fluxo da Jornada (Motivo 1 → Motivo 2 → Churn)")


def chart_correlacao_churn(df_d90: pd.DataFrame):
    """Correlação entre Nº de ligações e Score de Risco/Churn."""
    try:
        # Tenta gerar com trendline (requer statsmodels)
        fig = px.scatter(
            df_d90, x="N_LIGACOES_D90", y="TOKENS_TOTAL",
            size="SPAN_DIAS", color="N_LIGACOES_D90",
            color_continuous_scale="Reds",
            labels={"N_LIGACOES_D90": "Nº Ligações", "TOKENS_TOTAL": "Volume Termos"},
            trendline="ols", trendline_color_override=COLORS["primary"]
        )
    except Exception:
        # Fallback sem trendline se statsmodels falhar/não existir
        fig = px.scatter(
            df_d90, x="N_LIGACOES_D90", y="TOKENS_TOTAL",
            size="SPAN_DIAS", color="N_LIGACOES_D90",
            color_continuous_scale="Reds",
            labels={"N_LIGACOES_D90": "Nº Ligações", "TOKENS_TOTAL": "Volume Termos"}
        )
    return update_fig_layout(fig, "📊 Correlação: Frequência vs Volume de Texto")


def wordcloud_fig(df: pd.DataFrame, motive: str = None):
    """Gera nuvem de palavras das transcrições, opcionalmente filtrada por motivo."""
    try:
        from wordcloud import WordCloud
        import matplotlib.pyplot as plt
        import io as _io
        from PIL import Image

        STOPWORDS = {
            "que", "de", "o", "a", "os", "as", "um", "uma", "e", "em", "do",
            "da", "dos", "das", "com", "se", "para", "não", "por", "má", "me",
            "já", "mais", "meu", "minha", "eu", "está", "estou", "foi", "ser",
            "isso", "também", "quando", "muito", "ter", "mas", "como", "na",
            "no", "nos", "ao", "pelo", "pela", "ou", "este", "esta", "vocês",
            "ligando", "porque", "pelo", "pela", "estou", "estão", "esta", "este"
        }

        if motive and "MACRO_MOTIVO" in df.columns:
            subset = df[df["MACRO_MOTIVO"] == motive]
        else:
            subset = df

        if subset.empty: return None

        texto = " ".join(subset["TRANSCRICAO_LIGACAO_CLIENTE"].dropna().tolist())
        if not texto.strip(): return None

        wc = WordCloud(
            width=800, height=400,
            background_color="white",
            colormap="Reds",
            stopwords=STOPWORDS,
            max_words=100,
            collocations=False,
        ).generate(texto)

        buf = _io.BytesIO()
        plt.figure(figsize=(10, 5), facecolor="white")
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
        plt.close()
        buf.seek(0)
        return buf
    except Exception as e:
        return None
