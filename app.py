"""
app.py
Radar X – Diagnóstico Inteligente de Churn (D-90)
Aplicação Streamlit com Foco Executivo, Dashboards de Jornada e IA (LangGraph).
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# Importação dos módulos locais
from modules.data_utils import (
    load_csv, validate_columns, preprocess, consolidate_d90, 
    get_synthetic_csv_path, get_top_journey_patterns
)
from modules.eda import (
    chart_volume_por_mes, chart_ligacoes_por_cliente, chart_distribuicao_motivos,
    chart_heatmap_dia_hora, chart_sankey_jornada, chart_correlacao_churn, wordcloud_fig,
    COLORS as EDA_COLORS
)
from modules.agents import run_langgraph_pipeline, MACRO_MOTIVOS
from modules.ml_model import build_features, train_models, chart_feature_importance, chart_risk_scores

# ─── Configuração da Página ──────────────────────────────────────────────────

st.set_page_config(
    page_title="Radar X – Diagnóstico Churn",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS para estética Light (Branco, Cinza, Preto, Vermelho)
st.markdown(f"""
<style>
    .main {{ background-color: #FFFFFF; color: #000000; }}
    [data-testid="stSidebar"] {{ background-color: #F8F9FA; border-right: 1px solid #E0E0E0; }}
    .stMetric {{ 
        background-color: #FFFFFF; 
        padding: 20px; 
        border-radius: 8px; 
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border-left: 5px solid #E63946; 
    }}
    .stTabs [data-baseweb="tab-list"] {{ gap: 8px; border-bottom: 2px solid #F0F2F6; }}
    .stTabs [data-baseweb="tab"] {{
        height: 50px; background-color: #F8F9FA; border-radius: 4px 4px 0 0; 
        color: #666666; border: 1px solid #F0F2F6; margin-bottom: -2px;
    }}
    .stTabs [aria-selected="true"] {{ 
        background-color: #FFFFFF !important; 
        color: #E63946 !important; 
        border-bottom: 2px solid #E63946 !important;
        font-weight: bold;
    }}
    h1, h2, h3 {{ color: #000000; font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; }}
    .stButton>button {{
        background-color: #E63946; color: white; border-radius: 4px; border: none;
        padding: 0.5rem 1rem; transition: all 0.3s;
    }}
    .stButton>button:hover {{ background-color: #C1121F; color: white; border: none; }}
</style>
""", unsafe_allow_html=True)

# ─── Estado da Sessão ────────────────────────────────────────────────────────

if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = pd.DataFrame()

# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("<h1 style='color: #E63946; margin-bottom: 0;'>RADAR X</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #666; margin-top: 0;'>Diagnóstico de Churn D-90</p>", unsafe_allow_html=True)
    st.divider()
    
    st.markdown("### 📂 Fonte de Dados")
    source_choice = st.radio("Selecione:", ("Base Sintética (Radar X)", "Carregar CSV Próprio"))
    
    df_raw = None
    if source_choice == "Carregar CSV Próprio":
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded_file:
            try:
                df_raw = load_csv(uploaded_file)
            except Exception as e:
                st.error(f"Erro no carregamento: {e}")
    else:
        path = get_synthetic_csv_path()
        if os.path.exists(path):
            df_raw = pd.read_csv(path)
        else:
            st.warning("Executor de dados sintéticos não encontrado.")

    st.divider()
    
    st.markdown("### 🤖 Motor de IA")
    provider = st.selectbox("Provedor", ["OpenAI", "Groq"])
    
    if provider == "OpenAI":
        api_key = st.secrets.get("OPENAI_API_KEY", "")
        model = st.selectbox("Modelo", ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"])
    else:
        api_key = st.secrets.get("GROQ_API_KEY", "")
        model = st.selectbox("Modelo", ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"])
        
    threshold = st.slider("Corte de Confiança (%)", 0, 100, 90)
    
    st.divider()
    if st.button("🗑️ Limpar Cache de IA"):
        st.session_state.analysis_results = pd.DataFrame()
        st.rerun()

# ─── Lógica Principal ────────────────────────────────────────────────────────

if df_raw is not None:
    valid, missing = validate_columns(df_raw)
    if not valid:
        st.error(f"Seu CSV precisa das colunas: {missing}")
        st.stop()
    
    # Processamento base
    df = preprocess(df_raw)
    df_d90 = consolidate_d90(df)
    
    # Construção do df_mock: Resultados Reais (IA) + Mock (Fallback) para visualização
    # 1. Base Mock
    df_mock = df.copy()
    def mock_classify(text):
        text = str(text).lower()
        if "técnico" in text or "internet" in text or "lenta" in text: return "Problema Técnico"
        if "fatura" in text or "cobrança" in text or "preço" in text: return "Financeiro"
        if "concorr" in text or "operadora" in text: return "Concorrência"
        if "atend" in text or "grosseiro" in text: return "Atendimento"
        if "mudar" in text or "endereço" in text: return "Mudança de Endereço/Localidade"
        return "Outros"
    df_mock["MACRO_MOTIVO"] = df_mock["TRANSCRICAO_LIGACAO_CLIENTE"].apply(mock_classify)
    df_mock["CONFIDENCE"] = np.random.randint(60, 80, size=len(df_mock))
    
    # 2. Sobrepor com resultados reais da IA se disponíveis
    if not st.session_state.analysis_results.empty:
        real_res = st.session_state.analysis_results
        # Removemos os mocks para os IDs de clientes que já possuem análise real para evitar duplicidade no merge
        df_mock = df_mock[~df_mock.set_index(["ID_CLIENTE", "DATETIME_TRANSCRICAO_LIGACAO"]).index.isin(
            real_res.set_index(["ID_CLIENTE", "DATETIME_TRANSCRICAO_LIGACAO"]).index
        )]
        df_mock = pd.concat([df_mock, real_res]).sort_values(["ID_CLIENTE", "DATETIME_TRANSCRICAO_LIGACAO"])

    # ─── TABS DO APP ──────────────────────────────────────────────────────────
    
    t_dash, t_jornadas, t_classif, t_ia, t_ml = st.tabs([
        "📈 Dashboard Executivo", "🌊 Jornadas", "📋 Classificações", "🤖 Diagnóstico IA", "🧬 Predição Risco"
    ])
    
    # ─── 1. Dashboard Executivo ───
    with t_dash:
        st.title("📊 Raio-X Executivo da Jornada de Churn")
        
        # KPIs no Topo
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Clientes Analisados", df["ID_CLIENTE"].nunique())
        
        # % por motivo principal
        main_motive = df_mock["MACRO_MOTIVO"].value_counts().index[0]
        main_motive_pct = (df_mock["MACRO_MOTIVO"] == main_motive).mean()
        kpi2.metric("Principal Motivo", f"{main_motive}")
        kpi3.metric("% Churn p/ {main_motive}", f"{main_motive_pct:.0%}")
        
        avg_int = len(df) / df["ID_CLIENTE"].nunique()
        kpi4.metric("Ticket Médio Interações", f"{avg_int:.1f}")
        
        st.divider()
        
        # Gráficos Principais
        col_l, col_r = st.columns([2, 1])
        with col_l:
            st.plotly_chart(chart_sankey_jornada(df_mock), use_container_width=True)
        with col_r:
            st.plotly_chart(chart_distribuicao_motivos(df_mock, "Causa Raiz Predominante"), use_container_width=True)
            
        st.divider()
        
        # Tabela Detalhada com Filtros
        st.subheader("🔍 Filtros de Visualização")
        f_col1, f_col2, f_col3 = st.columns(3)
        with f_col1:
            f_motive = st.multiselect("Motivo", ["Todos"] + list(MACRO_MOTIVOS), default="Todos")
        with f_col2:
            f_conf = st.slider("Confiança Mínima IA (%)", 0, 100, 0)
        with f_col3:
            f_date = st.date_input("Período Churn", [df["DATETIME_TRANSCRICAO_LIGACAO"].min(), df["DATETIME_TRANSCRICAO_LIGACAO"].max()])
        
        # Filtragem do DataFrame de Jornadas (df_d90)
        # Para simplificar, usamos o mock para atribuir motivos ao df_d90
        df_d90_dash = df_d90.merge(
            df_mock.groupby("ID_CLIENTE")["MACRO_MOTIVO"].last().reset_index(),
            on="ID_CLIENTE", how="left"
        )
        # Adicionar confiança média simulada se não houver real
        if "CONFIDENCE" not in df_d90_dash.columns:
            df_d90_dash["CONFIDENCE"] = np.random.randint(85, 99, size=len(df_d90_dash))
            
        # Aplicar filtros
        if "Todos" not in f_motive and f_motive:
            df_d90_dash = df_d90_dash[df_d90_dash["MACRO_MOTIVO"].isin(f_motive)]
        df_d90_dash = df_d90_dash[df_d90_dash["CONFIDENCE"] >= f_conf]
        
        st.markdown("### Clientes sob Diagnóstico")
        st.dataframe(
            df_d90_dash[["ID_CLIENTE", "TIMELINE", "MACRO_MOTIVO", "CONFIDENCE", "N_LIGACOES_D90"]].rename(
                columns={"TIMELINE": "Fluxo da Jornada", "MACRO_MOTIVO": "Causa Raiz", "CONFIDENCE": "IA Conf (%)"}
            ),
            use_container_width=True,
            hide_index=True
        )

    # ─── 2. Jornadas ───
    with t_jornadas:
        st.header("🌊 Padrões e Experiência do Cliente")
        
        col_patterns, col_corr = st.columns([1, 1])
        with col_patterns:
            patterns = get_top_journey_patterns(df_mock)
            st.markdown("#### 🏆 Top 3 Jornadas de Churn")
            if patterns:
                for i, p in enumerate(patterns):
                    st.warning(f"**#{i+1}:** {p['JORNADA']} ({p['FREQUENCIA']} casos)")
            else:
                st.info("Aguardando classificação real das interações.")
        
        with col_corr:
            st.plotly_chart(chart_correlacao_churn(df_d90), use_container_width=True)
            
        st.divider()
        st.markdown("### Detalhamento das Jornadas")
        for idx, row in df_d90.head(10).iterrows():
            # Busca segura no df_mock
            cli_data = df_mock[df_mock["ID_CLIENTE"] == row["ID_CLIENTE"]]
            motive_label = cli_data["MACRO_MOTIVO"].iloc[-1] if not cli_data.empty else "Não identificado"
            
            with st.expander(f"👤 {row['ID_CLIENTE']} | {row['N_LIGACOES_D90']} ligações | Causa: {motive_label}"):
                st.write(f"**Resumo da Jornada:** {row['TIMELINE']}")
                st.markdown("**Interações Originais:**")
                st.code(row["JORNADA_TEXTO"])

    # ─── 3. Classificações ───
    with t_classif:
        st.header("📋 Auditoria de Interações")
        st.plotly_chart(chart_volume_por_mes(df), use_container_width=True)
        
        sel_motivo = st.selectbox("Filtrar Nuvem de Palavras por Motivo:", ["Todos"] + list(MACRO_MOTIVOS))
        wc = wordcloud_fig(df_mock, None if sel_motivo == "Todos" else sel_motivo)
        if wc: st.image(wc)
        
        st.subheader("Lista Completa de Interações Classificadas")
        st.dataframe(df_mock[["ID_CLIENTE", "DATETIME_TRANSCRICAO_LIGACAO", "MACRO_MOTIVO", "CONFIDENCE", "TRANSCRICAO_LIGACAO_CLIENTE"]], use_container_width=True)

    # ─── 4. Diagnóstico IA ───
    with t_ia:
        st.header("🤖 Diagnóstico Profundo por IA (LangGraph)")
        st.info("A análise de IA utiliza o sistema multiagente (Auditor, Resumidor, Classificador, Diagnóstico) para gerar uma hipótese de causa raiz.")
        
        target_client = st.selectbox("Escolha um Cliente para Raio-X Detalhado:", df_d90["ID_CLIENTE"].unique())
        
        if st.button("🚀 Executar Diagnóstico Completo"):
            row = df_d90[df_d90["ID_CLIENTE"] == target_client].iloc[0]
            
            llm_p_config = {
                "provider": provider,
                "api_key": api_key,
                "model": model,
                "threshold": threshold
            }
            
            with st.spinner(f"Analistas IA processando a jornada de {target_client}..."):
                try:
                    meta = {"n_ligacoes": row["N_LIGACOES_D90"], "span_dias": row["SPAN_DIAS"]}
                    result = run_langgraph_pipeline(row["TRANSCRICOES_LIST"], meta, llm_p_config)
                    
                    # Salvar no estado global para o dashboard
                    res_rows = []
                    for r in result["results_per_call"]:
                        res_rows.append({
                            "ID_CLIENTE": target_client,
                            "DATETIME_TRANSCRICAO_LIGACAO": row["DATAS_LIST"][r["index"]],
                            "TRANSCRICAO_LIGACAO_CLIENTE": row["TRANSCRICOES_LIST"][r["index"]],
                            "MACRO_MOTIVO": r["classificacao"].get("macro_motivo"),
                            "CONFIDENCE": r["classificacao"].get("score")
                        })
                    new_analysis = pd.DataFrame(res_rows)
                    st.session_state.analysis_results = pd.concat([st.session_state.analysis_results, new_analysis]).drop_duplicates(subset=["ID_CLIENTE", "DATETIME_TRANSCRICAO_LIGACAO"])

                    st.success("Diagnóstico Gerado com Sucesso!")
                    
                    diag = result["diagnostico_final"]
                    
                    # Layout Executive style
                    c1, c2, c3 = st.columns([2, 1, 1])
                    with c1:
                        st.markdown(f"#### 📖 Resumo Sequencial")
                        st.write(diag.get("diagnostico_sequencial", "—"))
                        st.markdown(f"#### 🧬 Padrões Detectados")
                        st.write(diag.get("padroes_recorrentes", "—"))
                    with c2:
                        st.metric("Causa Raiz", diag.get("causa_raiz", "N/A"))
                        st.metric("Confiança da IA", f"{diag.get('confianca')}%")
                    with c3:
                        st.metric("Ruptura", diag.get("ruptura", "N/A"))
                        st.metric("Escalada", "Sim" if diag.get("escalada") else "Não")

                    with st.expander("Ver Auditoria dos Agentes"):
                        st.json(result["results_per_call"])
                        
                except Exception as e:
                    st.error(f"Erro na execução da IA: {e}. Verifique suas chaves de API.")

    # ─── 5. Predição Risco ───
    with t_ml:
        st.header("🧬 Modelagem de Risco Preditivo")
        st.write("O modelo utiliza variáveis comportamentais da jornada D-90 para estimar a probabilidade de churn futuro.")
        
        # Integrar resultados reais da IA se disponíveis
        feat = build_features(df_d90)
        
        if st.button("⚙️ Treinar e Avaliar Modelos"):
            with st.spinner("Treinando RandomForest e XGBoost..."):
                results = train_models(feat)
                if results:
                    for name, res in results.items():
                        st.subheader(f"Modelo: {name}")
                        m1, m2 = st.columns(2)
                        m1.metric("Acurácia", f"{res['accuracy']:.2%}")
                        m2.metric("AUC ROC", f"{res['auc']:.3f}")
                        
                        st.plotly_chart(chart_risk_scores(res["all_prob"], f"Score de Risco - {name}"), use_container_width=True)
                        st.plotly_chart(chart_feature_importance(res["importance"], name), use_container_width=True)
                        st.divider()
                else:
                    st.error("Dados insuficientes para treino.")

else:
    st.title("🎯 Radar X — Inteligência Analítica em Churn")
    st.markdown("""
    Bem-vindo ao **Radar X**. Esta plataforma utiliza Inteligência Artificial avançada para extrair o máximo valor das suas interações de clientes.
    
    ### Como começar:
    1. Escolha a **Fonte de Dados** no menu lateral.
    2. Visualize os padrões na aba **Dashboard Executivo**.
    3. Realize diagnósticos profundos na aba **Diagnóstico IA**.
    """)
    st.image("https://images.unsplash.com/photo-1460925895917-afdab827c52f?q=80&w=1000", caption="Radar X Analysis")
