"""
modules/agents.py
Arquitetura Multiagentes para análise de transcrições de churn usando LangGraph.
"""

import json
import re
import time
from typing import TypedDict, List, Optional, Annotated
import operator

from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

# ─── Configurações e Prompts ──────────────────────────────────────────────────

MACRO_MOTIVOS = [
    "Problema Técnico",
    "Concorrência",
    "Financeiro",
    "Mudança de Endereço/Localidade",
    "Atendimento",
    "Pessoal",
    "Não identificado",
]

FEW_SHOT_CLASSIFICACAO = """
Exemplos de classificação (few-shot):

TRANSCRIÇÃO: "Minha internet fica caindo toda hora, já mandaram técnico e não resolveu."
MACRO_MOTIVO: Problema Técnico
EVIDÊNCIA: "internet fica caindo toda hora"
SCORE: 97

TRANSCRIÇÃO: "A outra operadora está me oferecendo o dobro da velocidade pelo mesmo preço."
MACRO_MOTIVO: Concorrência
EVIDÊNCIA: "outra operadora está me oferecendo o dobro da velocidade"
SCORE: 95
"""

# ─── Definição do Estado do LangGraph ──────────────────────────────────────────

class AgentState(TypedDict):
    """Estado do grafo para processar a jornada de um cliente."""
    transcriptions: List[str]
    current_index: int
    results_per_call: Annotated[List[dict], operator.add]
    journey_metadata: dict  # n_ligacoes, span_dias
    diagnostico_final: dict
    llm_p_config: dict # Renamed to avoid collision with LangGraph 'config'

# ─── Utilitários ─────────────────────────────────────────────────────────────

def _get_llm(llm_p_config: dict):
    provider = llm_p_config.get("provider", "OpenAI")
    api_key = llm_p_config.get("api_key", "")
    model_name = llm_p_config.get("model", "gpt-4o-mini") # Default to 4o-mini
    
    if provider == "OpenAI":
        # api_key is the standard for langchain-openai >= 0.1.0
        return ChatOpenAI(api_key=api_key, model=model_name, temperature=0.2)
    elif provider == "Groq":
        # groq_api_key is standard for langchain-groq
        return ChatGroq(groq_api_key=api_key, model_name=model_name, temperature=0.2)
    return None

def _parse_json_safe(text: str) -> dict:
    match = re.search(r'\{.*?\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except:
            pass
    return {}

# ─── Nós do Grafo (Agentes) ───────────────────────────────────────────────────

def node_process_call(state: AgentState):
    """Processa uma ligação individual usando Auditor, Resumidor e Classificador."""
    idx = state["current_index"]
    transcr = state["transcriptions"][idx]
    llm = _get_llm(state["llm_p_config"])
    threshold = state["llm_p_config"].get("threshold", 90)
    
    if not llm:
        return {"current_index": idx + 1}

    # Auditor
    p1 = f"Auditor de qualidade (JSON): {transcr}"
    r1 = llm.invoke([SystemMessage(content="Responda JSON: {\"score\":int, \"ruido\":bool}"), HumanMessage(content=p1)])
    auditoria = _parse_json_safe(r1.content)
    
    # Resumidor
    p2 = f"Resumo estruturado (JSON): {transcr}"
    r2 = llm.invoke([SystemMessage(content="Responda JSON: {\"resumo\":str, \"sentimento\":str, \"evento\":str}"), HumanMessage(content=p2)])
    resumo = _parse_json_safe(r2.content)
    
    # Classificador
    motivos_str = ", ".join(MACRO_MOTIVOS)
    p3 = f"{FEW_SHOT_CLASSIFICACAO}\nClassifique (JSON): {transcr}\nMotivos: {motivos_str}"
    r3 = llm.invoke([SystemMessage(content="Responda JSON: {\"macro_motivo\":str, \"score\":int}"), HumanMessage(content=p3)])
    classif = _parse_json_safe(r3.content)
    
    if classif.get("score", 0) < threshold:
        classif["macro_motivo"] = "Não identificado"
        
    call_result = {
        "index": idx,
        "auditoria": auditoria,
        "resumo": resumo,
        "classificacao": classif
    }
    
    return {
        "results_per_call": [call_result],
        "current_index": idx + 1
    }

def node_diagnostician(state: AgentState):
    """Agente 4: Diagnóstico de Jornada (D-90)."""
    llm = _get_llm(state["llm_p_config"])
    meta = state["journey_metadata"]
    
    if not llm:
        return {"diagnostico_final": {"error": "LLM not configured"}}

    # Consolidar texto para o diagnóstico
    jornada_texto = "\n".join([f"Ligação {r['index']+1}: {r['resumo'].get('resumo', '')}" for r in state["results_per_call"]])
    
    prompt = f"""Analise a jornada de churn (D-90) e identifique padrões e causa raiz. 
Descreva a sequência de eventos, identifique padrões recorrentes e levante a hipótese da causa raiz.

Total ligações: {meta['n_ligacoes']} em {meta['span_dias']} dias.
Jornada:
{jornada_texto}

Responda SOMENTE JSON:
{{
  "diagnostico_sequencial": "string (ex: Cliente ligou relatando X, depois reclamou de Y...)",
  "padroes_recorrentes": "string (ex: três ligações sobre o mesmo problema técnico...)",
  "causa_raiz": "string",
  "ruptura": "string (momento de virada)",
  "confianca": 0-100,
  "escalada": bool
}}"""
    
    res = llm.invoke([HumanMessage(content=prompt)])
    diagnostico = _parse_json_safe(res.content)
    return {"diagnostico_final": diagnostico}

# ─── Construção do Grafo ──────────────────────────────────────────────────────

def create_churn_graph():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("process_call", node_process_call)
    workflow.add_node("diagnostician", node_diagnostician)
    
    workflow.set_entry_point("process_call")
    
    def route_after_call(state):
        if state["current_index"] < len(state["transcriptions"]):
            return "process_call"
        return "diagnostician"
    
    workflow.add_conditional_edges(
        "process_call",
        route_after_call,
        {
            "process_call": "process_call",
            "diagnostician": "diagnostician"
        }
    )
    
    workflow.add_edge("diagnostician", END)
    
    return workflow.compile()

# ─── Função de Interface para o App ───────────────────────────────────────────

def run_langgraph_pipeline(
    transcriptions: List[str],
    journey_metadata: dict,
    llm_p_config: dict
):
    """Executa o grafo do LangGraph e retorna o estado final."""
    app = create_churn_graph()
    
    initial_state = {
        "transcriptions": transcriptions,
        "current_index": 0,
        "results_per_call": [],
        "journey_metadata": journey_metadata,
        "diagnostico_final": {},
        "llm_p_config": llm_p_config
    }
    
    final_output = app.invoke(initial_state)
    return final_output
