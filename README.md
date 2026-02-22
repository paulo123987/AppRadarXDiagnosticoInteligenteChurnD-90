# AppRadarXDiagnosticoInteligenteChurnD-90                                                                                                                                # Radar X – Diagnóstico Inteligente de Churn (D-90)

Aplicação avançada de análise de churn que utiliza **IA Generativa (Multiagentes)**, **LangGraph** e **Machine Learning** para identificar a causa raiz do cancelamento em jornadas de 90 dias antes do evento (D-90).

## 🚀 Como Executar Localmente

### 1. Criar Ambiente Conda
```bash
conda create --name churn python=3.11
source activate churn
```

### 2. Instalar Dependências
```bash
pip install -r requirements.txt
```

### 3. Gerar Base de Dados Sintética (Opcional)
```bash
python generate_dataset.py
```

### 4. Configurar API Keys
Crie um arquivo `.streamlit/secrets.toml` (se não existir) com:
```toml
OPENAI_API_KEY="sua_chave_openai"
GROQ_API_KEY="sua_chave_groq"
```

### 5. Rodar o App
```bash
streamlit run app.py
```

## ☁️ Deploy no Streamlit Cloud

1. **Repositório:** Suba o código para um repositório no GitHub.
2. **Secrets:** No painel do Streamlit Cloud, vá em **Settings > Secrets** e cole o conteúdo do seu `secrets.toml`.
3. **Deploy:** Aponte o Streamlit Cloud para o arquivo `app.py`.

## 🤖 Arquitetura Multiagente (LangGraph)

O projeto utiliza uma estrutura de grafo para orquestrar 4 agentes especializados:
1. **Auditor:** Valida a qualidade da transcrição.
2. **Resumidor:** Extrai contexto e sentimento.
3. **Classificador:** Categoriza o motivo macro (Problema Técnico, Concorrência, etc).
4. **Diagnóstico (D-90):** Analisa a jornada temporal completa e gera a hipótese de causa raiz.

## 🧬 Machine Learning
Utiliza as features da jornada (número de ligações, volume de texto, repetição de motivos) para treinar modelos preditivos (RandomForest, XGBoost) que calculam o score de risco por cliente.

---
**Radar X - Inteligência na Retenção de Clientes**
