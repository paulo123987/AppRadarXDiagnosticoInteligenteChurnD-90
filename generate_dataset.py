"""
Script para gerar dataset sintético de call center - Radar X Churn
Execute: python generate_dataset.py
Gera: dados_churn_sintetico.csv com 100 registros
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

random.seed(42)
np.random.seed(42)

# ─── Definição de clientes e perfis ─────────────────────────────────────────
CLIENTES = {
    "CLI001": {"nome": "Ana Paula Mendes",    "perfil": "tecnico",      "ligacoes": 6},
    "CLI002": {"nome": "Roberto Alves",        "perfil": "financeiro",   "ligacoes": 4},
    "CLI003": {"nome": "Carla Souza",          "perfil": "atendimento",  "ligacoes": 5},
    "CLI004": {"nome": "Marcos Lima",          "perfil": "concorrencia", "ligacoes": 3},
    "CLI005": {"nome": "Juliana Costa",        "perfil": "tecnico",      "ligacoes": 7},
    "CLI006": {"nome": "Fernando Rocha",       "perfil": "financeiro",   "ligacoes": 4},
    "CLI007": {"nome": "Patrícia Nunes",       "perfil": "mudanca",      "ligacoes": 3},
    "CLI008": {"nome": "Diego Martins",        "perfil": "atendimento",  "ligacoes": 5},
    "CLI009": {"nome": "Beatriz Freitas",      "perfil": "tecnico",      "ligacoes": 6},
    "CLI010": {"nome": "Ricardo Santos",       "perfil": "pessoal",      "ligacoes": 2},
    "CLI011": {"nome": "Vanessa Torres",       "perfil": "concorrencia", "ligacoes": 4},
    "CLI012": {"nome": "André Oliveira",       "perfil": "tecnico",      "ligacoes": 5},
    "CLI013": {"nome": "Mônica Barbosa",       "perfil": "financeiro",   "ligacoes": 3},
    "CLI014": {"nome": "Thiago Pereira",       "perfil": "atendimento",  "ligacoes": 4},
    "CLI015": {"nome": "Luciana Ferreira",     "perfil": "tecnico",      "ligacoes": 6},
    "CLI016": {"nome": "Eduardo Gomes",        "perfil": "mudanca",      "ligacoes": 3},
    "CLI017": {"nome": "Camila Ribeiro",       "perfil": "financeiro",   "ligacoes": 5},
    "CLI018": {"nome": "Bruno Carvalho",       "perfil": "concorrencia", "ligacoes": 3},
    "CLI019": {"nome": "Renata Azevedo",       "perfil": "atendimento",  "ligacoes": 4},
    "CLI020": {"nome": "Gabriel Silva",        "perfil": "tecnico",      "ligacoes": 5},
    "CLI021": {"nome": "Priscila Moura",       "perfil": "financeiro",   "ligacoes": 3},
    "CLI022": {"nome": "Felipe Castro",        "perfil": "tecnico",      "ligacoes": 4},
    "CLI023": {"nome": "Adriana Pinto",        "perfil": "atendimento",  "ligacoes": 3},
    "CLI024": {"nome": "Leandro Araújo",       "perfil": "concorrencia", "ligacoes": 2},
    "CLI025": {"nome": "Fernanda Correia",     "perfil": "mudanca",      "ligacoes": 3},
}

# ─── Templates de transcrições por perfil ───────────────────────────────────
TRANSCRICOES = {
    "tecnico": [
        "Olá, estou ligando porque minha internet está extremamente lenta faz mais de uma semana. "
        "Fiz vários testes e a velocidade está muito abaixo do contratado. Já reiniciei o roteador várias vezes. "
        "Ninguém conseguiu resolver meu problema até agora. Estou muito frustrado com o serviço.",

        "Boa tarde, eu já liguei antes sobre a lentidão da internet. Vocês mandaram um técnico que disse que estava tudo ok, "
        "mas o problema continua. Minha internet cai todo dia, principalmente à noite. Isso está afetando meu trabalho home office. "
        "Preciso de uma solução definitiva urgente.",

        "Estou ligando novamente sobre o problema técnico que nunca foi resolvido. Já é a terceira vez que ligo. "
        "Ficam me passando de setor em setor e ninguém resolve. A internet cai várias vezes ao dia, "
        "perco reuniões importantes. Estou seriamente pensando em cancelar o contrato.",

        "Olá, é a quarta vez que ligo sobre o mesmo problema. A internet ainda está instável, com quedas frequentes. "
        "Já pedi visita técnica duas vezes e o problema não foi resolvido. Paguei pelo serviço que não está funcionando. "
        "Isso é inaceitável. Quero um posicionamento definitivo senão vou cancelar.",

        "Boa tarde, liguei várias vezes e o problema persiste até hoje. Minha internet está completamente instável. "
        "Não consigo trabalhar, fazer videoconferências, nem usar streaming. Quero cancelar meu contrato imediatamente. "
        "Nunca mais vou indicar esta empresa para ninguém. Estou muito insatisfeito.",

        "Olá, quero registrar mais uma reclamação. A internet caiu novamente hoje e fica caindo toda semana. "
        "Vocês prometeram resolver mas nunca resolvem. Já estou procurando outras operadoras. Quero o contato do SAC.",
    ],
    "financeiro": [
        "Bom dia, estou ligando porque recebi uma fatura com valor muito acima do que contratei. "
        "Há cobranças que eu não reconheço. Já tentei resolver pelo aplicativo mas não consegui. "
        "Preciso que me expliquem essas cobranças indevidas.",

        "Boa tarde, já liguei no mês passado sobre a fatura errada e disseram que iam corrigir, "
        "mas essa mês vieram as mesmas cobranças indevidas novamente. Estou pagando por serviços que não contratei. "
        "Isso é um absurdo. Quero estorno imediato.",

        "Olá, quero falar sobre os reajustes absurdos na minha conta. A mensalidade aumentou mais de 30% "
        "sem me avisar corretamente. Outros operadores estão oferecendo o mesmo serviço por muito menos. "
        "Não tenho como continuar pagando esse valor. Preciso de uma proposta de redução.",

        "Estou muito insatisfeito com a cobrança indevida que continua aparecendo na minha fatura. "
        "Já reclamei três vezes e nada foi resolvido. Vou pagar só o valor correto e quero cancelar "
        "os serviços adicionais que nunca solicitei.",

        "Bom dia, quero cancelar meu plano. O preço está muito alto comparado com a concorrência "
        "e o serviço não justifica o custo. Já recebi propostas muito melhores de outras operadoras. "
        "Quero efetuar o cancelamento hoje.",
    ],
    "atendimento": [
        "Boa tarde, estou ligando para reclamar do atendimento que recebi na semana passada. "
        "O atendente foi extremamente grosseiro e não quis me ajudar. Fiquei mais de uma hora esperando "
        "e ao final nada foi resolvido. Isso é inadmissível.",

        "Olá, liguei no dia 15 e o atendente desligou o telefone na minha cara. "
        "Já faz duas semanas que ninguém entrou em contato para resolver minha reclamação. "
        "Sinto que a empresa não se importa com os clientes. Estou muito decepcionado.",

        "Boa tarde, estou há mais de um mês tentando resolver um problema simples e cada vez que ligo "
        "preciso explicar tudo do zero. Nenhum atendente registra o histórico. "
        "É extremamente desgastante. Preciso falar com um supervisor urgente.",

        "Estou perdendo a paciência. Já liguei cinco vezes e cada hora uma informação diferente. "
        "Ninguém sabe o que está acontecendo com minha solicitação. O atendimento é péssimo. "
        "Vou registrar reclamação no Procon se não resolverem hoje.",

        "Boa tarde, minha última experiência de atendimento foi horrível. Fui transferido quatro vezes, "
        "fiquei 2 horas no telefone e no final disseram que não podiam me ajudar. "
        "Quero cancelar tudo. Nunca fui tão mal atendido em minha vida.",
    ],
    "concorrencia": [
        "Oi, estou ligando porque recebi uma proposta muito interessante de outra operadora com o dobro da velocidade "
        "pelo mesmo preço que pago aqui. Queria saber se vocês conseguem fazer uma contraproposta antes de eu cancelar.",

        "Bom dia, a operadora X está me oferecendo fibra óptica com muito mais velocidade e desconto especial "
        "para os três primeiros meses. O serviço de vocês já está me decepcionando há tempos. "
        "Quero saber o que vocês oferecem para me manter como cliente.",

        "Boa tarde, já tomei a decisão. A concorrente instalou a fibra na minha rua e o preço é muito melhor. "
        "Meu plano com vocês não está atendendo mais minhas necessidades e o preço está alto demais. "
        "Quero cancelar meu contrato.",

        "Estou ligando para cancelar definitivamente. A outra operadora já fez a instalação e estou muito mais satisfeito. "
        "A velocidade é muito superior e o custo menor. Não tenho mais interesse em continuar com vocês.",
    ],
    "mudanca": [
        "Bom dia, vou me mudar para outro bairro e gostaria de saber se vocês atendem a região. "
        "Preciso transferir meu contrato ou instalar um novo ponto no novo endereço.",

        "Boa tarde, já entrei em contato antes sobre a mudança de endereço. Meu novo endereço não tem cobertura "
        "de vocês. Preciso cancelar meu contrato pois vou para uma região que vocês não atendem.",

        "Olá, preciso cancelar meu serviço. Me mudei para outro município e infelizmente a empresa não tem cobertura "
        "no meu novo endereço. Foi uma boa experiência mas não tenho como continuar.",
    ],
    "pessoal": [
        "Bom dia, estou passando por dificuldades financeiras e preciso cancelar alguns serviços. "
        "Gostaria de reduzir meu plano para algo mais básico e acessível no momento.",

        "Boa tarde, estou desempregado há dois meses e não consigo mais pagar o plano atual. "
        "Preciso cancelar ou encontrar uma opção mais barata urgentemente.",
    ],
}

# ─── Gerar registros ─────────────────────────────────────────────────────────
registros = []
data_churn_base = datetime(2025, 11, 30)

for cli_id, info in CLIENTES.items():
    perfil = info["perfil"]
    n_ligacoes = info["ligacoes"]
    templates = TRANSCRICOES[perfil]

    # Data de churn aleatória no mês de novembro de 2025
    dias_churn = random.randint(0, 29)
    data_churn = data_churn_base - timedelta(days=dias_churn)

    # Gerar datas das ligações dentro dos 90 dias anteriores ao churn
    datas_ligacoes = sorted([
        data_churn - timedelta(days=random.randint(1, 89))
        for _ in range(n_ligacoes)
    ])

    for i, data in enumerate(datas_ligacoes):
        transcricao = templates[i % len(templates)]
        
        # Adicionar variações na transcrição
        variacao = random.choice([
            " Já estou muito cansado de ligar.",
            " Quero uma solução definitiva.",
            " Isso está me causando muitos prejuízos.",
            " Preciso de retorno urgente.",
            " Vou avaliar outras opções.",
            "",
        ])
        transcricao_final = transcricao.strip() + variacao

        hora = random.randint(8, 19)
        minuto = random.randint(0, 59)
        segundo = random.randint(0, 59)
        dt_formatado = data.replace(hour=hora, minute=minuto, second=segundo)

        registros.append({
            "ID_CLIENTE": cli_id,
            "TRANSCRICAO_LIGACAO_CLIENTE": transcricao_final,
            "DATETIME_TRANSCRICAO_LIGACAO": dt_formatado.strftime("%Y-%m-%d %H:%M:%S"),
        })

# ─── Criar DataFrame e salvar ────────────────────────────────────────────────
df = pd.DataFrame(registros)
df = df.sort_values(["ID_CLIENTE", "DATETIME_TRANSCRICAO_LIGACAO"]).reset_index(drop=True)

output_path = "dados_churn_sintetico.csv"
df.to_csv(output_path, index=False, encoding="utf-8-sig")

print(f"✅ Dataset gerado com sucesso!")
print(f"   Total de registros: {len(df)}")
print(f"   Clientes únicos: {df['ID_CLIENTE'].nunique()}")
print(f"   Arquivo: {output_path}")
print(df.head(5).to_string())
