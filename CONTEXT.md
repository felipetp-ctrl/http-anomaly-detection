# HTTP Anomaly Detection — Contexto para Implementação

## Objetivo do Projeto

Projeto prático de preparação para entrevista na CloudWalk (vaga: ML Engineer — Security).
Demonstrar capacidade de aplicar anomaly detection ao domínio de segurança HTTP.

O entregável final é um repositório GitHub com:
- Modelo Isolation Forest treinado sobre features extraídas de logs HTTP
- API FastAPI servindo o modelo com endpoint `/predict`
- Código limpo, documentado, e com decisões de design justificáveis em entrevista

---

## Decisões de Design (já validadas)

Todas as decisões abaixo foram tomadas deliberadamente durante a fase de design. Implementar conforme descrito — não simplificar nem alterar sem justificativa.

### 1. Unidade de Observação

- A unidade de observação é o **comportamento agregado de um IP em uma janela de tempo**, nunca um request individual.
- Um request isolado (ex: `POST /login` com status 401) é indistinguível de um erro legítimo. O contexto temporal é o que revela o padrão de ataque.

### 2. Janelas Temporais

- Duas janelas paralelas: **30 segundos** e **5 minutos**.
- Ambas geram features que entram no **mesmo vetor**, consumido por um **único modelo**.
- A janela de 30s captura rajadas agressivas (DDoS, credential stuffing rápido). A janela de 5min captura ataques lentos e distribuídos (bots sofisticados).
- A relação entre valores das duas janelas (ex: 200 requests em 30s e 200 em 5min vs 10 em 30s e 500 em 5min) carrega informação que uma janela única não captura.

### 3. Feature Engineering — 9 Features

Cada feature tem justificativa vinculada a um tipo de ataque:

| # | Feature | Tipo | Justificativa |
|---|---------|------|---------------|
| 1 | `request_count_30s` | Contagem | Volume bruto — sinal primário de DDoS e credential stuffing |
| 2 | `request_count_5min` | Contagem | Volume em janela longa — captura ataques distribuídos |
| 3 | `endpoint_entropy` | Entropia (categórica) | Concentração em endpoints (baixa entropia = credential stuffing em `/login`) |
| 4 | `status_code_entropy` | Entropia (categórica) | Distribuição de status codes (baixa entropia + muitos 401 = credential stuffing) |
| 5 | `status_401_ratio` | Proporção | Taxa de falhas de autenticação — sinal direto de credential stuffing |
| 6 | `interval_std` | Desvio padrão (numérico) | Regularidade dos intervalos entre requests (std ≈ 0 = automação/bot) |
| 7 | `unique_ua_ratio` | Razão | User agents únicos / total de requests — rotação artificial indica bot sofisticado |
| 8 | `known_ua_ratio` | Proporção | Fração de requests com user agents de navegadores reais — bots simples usam `python-requests`, `curl`, etc. |
| 9 | `payload_size_std` | Desvio padrão (numérico) | Uniformidade do payload — credential stuffing envia payloads de tamanho constante |
| 10 | `response_time_std` | Desvio padrão (numérico) | Uniformidade do response time — automação gera padrões uniformes |

**Nota:** São 10 features na tabela, mas a decisão original foi 9. O `response_time_std` foi adicionado na discussão final. Manter todas as 10.

### 4. Estado em Memória — Estrutura de Dados

- Dicionário Python: `Dict[str, deque[RequestRecord]]` — chave é o IP, valor é um `deque` de registros.
- Cada registro é uma **`namedtuple`** (não `dict`, não `dataclass`, não JSON string). Motivo: footprint de memória equivalente a uma tupla, com acesso por nome de campo. Sob ataque, um IP pode ter milhares de registros — o overhead de `dict` por registro seria significativo.

```python
from collections import namedtuple
RequestRecord = namedtuple('RequestRecord', [
    'timestamp', 'endpoint', 'status_code', 
    'payload_size', 'user_agent', 'response_time'
])
```

- **Limpeza por janela de tempo**, não por `maxlen`. Motivo: `maxlen` descarta registros antigos mas trunca a contagem real (um IP com 50.000 requests em 5min apareceria como tendo apenas `maxlen` requests, subestimando a anomalia). A limpeza temporal remove tudo anterior à janela máxima (5 minutos), preservando a contagem verdadeira.

### 5. Arquitetura do Projeto

Quatro macropastas com responsabilidades separadas:

```
http-anomaly-detection/
├── lib/                      # Biblioteca compartilhada
│   ├── feature_engineering.py  # Cálculo das 10 features — ÚNICA fonte de verdade
│   ├── model_loader.py         # Carregamento de artefatos (modelo, scaler, metadados)
│   └── known_user_agents.py    # Lista de UAs reconhecidos como navegadores reais
│
├── training/                 # Pipeline de treino (batch, offline)
│   ├── prepare_features.py     # Lê logs brutos → agrega por IP/janela → gera dataset de features
│   ├── train_model.py          # Treina Isolation Forest, salva artefatos
│   └── evaluate.py             # Avalia o modelo contra labels do dataset sintético
│
├── api/                      # Serviço de inferência (online, tempo real)
│   ├── main.py                 # FastAPI app, lifespan, endpoint /predict
│   ├── state.py                # Gerenciamento do dicionário de deques em memória
│   └── schemas.py              # Pydantic models para request/response
│
├── artifacts/                # Artefatos produzidos pelo treino
│   ├── model.joblib            # Isolation Forest serializado
│   ├── scaler.joblib           # StandardScaler ajustado nos dados de treino
│   └── metadata.json           # Lista ordenada de features, hiperparâmetros, data de treino
│
├── data/                     # Dados
│   ├── generate_dataset.py     # Gerador de logs sintéticos (já implementado)
│   └── http_logs.csv           # Dataset gerado (já disponível)
│
├── README.md
└── requirements.txt
```

**Ponto crítico:** `lib/feature_engineering.py` é a ÚNICA implementação da lógica de features. O pipeline de treino (`training/prepare_features.py`) e a API (`api/main.py`) ambos importam deste módulo. Isso garante que o treino e a inferência calculam as features de forma idêntica. Se uma feature for adicionada ou modificada, a mudança ocorre em um único lugar.

### 6. Artefatos — Contrato entre Treino e Inferência

O diretório `artifacts/` contém três arquivos produzidos juntos pelo pipeline de treino:

1. **`model.joblib`** — Isolation Forest serializado
2. **`scaler.joblib`** — StandardScaler ajustado nos dados de treino (mesmas médias e desvios padrão devem ser usados na inferência)
3. **`metadata.json`** — Contrato explícito:
   ```json
   {
     "feature_names": ["request_count_30s", "request_count_5min", ...],
     "feature_order": [0, 1, 2, ...],
     "n_estimators": 100,
     "contamination": 0.05,
     "trained_at": "2026-07-03T...",
     "training_samples": 533
   }
   ```
   Na inicialização, a API carrega os metadados e **valida** que as features calculadas correspondem ao que o modelo espera, na mesma ordem. Se não corresponderem, o servidor falha explicitamente em vez de retornar scores sem sentido.

### 7. Decisões de Latência

- **Carregamento do modelo:** uma única vez na inicialização do servidor (via `lifespan` do FastAPI), mantido em memória. Nunca carregar do disco por request.
- **Endpoint síncrono:** `def predict(...)` sem `async`. O cálculo de features e a inferência são CPU-bound — não fazem I/O, não liberam o event loop. Definir como `def` faz o FastAPI executar a função em um thread pool automaticamente, sem bloquear o event loop para outras requisições.
- **Namedtuples:** menor overhead de memória por registro comparado a dicts.
- **Instrumentação obrigatória:** usar `time.perf_counter()` para medir o tempo de cada etapa dentro do endpoint (parsing, feature calculation, scaling, prediction, total). Logar esses tempos para identificar gargalos reais — não assumir onde está o gargalo.

### 8. Dataset

- **Logs brutos sintéticos** (não vetores de features pré-calculados). Motivo: testar o pipeline completo — ingestão de logs, agregação por IP/janela, cálculo de features, treino.
- O gerador (`data/generate_dataset.py`) simula quatro perfis:
  - **Tráfego legítimo (~93% dos IPs):** intervalos irregulares (distribuição exponencial), endpoints diversos, status codes variados, 1-2 user agents consistentes, payload e response time variáveis.
  - **Credential stuffing (~3% dos IPs):** rajadas em `/login` ou `/auth`, status 401 dominante, intervalos ultra-regulares (~5% jitter), user agent de bot, payload uniforme.
  - **L7 DDoS (~2% dos IPs):** volume extremo, endpoints pesados, intervalos regulares (~3% jitter), user agent real (tenta parecer legítimo), mix de 200/503/504.
  - **Bots maliciosos (~1.5% dos IPs):** muitos endpoints distintos (crawling sistemático), rotação de user agents, intervalos semi-regulares, acessa paths suspeitos (`/.env`, `/.git/config`, `/admin`).
- Campo `label` no CSV existe para **validação**, não para treinamento (Isolation Forest é não supervisionado).

### 9. O que NÃO Fazer

- **Não vazar dados:** separação treino/teste deve respeitar ordem temporal (teste posterior ao treino).
- **Não ignorar o baseline:** ter claro onde o modelo supera heurísticas simples (ex: "bloquear IPs com >100 req/min"). Se uma regra simples resolve, o ML não se justifica.
- **Não over-engineer:** sem Kubernetes, Kafka, microsserviços. O foco é: modelo funcional, API responsiva, código limpo, decisões justificáveis.
- **Não negligenciar explicabilidade:** se o modelo flaga um IP, deve ser possível dizer POR QUÊ ("500 requests ao /login em 30s com mesmo user_agent"), não apenas "score = -0.87".

---

## Stack Técnica

- **Python 3.11+**
- **scikit-learn:** Isolation Forest, StandardScaler
- **FastAPI + Uvicorn:** API de inferência
- **joblib:** serialização de artefatos
- **collections:** namedtuple, deque
- **pandas:** manipulação de dados no pipeline de treino (opcional, pode ser stdlib)
- **math:** cálculo de entropia

---

## Ordem de Implementação Sugerida

1. `lib/feature_engineering.py` — implementar o cálculo das 10 features primeiro (é a base de tudo)
2. `training/prepare_features.py` — consumir `http_logs.csv`, agregar por IP/janela, gerar dataset de features
3. `training/train_model.py` — treinar Isolation Forest, salvar os 3 artefatos
4. `training/evaluate.py` — avaliar contra labels, comparar com baseline de regras simples
5. `api/` — implementar o serviço FastAPI com estado em memória
6. Instrumentação de latência
7. `README.md` com documentação das decisões

---

## Contexto de Entrevista

Este projeto será apresentado na entrevista como demonstração de:
- Capacidade de aprender um domínio novo (segurança HTTP) e aplicar conhecimento existente (anomaly detection)
- Decisões de design justificadas tecnicamente, não escolhas arbitrárias
- Pipeline completo: dados → features → treino → deploy → monitoramento de latência
- Uso de IA (Claude) como ferramenta de desenvolvimento conjunto — a conversa de design precedeu a implementação
