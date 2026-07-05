# MLOps Pipeline — Design Spec

**Projeto**: cloudwalk_fastapi_model (HTTP Anomaly Detection com Isolation Forest)
**Data**: 2026-07-04
**Objetivo**: Pipeline completo de MLOps com versionamento, monitoramento de drift e CI/CD na Azure — foco em demonstração para entrevista.

## Decisões

| Aspecto | Escolha | Motivo |
|---------|---------|--------|
| CI/CD | GitHub Actions | Já está no GitHub, zero setup extra |
| Versionamento | MLflow on Azure ML | Padrão de mercado, tracking + registry integrado |
| Drift monitoring | Azure ML Data Drift Monitor | Nativo, sem infra extra |
| Retrain | Automático via webhook | Drift → alerta → retrain → deploy |
| Deploy | Azure Container Apps com canário | Já em uso, suporta traffic splitting nativo |

## 1. Arquitetura

```
GitHub push/PR
  └─► GitHub Actions CI/CD
       ├─ lint + test
       ├─ train (Azure ML Compute)
       ├─ log metrics + register model (MLflow)
       ├─ quality gate (new > baseline?)
       └─ deploy canário (Azure Container Apps)

Azure ML Workspace
  ├─ MLflow Tracking Server
  ├─ Model Registry (versioned, staged)
  └─ Data Drift Monitor (daily)
       └─► Azure Monitor Alert → webhook → GitHub Actions retrain

Azure Container Apps
  ├─ Production (modelo v N, 90% tráfego)
  └─ Canary (modelo v N+1, 10% tráfego, validação)
```

Fluxo completo: Code push → CI/CD → Train → Validate → Register → Deploy canário → Monitor drift → Retrain automático.

## 2. Versionamento com MLflow + Azure ML

### MLflow Tracking
- Hospedado no Azure ML workspace (sem servidor próprio)
- Cada run loga: hiperparâmetros (`n_estimators`, `contamination`), métricas (anomaly score distribution, silhouette score), artifacts (`model.joblib`, `scaler.joblib`)
- Experimento: `http-anomaly-isolation-forest`

### Model Registry
- Estágios: `None` → `Staging` → `Production`
- Cada versão inclui: hash do dataset, métricas de validação, link pro commit Git
- Transição `Staging → Production` só após quality gate

### Dados de treino
- Dataset (`features.csv`) registrado como Azure ML Dataset com timestamp
- Referência ao dataset no metadata do run MLflow
- Sem DVC — Azure ML Dataset já versiona, CSV é pequeno (533 samples)

### Mudanças no código
- `train_model.py`: wrapper com `mlflow.start_run()`, loga params/metrics/artifacts
- `metadata.json`: gerado como artifact do MLflow (não mais commitado no repo)
- `model_loader.py`: opção de carregar modelo do registry por versão/estágio

## 3. CI/CD com GitHub Actions

### Workflow 1 — CI (push/PR)
- Lint (`ruff`) + type check (`mypy`)
- Testes unitários (feature engineering, schemas)
- Testes de integração (predict endpoint com modelo de teste)
- Validação do Dockerfile (build sem push)

### Workflow 2 — Train & Deploy (push em `main` ou manual)
1. Login Azure via OIDC (federated credentials)
2. Submit job de treino ao Azure ML Compute (`az ml job create`)
3. MLflow loga métricas, registra modelo como `Staging`
4. **Quality Gate**: compara novo vs `Production` atual
   - Silhouette score ≥ baseline
   - Anomaly rate em range esperado (3-7%)
   - Falha → pipeline para, modelo fica em `Staging`, notifica via PR comment
5. Gate ok → promove modelo para `Production`
6. Deploy canário no Container Apps:
   - Nova revision recebe 10% tráfego
   - Monitora 10 min (latência, error rate)
   - Ok → ramp up 100%
   - Degradação → rollback automático

### Workflow 3 — Retrain on Drift (webhook trigger)
- Mesmo fluxo do Workflow 2, disparado pelo alerta de drift
- Usa dataset atualizado com dados de produção

### Secrets no GitHub
- `AZURE_CLIENT_ID`, `AZURE_TENANT_ID`, `AZURE_SUBSCRIPTION_ID` (OIDC)
- `AZURE_ML_WORKSPACE`, `AZURE_RESOURCE_GROUP`

## 4. Monitoramento de Drift

### Baseline dataset
- 10 features do dataset de treino + timestamp, registradas no Azure ML

### Target dataset (produção)
- A cada `/predict`, app loga features calculadas + timestamp + prediction → Azure Blob (append blob, um arquivo por dia)
- Job agendado coleta logs e registra como `target` dataset

### Data Drift Monitor
- Schedule diário no Azure ML
- Métricas: PSI (threshold > 0.2) + Wasserstein distance
- Monitora todas 10 features, atenção em `request_count_30s`, `status_401_ratio`, `known_ua_ratio`

### Alertas e retrain automático
- Drift detectado (PSI > 0.2 em ≥ 2 features) → Azure Monitor Alert
- Action Group com webhook → GitHub Actions Workflow 3
- Fluxo: drift → alerta → retrain → quality gate → deploy canário → produção

### Concept drift
- Se houver labels de feedback, comparar taxa de acerto ao longo do tempo
- Para escopo de entrevista: documentar como seria, implementar data drift automático

### Mudanças no código
- Novo `monitoring/logger.py`: loga features para Azure Blob
- `/predict` chama logger async (sem impactar latência)
- Novo `monitoring/register_baseline.py`: registra baseline no Azure ML
- Novo `monitoring/setup_drift_monitor.py`: configura o monitor

## 5. Estrutura de Arquivos

```
cloudwalk_fastapi_model/
├── .github/workflows/
│   ├── ci.yml                    # Lint, test, build
│   ├── train-deploy.yml          # Train → gate → deploy canário
│   └── retrain-on-drift.yml      # Webhook → retrain
├── training/
│   ├── train_model.py            # Mod: wrapper MLflow
│   ├── prepare_features.py       # Existente
│   ├── evaluate.py               # Existente
│   └── azureml_job.yml           # Azure ML job spec
├── monitoring/
│   ├── logger.py                 # Features → Azure Blob
│   ├── register_baseline.py      # Baseline dataset
│   └── setup_drift_monitor.py    # Configura drift monitor
├── infra/
│   ├── setup_workspace.sh        # Azure ML workspace + compute
│   └── deploy_container_app.sh   # Deploy com canário
├── api/                          # Minor changes (add logger)
├── lib/                          # Sem mudanças
├── data/                         # Existente
├── artifacts/                    # Local dev only
└── ultra-low-latency/            # Existente
```

~10 arquivos novos. Nenhum existente deletado. Modificados: `train_model.py`, `api/main.py`.
