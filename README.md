# 🏦 Predição de Salários — Brasil

> **UC Aprendizado de Máquina 2026/1 — Projeto Final**
>
> ⚠️ **Este conteúdo é destinado apenas para fins educacionais. Os dados exibidos são ilustrativos e podem não corresponder a situações reais.**

---

## Estrutura do Projeto

```
salarios-ml/
├── data/
│   └── dataset_salarios_brasil.csv
├── models/                         # criado ao treinar
│   ├── best_model.joblib
│   ├── best_model_meta.json
│   └── *.joblib                    # um por modelo
├── results/                        # criado ao treinar
│   └── model_comparison.csv
├── mlruns/                         # criado pelo MLflow ao treinar
├── notebooks/
│   └── 02_resultados.py
├── src/
│   ├── pipeline/
│   │   ├── preprocessing.py        # limpeza + feature engineering
│   │   ├── features.py             # ColumnTransformer sklearn
│   │   ├── models.py               # catálogo de modelos
│   │   └── train.py                # script principal
│   └── api/
│       └── app.py                  # FastAPI
├── app_hf.py                       # Gradio para Hugging Face Spaces
├── Dockerfile
├── Makefile
└── requirements.txt
```

---

## Passo a Passo para Rodar

### 1. Clonar e instalar

```bash
git clone https://github.com/seu-usuario/salarios-ml.git
cd salarios-ml

pip install -r requirements.txt
```

### 2. Treinar os modelos

**Rode sempre a partir da raiz do projeto** (a pasta `salarios-ml/`):

```bash
python src/pipeline/train.py
```

O script vai:
- Carregar `data/dataset_salarios_brasil.csv`
- Aplicar todos os filtros de limpeza (detalhes abaixo)
- Treinar 8 modelos com CV-5
- Criar a pasta `mlruns/` com os experimentos no formato correto do MLflow
- Salvar o melhor modelo em `models/best_model.joblib`
- Salvar `results/model_comparison.csv`

### 3. Visualizar os experimentos no MLflow

```bash
python mlflow_ui.py
```

Acesse **http://localhost:5000** no browser.

> Use sempre `python mlflow_ui.py` em vez de `mlflow ui` diretamente.
> O script resolve o caminho absoluto para `mlruns/` automaticamente,
> evitando o erro `meta.yaml does not exist` que ocorre quando o path relativo
> não bate entre onde o `train.py` gravou e onde o `mlflow ui` procura.

### 4. Subir a API

```bash
uvicorn src.api.app:app --reload
```

Acesse:
- **http://localhost:8000** — página inicial com disclaimer
- **http://localhost:8000/docs** — Swagger com todos os endpoints
- **http://localhost:8000/exemplos** — exemplos de payload prontos

> A API precisa que o treino já tenha sido executado (`models/best_model.joblib` deve existir).

### 5. Testar a API com curl

```bash
curl -X POST http://localhost:8000/predizer \
  -H "Content-Type: application/json" \
  -d '{
    "idade": 35,
    "anos_experiencia": 10,
    "escolaridade": "Superior",
    "regiao": "Sudeste",
    "profissao": "Engenheiro",
    "segunda_lingua": "Inglês",
    "terceira_lingua": "Nenhuma"
  }'
```

---

## Limpeza dos Dados

O dataset original tem 15.000 registros. Após todos os filtros restam **8.052 (53,7%)**.

| Filtro | Removidos | Justificativa |
|--------|-----------|---------------|
| Salário nulo ou `"erro"` | 497 | Target não pode ser imputado |
| `Idade − Anos_Experiencia < 14` | 4.746 | Dado impossível (ex: 25 anos, 26 de exp.) |
| Profissão exige superior + escolaridade insuficiente | 1.664 | Inconsistência de domínio |
| Outliers de salário > p99.5 (R$ 227k) | 41 | Reduz distorção na função de custo |
| **Total removido** | **6.948** | |
| **Restante** | **8.052** | |

Nulos em `Idade` (472) e `Anos_Experiencia` (481) são **imputados com mediana** antes dos filtros de consistência, pois os filtros dependem dessas colunas.

---

## Resultados dos Modelos

| Modelo | RMSE (R$) | MAE (R$) | R² | MAPE |
|---|---|---|---|---|
| **Ridge** ⭐ | **10.075** | **1.828** | **0.249** | **12,1%** |
| GradientBoosting | 10.173 | 2.015 | 0.234 | 14,4% |
| RandomForest | 10.202 | 2.137 | 0.230 | 15,5% |
| ElasticNet | 10.215 | 2.289 | 0.228 | 16,7% |
| Lasso | 10.375 | 2.760 | 0.203 | 21,3% |
| KNN (k=10) | 10.666 | 3.447 | 0.158 | 31,0% |
| Dummy (baseline) | 11.908 | 5.409 | -0.050 | 55,3% |
| DecisionTree | 12.334 | 3.623 | -0.126 | 33,5% |

O **Ridge** venceu os ensembles. A relação salário × features é essencialmente linear no espaço log, e com ~6.4k registros de treino os modelos complexos não têm dados suficientes para explorar interações não-lineares. O R² de ~0.25 é esperado — salário depende fortemente de variáveis não observadas (empresa, negociação individual, benefícios).

---

## Deploy

### Hugging Face Spaces (gratuito, recomendado)

1. Crie um Space em [huggingface.co](https://huggingface.co) com SDK **Gradio**
2. Faça upload dos arquivos:
   ```
   app_hf.py              → renomeie para app.py
   requirements.txt       → adicione "gradio" ao final
   models/best_model.joblib
   models/best_model_meta.json
   src/pipeline/preprocessing.py
   src/pipeline/features.py
   src/__init__.py
   src/pipeline/__init__.py
   ```
3. O Space builda e publica automaticamente

### Docker (Render / Railway)

```bash
# Build e rodar local
docker build -t salarios-ml .
docker run -p 8000:8000 salarios-ml

# Deploy no Render:
# 1. Suba o código para o GitHub (incluindo a pasta models/)
# 2. Novo Web Service → selecione o repositório → Runtime: Docker
# 3. Render faz o deploy automático a cada push
```

---

## Metodologia CRISP-DM

```
1. Entendimento do Negócio  →  Prever salário para suporte a decisões de carreira/RH
2. Entendimento dos Dados   →  EDA (notebook 01_eda.ipynb fornecido)
3. Preparação               →  preprocessing.py + features.py
4. Modelagem                →  train.py (8 modelos, CV-5, MLflow)
5. Avaliação                →  results/model_comparison.csv + notebooks/02_resultados.py
6. Deploy                   →  FastAPI (local/Docker) + Gradio (Hugging Face)
```

---

## Decisões Técnicas

| Decisão | Justificativa |
|---|---|
| `log1p(Salario)` como target | Distribuição assimétrica → log aproxima normal |
| Imputar nulos com mediana **antes** dos filtros | Filtros de consistência dependem de Idade e Exp. |
| Remover `dif_idade_exp < 14` | Dado fisicamente impossível |
| Remover prof. + esc. inconsistentes | Reduz ruído de dados inválidos |
| Outliers > p99.5 removidos | 41 registros com salário > R$227k distorcem o custo |
| OrdinalEncoder na Escolaridade | Hierarquia clara: Fundamental < ... < Doutorado |
| OneHotEncoder em Profissão/Região | Sem ordem natural |
| Estratificar split por Profissão | Garante todas as 19 profissões em treino e teste |
