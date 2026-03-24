# Relatório Técnico — Predição de Salários no Brasil

> **UC Aprendizado de Máquina 2026/1 — Projeto Final**
>
> ⚠️ Este conteúdo é destinado apenas para fins educacionais. Os dados exibidos são ilustrativos e podem não corresponder a situações reais.

---

## 1. Visão Geral do Projeto

O projeto tem como objetivo construir um sistema de **predição de salário** para profissionais brasileiros com base em características do perfil (profissão, escolaridade, região, idade, anos de experiência e idiomas). A aplicação segue a metodologia **CRISP-DM** e cobre todas as fases do ciclo de vida de um projeto de aprendizado de máquina: da coleta e limpeza dos dados até o deploy de uma API de predição.

O produto final é composto por:

- Uma **pipeline de ML** treinada e avaliada com rastreamento via MLflow;
- Uma **API REST** (FastAPI) para inferência em tempo real;
- Uma **interface interativa** (Gradio) publicável no Hugging Face Spaces.

---

## 2. Tecnologias, Bibliotecas e Ferramentas

| Categoria | Tecnologia / Biblioteca | Versão mínima |
|---|---|---|
| Linguagem | Python | 3.11 |
| Manipulação de dados | pandas | 2.0.0 |
| Computação numérica | NumPy | 1.24.0 |
| Machine Learning | scikit-learn | 1.3.0 |
| Boosting | XGBoost | 2.0.0 |
| Boosting | LightGBM | 4.0.0 |
| Rastreamento de experimentos | MLflow | 2.10.0 |
| Persistência de modelos | joblib | 1.3.0 |
| API REST | FastAPI + Uvicorn | 0.110.0 / 0.29.0 |
| Validação de esquema | Pydantic | 2.6.0 |
| Interface web | Gradio | — |
| Containerização | Docker | — |
| Gerenciamento de tarefas | Makefile | — |

---

## 3. Arquitetura e Abordagem

### 3.1 Organização de Diretórios

```
ml-salary-analysis/
├── data/
│   └── dataset_salarios_brasil.csv     # dataset bruto
├── models/                             # artefatos treinados
│   ├── best_model.joblib
│   ├── best_model_meta.json
│   └── <modelo>.joblib                 # um por modelo treinado
├── results/
│   └── model_comparison.csv            # comparação de todos os modelos
├── notebooks/
│   └── 02_resultados.py                # análise visual dos resultados
├── src/
│   ├── pipeline/
│   │   ├── preprocessing.py            # limpeza e feature engineering
│   │   ├── features.py                 # ColumnTransformer (encoding)
│   │   ├── models.py                   # catálogo de modelos candidatos
│   │   └── train.py                    # script principal de treino
│   └── api/
│       └── app.py                      # API REST (FastAPI)
├── app_hf.py                           # interface Gradio (Hugging Face Spaces)
├── mlflow_ui.py                        # helper para subir o MLflow UI
├── Dockerfile
├── Makefile
└── requirements.txt
```

### 3.2 Abordagem de Modelagem

A abordagem adota **comparação sistemática de múltiplos algoritmos** de diferentes famílias (lineares, árvores, ensembles e vizinhança). Todos os modelos são avaliados com **validação cruzada de 5 folds (CV-5)** e testados em um **holdout de 20%** reservado desde o início. Cada execução é rastreada no **MLflow**, garantindo reprodutibilidade e auditabilidade.

O target utilizado é `log1p(Salario)` em vez do salário bruto, decisão motivada pela distribuição assimétrica (cauda longa) da variável. As métricas finais são calculadas na **escala original em R$** após reversão com `expm1`, garantindo interpretabilidade.

A pipeline final de predição é um objeto `sklearn.Pipeline` com dois estágios:

```
Pipeline([
    ('pre',   ColumnTransformer),   # encoding + escalonamento
    ('model', <estimador>),         # algoritmo de regressão
])
```

Esse design garante que os mesmos parâmetros de transformação ajustados no treino sejam aplicados de forma consistente em produção.

---

## 4. Pipeline de ML — Etapas Detalhadas

### 4.1 Coleta de Dados

O dataset utilizado é `data/dataset_salarios_brasil.csv`, um conjunto sintético com **15.000 registros** e as seguintes colunas originais:

| Coluna | Tipo | Descrição |
|---|---|---|
| `Salario` | numérico | Salário bruto mensal em R$ (target) |
| `Idade` | numérico | Idade em anos |
| `Anos_Experiencia` | numérico | Anos de experiência profissional |
| `Escolaridade` | categórico | Nível de escolaridade |
| `Regiao` | categórico | Região do Brasil |
| `Profissao` | categórico | Profissão exercida |
| `Segunda_Lingua` | categórico | Segunda língua falada (ou "Nenhuma") |
| `Terceira_Lingua` | categórico | Terceira língua falada (ou "Nenhuma") |

### 4.2 Pré-Processamento e Limpeza (`preprocessing.py`)

A limpeza é realizada na função `limpar_dados()` com uma sequência ordenada de filtros:

| Etapa | Ação | Registros removidos | Justificativa |
|---|---|---|---|
| 1 | Remover `Salario` nulo ou `"erro"` | 497 | Target não pode ser imputado |
| 2 | Remover `Salario <= 0` | — | Valor fisicamente impossível |
| 3 | Imputar nulos em `Idade` (472) e `Anos_Experiencia` (481) com mediana | — | Necessário antes dos filtros de consistência que dependem dessas colunas |
| 4 | Remover `Idade − Anos_Experiencia < 14` | 4.746 | Dado impossível (ex.: 25 anos com 26 de experiência) |
| 5 | Remover profissão que exige nível superior + escolaridade insuficiente | 1.664 | Inconsistência de domínio |
| 6 | Remover outliers de salário acima do percentil 99,5 (> R$ 227k) | 41 | Reduz distorção na função de custo |

**Dataset final:** **8.052 registros** (53,7% do original).

As profissões que exigem nível superior são: Advogado, Agrônomo, Arquiteto, Cientista, Economista, Engenheiro, Médico e Pesquisador.

### 4.3 Feature Engineering (`preprocessing.py`)

Após a limpeza, são criadas três features derivadas na função `engenharia_features()`:

| Feature nova | Cálculo | Motivação |
|---|---|---|
| `tem_segunda_lingua` | `1` se `Segunda_Lingua != "Nenhuma"`, senão `0` | Flag binária para posse de idioma adicional |
| `tem_terceira_lingua` | `1` se `Terceira_Lingua != "Nenhuma"`, senão `0` | Flag binária para posse de segundo idioma adicional |
| `total_linguas` | `tem_segunda_lingua + tem_terceira_lingua` | Indicador ordinal de multilinguismo (0, 1 ou 2) |
| `dif_idade_exp` | `Idade − Anos_Experiencia` | Proxy do tempo de formação; já calculada na limpeza |

O target é transformado via `np.log1p(Salario)`.

### 4.4 Encoding e Escalonamento (`features.py`)

O `ColumnTransformer` aplica transformações distintas conforme o tipo semântico da variável:

| Grupo | Colunas | Transformação | Justificativa |
|---|---|---|---|
| Numéricas | `Idade`, `Anos_Experiencia`, `dif_idade_exp`, `total_linguas` | `StandardScaler` | Necessário para modelos sensíveis à escala (Ridge, KNN); não prejudica tree-based |
| Ordinais | `Escolaridade` | `OrdinalEncoder` com ordem explícita | Hierarquia clara: Fundamental < Médio < Técnico < Superior < Pós < Mestrado < Doutorado |
| Nominais | `Profissao`, `Regiao`, `Segunda_Lingua`, `Terceira_Lingua` | `OneHotEncoder` | Sem ordem natural entre categorias |
| Binárias | `tem_segunda_lingua`, `tem_terceira_lingua` | `passthrough` | Já estão em formato 0/1 |

Colunas não listadas (como `Salario` e `dif_idade_exp` bruto) são descartadas com `remainder="drop"`.

### 4.5 Divisão dos Dados

O split é realizado em proporção **80/20** com `stratify=Profissao`, garantindo que todas as 19 profissões do domínio estejam representadas tanto no conjunto de treino quanto no de teste:

- **Treino:** 6.441 registros
- **Teste (holdout):** 1.611 registros

`random_state=42` é fixado em todo o projeto para reprodutibilidade.

### 4.6 Treinamento (`train.py`)

Para cada modelo candidato:

1. Constrói um `Pipeline` com o pré-processador e o estimador;
2. Executa **CV-5** (`KFold`, `shuffle=True`) calculando RMSE na escala log (para comparação homogênea entre modelos);
3. Treina o pipeline completo no conjunto de treino;
4. Calcula métricas em R$ no holdout (revertendo `log1p`);
5. Loga parâmetros, métricas e o artefato `.joblib` no **MLflow**;
6. Salva o modelo em `models/<nome>.joblib`.

Ao final, o melhor modelo (menor RMSE no teste) é copiado para `models/best_model.joblib` e seus metadados são salvos em `models/best_model_meta.json`.

### 4.7 Rastreamento de Experimentos (MLflow)

Cada run no MLflow registra:

- **Tags:** nome do modelo, target, dataset, filtros aplicados
- **Parâmetros:** hiperparâmetros do estimador (apenas tipos primitivos)
- **Métricas:** RMSE teste/treino, MAE, R², MAPE, RMSE CV (log), tempo de treino
- **Artefato:** arquivo `.joblib` do pipeline completo

O servidor MLflow é iniciado via `python mlflow_ui.py` com caminho absoluto para `mlruns/`, evitando erros de path relativo.

### 4.8 Avaliação e Análise (`notebooks/02_resultados.py`)

O notebook de resultados gera:

- **Gráficos comparativos** de RMSE, MAE, R² e MAPE para todos os modelos;
- **Análise de overfitting** (RMSE treino vs. teste lado a lado);
- **Scatter real vs. predito**, distribuição de resíduos e homocedasticidade;
- **MAE por profissão e região** para análise de subgrupos.

---

## 5. Dados Após o Processamento

### 5.1 Resumo do Dataset Final

| Atributo | Valor |
|---|---|
| Registros totais | 8.052 |
| Proporção do original (15.000) | 53,7% |
| Conjunto de treino | 6.441 (80%) |
| Conjunto de teste | 1.611 (20%) |
| Número de features brutas | 11 colunas |
| Número de features após encoding | ~45 (numéricas + ordinais + OHE + binárias) |

### 5.2 Distribuição do Target

- O salário bruto apresenta distribuição assimétrica com cauda direita longa;
- Após `log1p`, a distribuição se aproxima de uma Gaussiana, melhorando o comportamento dos modelos lineares e a função de custo dos ensembles;
- Corte de outliers acima do percentil 99,5 (R$ 227k) eliminou 41 registros extremos.

### 5.3 Categorias Presentes

- **Profissões (19):** Administrador, Advogado, Agrônomo, Analista de Dados, Arquiteto, Cientista, Consultor, Designer, Economista, Enfermeiro, Engenheiro, Motorista, Médico, Pesquisador, Policial, Professor, Programador, Técnico, Vendedor
- **Regiões (5):** Norte, Nordeste, Centro-Oeste, Sudeste, Sul
- **Escolaridade (7 níveis):** Fundamental, Médio, Técnico, Superior, Pós, Mestrado, Doutorado
- **Idiomas disponíveis (9):** Nenhuma, Inglês, Espanhol, Francês, Alemão, Italiano, Mandarim, Japonês, Português

---

## 6. Decisões Técnicas e Justificativas

| Decisão | Justificativa |
|---|---|
| `log1p(Salario)` como target | A distribuição salarial é assimétrica (cauda direita). A transformação logarítmica aproxima o target de uma Gaussiana, reduzindo o impacto de valores extremos na função de custo e melhorando a convergência de modelos lineares. |
| Imputar nulos com mediana **antes** dos filtros de consistência | Os filtros de consistência (etapas 4 e 5) dependem de `Idade` e `Anos_Experiencia`. Se a imputação fosse posterior, esses registros seriam removidos desnecessariamente. A mediana é preferida à média por ser robusta a outliers. |
| Remover `dif_idade_exp < 14` | Ninguém começa a trabalhar antes dos ~14 anos. Uma diferença negativa ou menor que 14 indica dado inconsistente ou erro de coleta, e não deve ser imputado pois o "true value" é desconhecido. |
| Remover profissão + escolaridade inconsistentes | Advogado, Médico, Engenheiro, etc., exigem formação superior por lei/regulamentação profissional. Manter esses registros introduziria ruído sistemático no modelo. |
| Remover outliers acima do percentil 99,5 | 41 registros com salário superior a R$ 227k distorcem a função de custo e reduzem a capacidade do modelo de generalizar para o intervalo principal. O corte em p99,5 (em vez de p99 ou p95) é conservador, preservando o máximo de dados. |
| `OrdinalEncoder` em `Escolaridade` | Existe hierarquia clara e consensual entre os níveis: Fundamental < Médio < Técnico < Superior < Pós < Mestrado < Doutorado. Usar OHE descartaria essa informação ordinal. |
| `OneHotEncoder` em `Profissao`, `Regiao` e idiomas | Não existe ordenação natural entre profissões ou regiões. OHE evita que o modelo interprete uma ordem numérica arbitrária. |
| `StandardScaler` em todas as numéricas | Necessário para Ridge e KNN (sensíveis à escala). Aplicar uniformemente simplifica o pipeline e não prejudica modelos de árvore. |
| Split estratificado por `Profissao` | Garante que todas as 19 profissões apareçam em treino e teste, evitando que o modelo avalie em categorias nunca vistas. |
| `sklearn.Pipeline` encapsulando pré-processador + modelo | Previne data leakage (pré-processamento é ajustado apenas no treino), facilita serialização com `joblib` e torna o deploy trivial — uma única chamada `pipe.predict(df)`. |
| Múltiplas famílias de algoritmos | Cobertura ampla do espaço de hipóteses: lineares (Ridge, Lasso, ElasticNet), vizinhança (KNN), árvore simples (DecisionTree), ensemble bagging (RandomForest) e ensemble boosting (GradientBoosting, XGBoost, LightGBM). |
| MLflow para rastreamento | Garante reprodutibilidade, auditoria de experimentos e comparação objetiva entre runs. O URI é definido como caminho absoluto para evitar erros de path relativo. |

---

## 7. Resultados e Métricas de Desempenho

### 7.1 Comparação de Modelos (ordenados por RMSE no teste)

| Posição | Modelo | RMSE (R$) | MAE (R$) | R² | MAPE (%) | RMSE Treino (R$) | R² Treino |
|---|---|---|---|---|---|---|---|
| 1 ⭐ | **XGBoost** | **10.100** | **1.979** | **0,2449** | **14,1** | 10.836 | 0,406 |
| 2 | LightGBM | 10.118 | 2.017 | 0,2422 | 14,5 | 11.620 | 0,317 |
| 3 | GradientBoosting | 10.173 | 2.015 | 0,2341 | 14,4 | 11.184 | 0,367 |
| 4 | RandomForest | 10.202 | 2.137 | 0,2297 | 15,5 | 12.447 | 0,216 |
| 5 | ElasticNet | 10.215 | 2.289 | 0,2277 | 16,7 | 13.071 | 0,136 |
| 6 | Lasso | 10.375 | 2.760 | 0,2033 | 21,3 | 13.167 | 0,123 |
| 7 | KNN (k=10) | 10.665 | 3.445 | 0,1581 | 31,0 | 37 | **0,9999** |
| 8 | Dummy (mediana) | 11.908 | 5.409 | -0,0496 | 55,3 | 14.368 | -0,044 |
| 9 | DecisionTree | 12.334 | 3.623 | -0,1260 | 33,5 | 11.848 | 0,290 |

> **Melhor modelo:** XGBoost (salvo em `models/best_model.joblib`)

### 7.2 Análise do Melhor Modelo (XGBoost)

| Métrica | Valor |
|---|---|
| RMSE (teste) | R$ 10.100,26 |
| MAE (teste) | R$ 1.978,74 |
| R² (teste) | 0,2449 |
| MAPE (teste) | 14,1% |

**Interpretação:**

- **R² ≈ 0,25:** O modelo explica aproximadamente 25% da variância salarial observada. Esse valor é esperado para dados de salário: fatores não observados no dataset — como empresa específica, negociação individual, benefícios, cidade dentro da região e senioridade informal — são determinantes relevantes do salário e não estão capturados nas features disponíveis.

- **MAE ≈ R$ 1.979:** Em média, a predição erra cerca de R$ 2.000 para mais ou para menos. Para um salário mediano em torno de R$ 3.000–R$ 5.000, esse erro representa aproximadamente 14% (MAPE), o que é aceitável para uma estimativa orientativa baseada apenas em perfil público.

- **RMSE (treino) > RMSE (teste):** O gap treino/teste do XGBoost é pequeno, indicando boa generalização e ausência de overfitting severo. Contraste com o KNN, que atinge R²=0,9999 no treino mas apenas 0,16 no teste — overfitting extremo.

- **Boosting vs. lineares:** Os modelos de boosting (XGBoost, LightGBM, GradientBoosting) superaram os lineares (Lasso, ElasticNet) em todas as métricas, sugerindo que há interações não-lineares captáveis nas features — por exemplo, o efeito da escolaridade sobre o salário varia conforme a profissão. O dataset, com ~6.400 registros de treino, é suficiente para que os ensembles explorem essas interações com regularização adequada.

### 7.3 Baseline

O modelo Dummy (predição constante = mediana) apresenta RMSE de R$ 11.908 e R² = -0,05. O XGBoost supera esse baseline em **~16% de RMSE** e ~0,29 pontos de R², confirmando que o modelo efetivamente aprende padrões do perfil profissional.

### 7.4 Caminhos para Melhora

Para aumentar o R² futuramente, as principais alavancas são:

- Adicionar variáveis: porte da empresa, cidade específica, área de atuação dentro da profissão;
- Engenharia de interações explícitas: `Profissao × Escolaridade`, `Regiao × Profissao`;
- Ampliar o dataset (dados reais, maior volume);
- Tuning de hiperparâmetros via `GridSearchCV` ou `Optuna`.

---

## 8. Deploy e Serviço

### 8.1 API REST (FastAPI)

A API expõe os seguintes endpoints:

| Método | Rota | Descrição |
|---|---|---|
| GET | `/` | Página inicial com disclaimer e links |
| GET | `/modelo` | Metadados do modelo treinado (nome, RMSE, MAE, R², MAPE) |
| POST | `/predizer` | Predição individual dado um perfil profissional |
| POST | `/predizer/batch` | Predição em lote (até 100 perfis por requisição) |
| GET | `/exemplos` | Exemplos de payload prontos para teste |

O schema de entrada (`PerfilProfissional`) é validado por **Pydantic v2** com checagem de domínio em todos os campos categóricos. O resultado inclui o salário estimado, o intervalo de confiança (`± MAE`), o nome do modelo usado e um aviso educacional obrigatório.

A API é iniciada com:

```bash
uvicorn src.api.app:app --reload
```

### 8.2 Interface Gradio (Hugging Face Spaces)

O arquivo `app_hf.py` implementa uma interface Gradio com:

- Dropdowns para profissão, escolaridade, região e idiomas;
- Sliders para idade e anos de experiência;
- Exemplos pré-carregados (Médico, Vendedor, Engenheiro, Programador);
- Resultado formatado em Markdown com salário estimado, intervalo e métricas do modelo.

Para publicar no Hugging Face Spaces, basta renomear `app_hf.py` para `app.py`, adicionar `gradio` ao `requirements.txt` e fazer upload dos artefatos (`models/best_model.joblib`, `models/best_model_meta.json`, `src/pipeline/`).

### 8.3 Docker

O `Dockerfile` usa a imagem `python:3.11-slim`, instala as dependências, copia `src/`, `data/` e `models/`, e inicia a API com Uvicorn na porta 8000. Inclui healthcheck automático a cada 30 segundos.

```bash
docker build -t salarios-ml .
docker run -p 8000:8000 salarios-ml
```

---

## 9. Como Reproduzir o Projeto

```bash
# 1. Instalar dependências
pip install -r requirements.txt

# 2. Treinar os modelos (executar da raiz do projeto)
python src/pipeline/train.py
# → cria models/, results/ e mlruns/

# 3. Visualizar experimentos no MLflow
python mlflow_ui.py
# → acesse http://localhost:5000

# 4. Subir a API
uvicorn src.api.app:app --reload
# → acesse http://localhost:8000/docs

# 5. Analisar resultados (notebook)
# Abra notebooks/02_resultados.py no VS Code / Jupyter
```

---

## 10. Metodologia CRISP-DM

| Fase | O que foi feito |
|---|---|
| **1. Entendimento do Negócio** | Definição do objetivo: estimar salário para suporte a decisões de carreira e RH |
| **2. Entendimento dos Dados** | EDA exploratória; identificação de inconsistências (idade/experiência, profissão/escolaridade), outliers e distribuição do target |
| **3. Preparação dos Dados** | `preprocessing.py` (limpeza + imputação + feature engineering) + `features.py` (encoding + escalonamento) |
| **4. Modelagem** | `train.py`: 9 modelos, CV-5, MLflow, `sklearn.Pipeline` |
| **5. Avaliação** | `results/model_comparison.csv` + `notebooks/02_resultados.py` (gráficos e análise de overfitting) |
| **6. Deploy** | FastAPI (local/Docker) + Gradio (Hugging Face Spaces) |
