"""
train.py
========
Script principal de treinamento. Rode sempre a partir da raiz do projeto:

    python src/pipeline/train.py

Fluxo CRISP-DM:
  1. Carrega e limpa os dados (todos os filtros de inconsistência)
  2. Feature engineering
  3. Split 80/20 estratificado por Profissão
  4. Para cada modelo: treina, avalia com CV-5 e holdout, loga no MLflow
  5. Salva o melhor modelo em models/best_model.joblib

Métricas (calculadas na escala original R$, revertendo o log1p):
  RMSE, MAE, R², MAPE
"""

import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pipeline.features import build_preprocessor
from pipeline.models import get_models
from pipeline.preprocessing import (
    carregar_dados,
    criar_target_log,
    engenharia_features,
    limpar_dados,
)

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------
DATA_PATH   = PROJECT_ROOT / "data" / "dataset_salarios_brasil.csv"
MODELS_DIR  = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
MLFLOW_URI  = str(PROJECT_ROOT / "mlruns")  # absoluto — funciona de qualquer diretório
EXPERIMENT  = "salarios-brasil"
SEED        = 42
TEST_SIZE   = 0.2
CV_FOLDS    = 5

MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Métricas na escala original R$
# ---------------------------------------------------------------------------
def calcular_metricas(y_log_true: np.ndarray, y_log_pred: np.ndarray) -> dict:
    y_true = np.expm1(y_log_true)
    y_pred = np.clip(np.expm1(y_log_pred), 0, None)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae":  float(mean_absolute_error(y_true, y_pred)),
        "r2":   float(r2_score(y_true, y_pred)),
        "mape": float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100),
    }


# ---------------------------------------------------------------------------
# Treinamento + logging MLflow
# ---------------------------------------------------------------------------
def treinar_todos(df_train, df_test, y_train, y_test) -> list[dict]:
    modelos = get_models()
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    resultados = []

    for nome, estimador in modelos.items():
        print(f"\n{'─'*55}")
        print(f"  {nome}")
        print(f"{'─'*55}")

        with mlflow.start_run(run_name=nome):

            mlflow.set_tags({
                "modelo":  nome,
                "target":  "log1p(Salario)",
                "dataset": "salarios_brasil",
                "filtros": "dif_idade_exp>=14 | prof_esc_consistente | outliers_p99.5",
            })

            pipe = Pipeline([
                ("pre",   build_preprocessor()),
                ("model", estimador),
            ])

            # Cross-validation (métrica em escala log — comparação entre modelos)
            cv_neg = cross_val_score(
                pipe, df_train, y_train,
                cv=kf, scoring="neg_root_mean_squared_error", n_jobs=-1,
            )
            cv_rmse_log = -cv_neg

            # Treino completo
            t0 = time.time()
            pipe.fit(df_train, y_train)
            tempo = time.time() - t0

            # Métricas em R$ (escala original)
            mt = calcular_metricas(y_test.values,  pipe.predict(df_test))
            mtr= calcular_metricas(y_train.values, pipe.predict(df_train))

            # Parâmetros simples do estimador (MLflow não aceita objetos)
            params = {
                k: v for k, v in estimador.get_params().items()
                if isinstance(v, (int, float, str, bool, type(None)))
            }
            mlflow.log_params(params)

            mlflow.log_metrics({
                "rmse_teste":          mt["rmse"],
                "mae_teste":           mt["mae"],
                "r2_teste":            mt["r2"],
                "mape_teste":          mt["mape"],
                "rmse_treino":         mtr["rmse"],
                "r2_treino":           mtr["r2"],
                "cv_rmse_log_media":   float(cv_rmse_log.mean()),
                "cv_rmse_log_desvio":  float(cv_rmse_log.std()),
                "tempo_treino_s":      tempo,
            })

            # Salva modelo em disco (para a API)
            safe_name = nome.replace(" ", "_").replace("(", "").replace(")", "")
            model_path = MODELS_DIR / f"{safe_name}.joblib"
            joblib.dump(pipe, model_path)

            # Registra o artefato .joblib no MLflow
            mlflow.log_artifact(str(model_path), artifact_path="model")

            resultado = {
                "nome": nome, **mt,
                "rmse_treino": mtr["rmse"], "r2_treino": mtr["r2"],
                "cv_rmse_log_media": float(cv_rmse_log.mean()),
                "tempo_treino_s": tempo,
                "model_path": str(model_path),
            }
            resultados.append(resultado)

            print(f"  RMSE : R$ {mt['rmse']:>10,.2f}   MAE: R$ {mt['mae']:>8,.2f}")
            print(f"  R²   : {mt['r2']:>8.4f}          MAPE: {mt['mape']:.1f}%")
            print(f"  ΔR²  : {mtr['r2'] - mt['r2']:>+.4f} (overfitting)   {tempo:.1f}s")

    return resultados


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("\n" + "="*55)
    print("  PIPELINE ML — SALÁRIOS BRASIL")
    print("="*55)

    # 1. Dados
    print("\n[1/5] Carregando e limpando dados...")
    df_raw   = carregar_dados(str(DATA_PATH))
    df_clean = limpar_dados(df_raw)
    df_feat  = engenharia_features(df_clean)
    target   = criar_target_log(df_feat)
    print(f"  Dataset final: {len(df_feat):,} registros, {df_feat.shape[1]} colunas")

    # 2. Split
    print("\n[2/5] Split 80/20 estratificado por Profissão...")
    df_train, df_test, y_train, y_test = train_test_split(
        df_feat, target,
        test_size=TEST_SIZE, random_state=SEED,
        stratify=df_feat["Profissao"],
    )
    print(f"  Treino: {len(df_train):,} | Teste: {len(df_test):,}")

    # 3. MLflow
    #    IMPORTANTE: mlflow.set_tracking_uri com caminho relativo
    #    funciona corretamente quando o script é rodado da raiz do projeto.
    print("\n[3/5] Configurando MLflow...")
    mlflow.set_tracking_uri(MLFLOW_URI)
    experiment = mlflow.set_experiment(EXPERIMENT)
    print(f"  URI        : {MLFLOW_URI}/")
    print(f"  Experiment : {EXPERIMENT} (id={experiment.experiment_id})")

    # 4. Treinamento
    print("\n[4/5] Treinando modelos...")
    resultados = treinar_todos(df_train, df_test, y_train, y_test)

    # 5. Ranking e seleção
    print("\n[5/5] Selecionando melhor modelo...")
    df_res = pd.DataFrame(resultados).sort_values("rmse")

    print("\n" + "="*55)
    print("  RANKING (por RMSE no teste)")
    print("="*55)
    for _, row in df_res.iterrows():
        flag = " ⭐" if _ == df_res.index[0] else ""
        print(f"  {row['nome']:<20} R${row['rmse']:>9,.0f}  R²={row['r2']:.4f}{flag}")

    melhor = df_res.iloc[0]
    best_pipe = joblib.load(melhor["model_path"])
    joblib.dump(best_pipe, MODELS_DIR / "best_model.joblib")

    meta = {
        "nome":       melhor["nome"],
        "rmse_teste": melhor["rmse"],
        "mae_teste":  melhor["mae"],
        "r2_teste":   melhor["r2"],
        "mape_teste": melhor["mape"],
    }
    with open(MODELS_DIR / "best_model_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    df_res.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)

    print(f"\n  Melhor modelo : {melhor['nome']}")
    print(f"  RMSE          : R$ {melhor['rmse']:,.2f}")
    print(f"  MAE           : R$ {melhor['mae']:,.2f}")
    print(f"  R²            : {melhor['r2']:.4f}")
    print(f"\n  Salvo em      : models/best_model.joblib")
    print(f"  Resultados    : results/model_comparison.csv")
    print(f"\n  Ver no MLflow : mlflow ui --backend-store-uri mlruns")
    print("\n✅ Concluído!\n")


if __name__ == "__main__":
    main()
