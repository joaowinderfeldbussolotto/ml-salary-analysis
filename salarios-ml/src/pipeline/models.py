"""
models.py
=========
Catálogo de modelos candidatos para comparação.

Estratégia de seleção:
  Testamos modelos de diferentes famílias para cobrir:
  - Modelos lineares (baseline e regularizados): Ridge, Lasso, ElasticNet
  - Árvores de decisão simples: DecisionTreeRegressor
  - Ensembles (bagging): RandomForestRegressor — robusto, bom baseline não-linear
  - Ensembles (boosting): GradientBoostingRegressor — geralmente o melhor para tabulares
  - Vizinhança: KNeighborsRegressor — sensível a escala, bom para checar padrões locais

  XGBoost e LightGBM são incluídos condicionalmente se instalados, pois
  costumam superar o GradientBoosting do sklearn em datasets maiores.

Todos os modelos têm hiperparâmetros razoáveis para um primeiro experimento;
o script de treino faz GridSearchCV nos mais promissores.
"""

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor

# Tentar importar boosters opcionais
try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

try:
    from lightgbm import LGBMRegressor
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False


SEED = 42


def get_models() -> dict:
    """
    Retorna dicionário {nome: estimador} com todos os candidatos.
    Os estimadores NÃO incluem o pré-processamento (ficam dentro de Pipeline).
    """
    modelos = {
        # ── Baseline ─────────────────────────────────────────────────────────
        "Dummy (mediana)": DummyRegressor(strategy="median"),

        # # ── Lineares ─────────────────────────────────────────────────────────
        # # Ridge: regressão linear com regularização L2.
        # #   → bom baseline; rápido; interpretável pelos coeficientes.
        # "Ridge": Ridge(alpha=10.0),

        # Lasso: regularização L1; zera features irrelevantes automaticamente.
        #   → útil para seleção implícita de variáveis.
        "Lasso": Lasso(alpha=0.01, max_iter=5000),

        # ElasticNet: combinação L1+L2; mais estável que Lasso puro.
        "ElasticNet": ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000),

        # ── Vizinhança ────────────────────────────────────────────────────────
        # KNN: captura padrões locais; sensível a escala (StandardScaler já aplicado).
        "KNN (k=10)": KNeighborsRegressor(n_neighbors=10, weights="distance"),

        # ── Árvore simples ───────────────────────────────────────────────────
        # Árvore de decisão: interpretável; propenso a overfitting sem poda.
        "DecisionTree": DecisionTreeRegressor(max_depth=10, random_state=SEED),

        # ── Ensemble Bagging ─────────────────────────────────────────────────
        # Random Forest: reduz variância via média de N árvores.
        "RandomForest": RandomForestRegressor(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=SEED,
        ),

        # ── Ensemble Boosting ────────────────────────────────────────────────
        # GradientBoosting: treina árvores sequencialmente nos resíduos.
        #   → alta acurácia em dados tabulares; mais lento que RF.
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            random_state=SEED,
        ),
    }

    # Adiciona XGBoost se disponível
    if _HAS_XGB:
        modelos["XGBoost"] = XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=SEED,
            verbosity=0,
        )

    # Adiciona LightGBM se disponível
    if _HAS_LGB:
        modelos["LightGBM"] = LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=SEED,
            verbose=-1,
        )

    return modelos


# Hiperparâmetros para GridSearchCV nos top modelos
# Usamos prefixo "model__" pois os estimadores ficam dentro de Pipeline
PARAM_GRIDS = {
    "Ridge": {
        "model__alpha": [0.1, 1.0, 10.0, 100.0],
    },
    "RandomForest": {
        "model__n_estimators": [100, 200],
        "model__max_depth": [None, 15, 25],
        "model__min_samples_leaf": [3, 5],
    },
    "GradientBoosting": {
        "model__n_estimators": [200, 300],
        "model__learning_rate": [0.03, 0.05, 0.1],
        "model__max_depth": [4, 5, 6],
    },
}
