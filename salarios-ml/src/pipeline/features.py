"""
features.py
===========
Constrói o ColumnTransformer (pré-processamento + encoding) para o sklearn Pipeline.

Decisões de encoding (baseadas na EDA):
- Escolaridade → OrdinalEncoder (hierarquia clara: Fundamental < Doutorado)
- Profissão, Região → OneHotEncoder (sem ordem natural)
- Segunda/Terceira Língua → OneHotEncoder (categórica nominal)
- Numéricas → StandardScaler (necessário para Ridge/SVR; tree-based não precisa,
  mas não prejudica e facilita a comparação uniforme dos coeficientes)
- Features binárias (flags de língua) → passthrough (já são 0/1)
"""

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OrdinalEncoder,
    OneHotEncoder,
    StandardScaler,
)

from .preprocessing import ORDEM_ESCOLARIDADE

# ---------------------------------------------------------------------------
# Definição das colunas por tipo de transformação
# ---------------------------------------------------------------------------
COLS_NUMERICAS = ["Idade", "Anos_Experiencia", "dif_idade_exp", "total_linguas"]

COLS_ORDINAIS = ["Escolaridade"]
CATEGORIAS_ORDINAIS = [ORDEM_ESCOLARIDADE]  # ColumnTransformer espera lista de listas

COLS_NOMINAIS = ["Profissao", "Regiao", "Segunda_Lingua", "Terceira_Lingua"]

COLS_BINARIAS = ["tem_segunda_lingua", "tem_terceira_lingua"]   # já são 0/1


def build_preprocessor() -> ColumnTransformer:
    """
    Retorna o ColumnTransformer completo.

    Uso dentro de um Pipeline:
        pipe = Pipeline([
            ('pre', build_preprocessor()),
            ('model', SeuModelo()),
        ])
    """
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                COLS_NUMERICAS,
            ),
            (
                "ord",
                OrdinalEncoder(
                    categories=CATEGORIAS_ORDINAIS,
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
                COLS_ORDINAIS,
            ),
            (
                "nom",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                COLS_NOMINAIS,
            ),
            (
                "bin",
                "passthrough",
                COLS_BINARIAS,
            ),
        ],
        remainder="drop",   # descarta colunas não listadas (ex: Salario)
        verbose_feature_names_out=True,
    )
    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Retorna nomes legíveis das features após transformação (para importância)."""
    return list(preprocessor.get_feature_names_out())
