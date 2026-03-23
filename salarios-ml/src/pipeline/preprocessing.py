"""
preprocessing.py
================
Módulo de limpeza e preparação dos dados.

Filtros aplicados (baseados na EDA):
  1. Salário "erro" → NaN após to_numeric → removido (não imputamos o target)
  2. Salários nulos ou <= 0 → removidos
  3. Nulos em Idade e Anos_Experiencia → imputados com mediana (necessário antes dos
     filtros seguintes, que dependem dessas colunas)
  4. dif_idade_exp = Idade - Anos_Experiencia < 14 → removidos
     Justificativa: ninguém começa a trabalhar antes dos ~14 anos; diferença menor
     que isso indica dado inconsistente (ex: 25 anos com 26 de experiência)
  5. Profissão que exige superior + escolaridade < Superior → removidos
     Profissões: Advogado, Agrônomo, Arquiteto, Cientista, Economista,
                 Engenheiro, Médico, Pesquisador
  6. Outliers no salário > p99.5 → removidos

Dataset final: ~8.052 registros (53.7% do original de 15.000)
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constantes de domínio
# ---------------------------------------------------------------------------
ORDEM_ESCOLARIDADE = [
    "Fundamental", "Médio", "Técnico", "Superior", "Pós", "Mestrado", "Doutorado"
]

REGIOES_VALIDAS = ["Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"]

PROFISSOES_VALIDAS = [
    "Administrador", "Advogado", "Agrônomo", "Analista de Dados", "Arquiteto",
    "Cientista", "Consultor", "Designer", "Economista", "Enfermeiro",
    "Engenheiro", "Motorista", "Médico", "Pesquisador", "Policial",
    "Professor", "Programador", "Técnico", "Vendedor",
]

# Profissões que exigem, no mínimo, ensino Superior
PROFISSOES_EXIGEM_SUPERIOR = {
    "Advogado", "Agrônomo", "Arquiteto", "Cientista",
    "Economista", "Engenheiro", "Médico", "Pesquisador",
}

NIVEL_ESC = {e: i for i, e in enumerate(ORDEM_ESCOLARIDADE)}

OUTLIER_PERCENTIL    = 0.995
DIF_IDADE_EXP_MINIMA = 14      # Idade - Anos_Experiencia mínimo aceitável


# ---------------------------------------------------------------------------
# Carregamento
# ---------------------------------------------------------------------------

def carregar_dados(caminho: str) -> pd.DataFrame:
    """Carrega o CSV e força tipos corretos."""
    df = pd.read_csv(caminho)
    df["Salario"]          = pd.to_numeric(df["Salario"],          errors="coerce")
    df["Idade"]            = pd.to_numeric(df["Idade"],            errors="coerce")
    df["Anos_Experiencia"] = pd.to_numeric(df["Anos_Experiencia"], errors="coerce")

    for col in ["Escolaridade", "Regiao", "Profissao", "Segunda_Lingua", "Terceira_Lingua"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    return df


# ---------------------------------------------------------------------------
# Limpeza principal
# ---------------------------------------------------------------------------

def limpar_dados(df: pd.DataFrame,
                 outlier_percentil: float = OUTLIER_PERCENTIL) -> pd.DataFrame:
    """
    Pipeline de limpeza com todos os filtros de inconsistência da EDA.

    Ordem das etapas:
      1. Remove salários nulos/inválidos
      2. Remove salários <= 0
      3. Imputa nulos em Idade e Anos_Experiencia com mediana
         (antes dos filtros que dependem dessas colunas)
      4. Remove (Idade - Anos_Experiencia) < 14  → dado impossível
      5. Remove profissão que exige superior + escolaridade insuficiente
      6. Remove outliers de salário > p99.5
    """
    n_original = len(df)
    log = []

    # 1. Target nulo (inclui valores "erro" convertidos para NaN)
    n = len(df)
    df = df.dropna(subset=["Salario"]).copy()
    log.append(f"Salário nulo/inválido: {n - len(df):,} removidos")

    # 2. Salário não positivo
    n = len(df)
    df = df[df["Salario"] > 0]
    log.append(f"Salário <= 0: {n - len(df):,} removidos")

    # 3. Imputação de nulos numéricos — ANTES dos filtros de consistência
    for col in ["Idade", "Anos_Experiencia"]:
        mediana = df[col].median()
        nulos = df[col].isna().sum()
        if nulos > 0:
            df[col] = df[col].fillna(mediana)
            log.append(f"Imputados {nulos:,} nulos em '{col}' com mediana={mediana:.1f}")

    # 4. Consistência: Idade - Anos_Experiencia >= 14
    df["dif_idade_exp"] = df["Idade"] - df["Anos_Experiencia"]
    n = len(df)
    df = df[df["dif_idade_exp"] >= DIF_IDADE_EXP_MINIMA]
    log.append(f"dif(Idade - Exp) < {DIF_IDADE_EXP_MINIMA}: {n - len(df):,} removidos")

    # 5. Consistência: profissão que exige superior com escolaridade insuficiente
    df["_nivel_esc"] = df["Escolaridade"].map(NIVEL_ESC)
    n = len(df)
    mask_inc = (
        df["Profissao"].isin(PROFISSOES_EXIGEM_SUPERIOR) &
        (df["_nivel_esc"] < NIVEL_ESC["Superior"])
    )
    df = df[~mask_inc].drop(columns=["_nivel_esc"])
    log.append(f"Prof. exige superior + esc. insuficiente: {n - len(df):,} removidos")

    # 6. Outliers extremos no salário
    corte = df["Salario"].quantile(outlier_percentil)
    n = len(df)
    df = df[df["Salario"] <= corte]
    log.append(f"Outliers > p{outlier_percentil*100:.1f} (R${corte:,.0f}): {n - len(df):,} removidos")

    df = df.reset_index(drop=True)

    print(f"\n{'='*50}")
    print(f"  RESUMO DA LIMPEZA")
    print(f"{'='*50}")
    for linha in log:
        print(f"  → {linha}")
    print(f"  {'─'*46}")
    print(f"  Original : {n_original:,}")
    print(f"  Final    : {len(df):,} ({len(df)/n_original*100:.1f}% do original)")
    print(f"{'='*50}\n")

    return df


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def engenharia_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features derivadas dos insights da EDA.

    - tem_segunda_lingua  : flag 0/1
    - tem_terceira_lingua : flag 0/1
    - total_linguas       : 0, 1 ou 2
    - dif_idade_exp       : já criada na limpeza; garantida aqui se ausente
    """
    df = df.copy()

    df["tem_segunda_lingua"]  = (df["Segunda_Lingua"]  != "Nenhuma").astype(int)
    df["tem_terceira_lingua"] = (df["Terceira_Lingua"] != "Nenhuma").astype(int)
    df["total_linguas"]       = df["tem_segunda_lingua"] + df["tem_terceira_lingua"]

    if "dif_idade_exp" not in df.columns:
        df["dif_idade_exp"] = df["Idade"] - df["Anos_Experiencia"]

    return df


def criar_target_log(df: pd.DataFrame) -> pd.Series:
    """Retorna o target transformado: log1p(Salario)."""
    return np.log1p(df["Salario"])
