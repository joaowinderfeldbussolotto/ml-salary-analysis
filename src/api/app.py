"""
app.py
======
API de predição de salários com FastAPI.

Endpoints:
  GET  /           → health check + info do modelo
  GET  /modelo     → metadados do melhor modelo treinado
  POST /predizer   → predição de salário dado perfil profissional
  POST /predizer/batch → predição em lote (lista de perfis)
  GET  /exemplos   → retorna exemplos de payload para facilitar testes

Nota importante para o deploy:
  A página raiz inclui o disclaimer exigido pela atividade:
  "Este conteúdo é destinado apenas para fins educacionais..."
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import joblib

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR   = PROJECT_ROOT / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
META_PATH       = MODELS_DIR / "best_model_meta.json"

# ---------------------------------------------------------------------------
# Carregar modelo e metadados
# ---------------------------------------------------------------------------
_model = None
_meta  = {}

def carregar_modelo():
    global _model, _meta
    if not BEST_MODEL_PATH.exists():
        raise RuntimeError(
            f"Modelo não encontrado em {BEST_MODEL_PATH}. "
            "Execute 'python src/pipeline/train.py' primeiro."
        )
    _model = joblib.load(BEST_MODEL_PATH)
    if META_PATH.exists():
        with open(META_PATH, encoding="utf-8") as f:
            _meta = json.load(f)


# ---------------------------------------------------------------------------
# Schema de entrada
# ---------------------------------------------------------------------------
ESCOLARIDADES_VALIDAS = [
    "Fundamental", "Médio", "Técnico", "Superior", "Pós", "Mestrado", "Doutorado"
]
REGIOES_VALIDAS = ["Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"]
PROFISSOES_VALIDAS = [
    "Administrador", "Advogado", "Agrônomo", "Analista de Dados", "Arquiteto",
    "Cientista", "Consultor", "Designer", "Economista", "Enfermeiro",
    "Engenheiro", "Motorista", "Médico", "Pesquisador", "Policial",
    "Professor", "Programador", "Técnico", "Vendedor",
]
LINGUAS_VALIDAS = [
    "Nenhuma", "Inglês", "Espanhol", "Francês", "Alemão", "Italiano",
    "Mandarim", "Japonês", "Português",
]


class PerfilProfissional(BaseModel):
    idade: float = Field(..., ge=14, le=80, description="Idade em anos (14–80)")
    anos_experiencia: float = Field(..., ge=0, le=60, description="Anos de experiência profissional")
    escolaridade: str = Field(..., description=f"Um de: {ESCOLARIDADES_VALIDAS}")
    regiao: str = Field(..., description=f"Uma de: {REGIOES_VALIDAS}")
    profissao: str = Field(..., description=f"Uma de: {PROFISSOES_VALIDAS}")
    segunda_lingua: str = Field(default="Nenhuma", description="Segunda língua (opcional)")
    terceira_lingua: str = Field(default="Nenhuma", description="Terceira língua (opcional)")

    @field_validator("escolaridade")
    @classmethod
    def validar_escolaridade(cls, v):
        if v not in ESCOLARIDADES_VALIDAS:
            raise ValueError(f"Escolaridade inválida. Válidas: {ESCOLARIDADES_VALIDAS}")
        return v

    @field_validator("regiao")
    @classmethod
    def validar_regiao(cls, v):
        if v not in REGIOES_VALIDAS:
            raise ValueError(f"Região inválida. Válidas: {REGIOES_VALIDAS}")
        return v

    @field_validator("profissao")
    @classmethod
    def validar_profissao(cls, v):
        if v not in PROFISSOES_VALIDAS:
            raise ValueError(f"Profissão inválida. Válidas: {PROFISSOES_VALIDAS}")
        return v

    @field_validator("segunda_lingua", "terceira_lingua")
    @classmethod
    def validar_lingua(cls, v):
        if v not in LINGUAS_VALIDAS:
            raise ValueError(f"Língua inválida. Válidas: {LINGUAS_VALIDAS}")
        return v


class ResultadoPredição(BaseModel):
    salario_estimado: float
    salario_formatado: str
    intervalo_inferior: float   # ± MAE do modelo
    intervalo_superior: float
    modelo_usado: str
    aviso: str


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="API de Predição de Salários — Brasil",
    description=(
        "Predição de salário com base em perfil profissional. "
        "**Este conteúdo é destinado apenas para fins educacionais. "
        "Os dados exibidos são ilustrativos e podem não corresponder a situações reais.**"
    ),
    version="1.0.0",
)


@app.on_event("startup")
def startup():
    carregar_modelo()


# ---------------------------------------------------------------------------
# Função auxiliar de predição
# ---------------------------------------------------------------------------
def _preparar_df(perfil: PerfilProfissional) -> pd.DataFrame:
    """Converte o schema Pydantic para DataFrame com todas as features esperadas."""
    d = {
        "Idade":             [perfil.idade],
        "Anos_Experiencia":  [perfil.anos_experiencia],
        "Escolaridade":      [perfil.escolaridade],
        "Regiao":            [perfil.regiao],
        "Profissao":         [perfil.profissao],
        "Segunda_Lingua":    [perfil.segunda_lingua],
        "Terceira_Lingua":   [perfil.terceira_lingua],
        # Features derivadas (engenharia de features)
        "tem_segunda_lingua":  [int(perfil.segunda_lingua != "Nenhuma")],
        "tem_terceira_lingua": [int(perfil.terceira_lingua != "Nenhuma")],
        "total_linguas":       [int(perfil.segunda_lingua != "Nenhuma") +
                                int(perfil.terceira_lingua != "Nenhuma")],
        "dif_idade_exp":      [perfil.idade - perfil.anos_experiencia],
    }
    return pd.DataFrame(d)


def _predizer(perfil: PerfilProfissional) -> dict:
    df = _preparar_df(perfil)
    log_pred = _model.predict(df)[0]
    salario = float(np.expm1(log_pred))
    salario = max(salario, 0)

    mae = _meta.get("mae_teste", salario * 0.15)  # fallback: ±15%
    return {
        "salario_estimado": round(salario, 2),
        "salario_formatado": f"R$ {salario:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."),
        "intervalo_inferior": round(max(0, salario - mae), 2),
        "intervalo_superior": round(salario + mae, 2),
        "modelo_usado": _meta.get("nome", "desconhecido"),
        "aviso": (
            "Este conteúdo é destinado apenas para fins educacionais. "
            "Os dados exibidos são ilustrativos e podem não corresponder a situações reais."
        ),
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse, summary="Página inicial")
def raiz():
    """Página de boas-vindas com disclaimer e links para a documentação."""
    html = """
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
      <meta charset="UTF-8"/>
      <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
      <title>API Salários Brasil</title>
      <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 60px auto;
               padding: 0 20px; color: #333; background: #f9f9f9; }
        h1   { color: #2563eb; }
        .aviso { background: #fef9c3; border-left: 4px solid #ca8a04;
                 padding: 12px 16px; border-radius: 4px; margin: 20px 0; }
        .links a { display: inline-block; margin-right: 16px; color: #2563eb;
                   text-decoration: none; font-weight: bold; }
        .links a:hover { text-decoration: underline; }
        table { border-collapse: collapse; width: 100%; margin-top: 16px; }
        th, td { border: 1px solid #ddd; padding: 8px 12px; text-align: left; }
        th { background: #2563eb; color: white; }
      </style>
    </head>
    <body>
      <h1>🏦 API de Predição de Salários — Brasil</h1>

      <div class="aviso">
        ⚠️ <strong>Aviso:</strong> Este conteúdo é destinado apenas para fins educacionais.
        Os dados exibidos são ilustrativos e podem não corresponder a situações reais.
      </div>

      <p>API desenvolvida como projeto final da UC Aprendizado de Máquina 2026/1.</p>

      <div class="links">
        <a href="/docs">📖 Documentação Interativa (Swagger)</a>
        <a href="/redoc">📋 ReDoc</a>
        <a href="/modelo">🤖 Info do Modelo</a>
        <a href="/exemplos">💡 Exemplos</a>
      </div>

      <h2>Endpoints disponíveis</h2>
      <table>
        <tr><th>Método</th><th>Rota</th><th>Descrição</th></tr>
        <tr><td>GET</td><td>/</td><td>Esta página</td></tr>
        <tr><td>GET</td><td>/modelo</td><td>Metadados do modelo treinado</td></tr>
        <tr><td>POST</td><td>/predizer</td><td>Predição individual de salário</td></tr>
        <tr><td>POST</td><td>/predizer/batch</td><td>Predição em lote</td></tr>
        <tr><td>GET</td><td>/exemplos</td><td>Exemplos de payload</td></tr>
      </table>
    </body>
    </html>
    """
    return HTMLResponse(content=html)


@app.get("/modelo", summary="Metadados do modelo")
def info_modelo():
    """Retorna informações sobre o melhor modelo treinado."""
    if not _meta:
        raise HTTPException(status_code=503, detail="Modelo não carregado.")
    return {
        "status": "ok",
        "modelo": _meta,
        "aviso": (
            "Este conteúdo é destinado apenas para fins educacionais. "
            "Os dados exibidos são ilustrativos e podem não corresponder a situações reais."
        ),
    }


@app.post("/predizer", response_model=ResultadoPredição, summary="Predição individual")
def predizer(perfil: PerfilProfissional):
    """
    Recebe um perfil profissional e retorna o salário estimado em R$.

    O intervalo de confiança é calculado com base no MAE do modelo no conjunto de teste.
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Modelo não disponível.")
    try:
        return _predizer(perfil)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predizer/batch", summary="Predição em lote")
def predizer_batch(perfis: list[PerfilProfissional]):
    """Recebe uma lista de perfis e retorna predições para todos."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Modelo não disponível.")
    if len(perfis) > 100:
        raise HTTPException(status_code=400, detail="Máximo de 100 perfis por requisição.")
    resultados = [_predizer(p) for p in perfis]
    return {
        "total": len(resultados),
        "predições": resultados,
        "aviso": (
            "Este conteúdo é destinado apenas para fins educacionais. "
            "Os dados exibidos são ilustrativos e podem não corresponder a situações reais."
        ),
    }


@app.get("/exemplos", summary="Exemplos de payload")
def exemplos():
    """Retorna exemplos de requisição para facilitar os testes."""
    return {
        "exemplo_medico_sp": {
            "idade": 42,
            "anos_experiencia": 15,
            "escolaridade": "Doutorado",
            "regiao": "Sudeste",
            "profissao": "Médico",
            "segunda_lingua": "Inglês",
            "terceira_lingua": "Nenhuma",
        },
        "exemplo_vendedor_nordeste": {
            "idade": 28,
            "anos_experiencia": 4,
            "escolaridade": "Médio",
            "regiao": "Nordeste",
            "profissao": "Vendedor",
            "segunda_lingua": "Nenhuma",
            "terceira_lingua": "Nenhuma",
        },
        "exemplo_engenheiro_sul": {
            "idade": 35,
            "anos_experiencia": 10,
            "escolaridade": "Superior",
            "regiao": "Sul",
            "profissao": "Engenheiro",
            "segunda_lingua": "Inglês",
            "terceira_lingua": "Espanhol",
        },
        "curl_exemplo": (
            "curl -X POST http://localhost:8000/predizer "
            "-H 'Content-Type: application/json' "
            "-d '{\"idade\":35,\"anos_experiencia\":10,\"escolaridade\":\"Superior\","
            "\"regiao\":\"Sul\",\"profissao\":\"Engenheiro\","
            "\"segunda_lingua\":\"Inglês\",\"terceira_lingua\":\"Nenhuma\"}'"
        ),
    }
