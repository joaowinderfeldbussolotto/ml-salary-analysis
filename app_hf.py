"""
app_hf.py
=========
Versão adaptada para deploy no Hugging Face Spaces (Gradio).

No Hugging Face Spaces, o arquivo principal deve se chamar app.py.
Renomeie este arquivo para app.py ao fazer upload para o Space.

Estrutura esperada no Space:
  app.py                  ← este arquivo (renomeado)
  requirements.txt
  models/
    best_model.joblib
    best_model_meta.json
  src/
    pipeline/
      preprocessing.py
      features.py
"""

import json
import sys
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import joblib

# ---------------------------------------------------------------------------
# Caminhos — ajustados para Hugging Face Spaces
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

BEST_MODEL_PATH = BASE_DIR / "models" / "best_model.joblib"
META_PATH       = BASE_DIR / "models" / "best_model_meta.json"

# Carregar modelo
model = joblib.load(BEST_MODEL_PATH)
with open(META_PATH, encoding="utf-8") as f:
    meta = json.load(f)

# ---------------------------------------------------------------------------
# Constantes de domínio
# ---------------------------------------------------------------------------
ESCOLARIDADES = ["Fundamental", "Médio", "Técnico", "Superior", "Pós", "Mestrado", "Doutorado"]
REGIOES       = ["Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"]
PROFISSOES    = sorted([
    "Administrador", "Advogado", "Agrônomo", "Analista de Dados", "Arquiteto",
    "Cientista", "Consultor", "Designer", "Economista", "Enfermeiro",
    "Engenheiro", "Motorista", "Médico", "Pesquisador", "Policial",
    "Professor", "Programador", "Técnico", "Vendedor",
])
LINGUAS = ["Nenhuma", "Inglês", "Espanhol", "Francês", "Alemão", "Italiano",
           "Mandarim", "Japonês", "Português"]

# ---------------------------------------------------------------------------
# Função de predição
# ---------------------------------------------------------------------------
def predizer(profissao, escolaridade, regiao, idade, anos_exp, segunda_lingua, terceira_lingua):
    df = pd.DataFrame({
        "Idade":             [float(idade)],
        "Anos_Experiencia":  [float(anos_exp)],
        "Escolaridade":      [escolaridade],
        "Regiao":            [regiao],
        "Profissao":         [profissao],
        "Segunda_Lingua":    [segunda_lingua],
        "Terceira_Lingua":   [terceira_lingua],
        "tem_segunda_lingua":  [int(segunda_lingua != "Nenhuma")],
        "tem_terceira_lingua": [int(terceira_lingua != "Nenhuma")],
        "total_linguas":       [int(segunda_lingua != "Nenhuma") + int(terceira_lingua != "Nenhuma")],
        "dif_idade_exp":       [float(idade) - float(anos_exp)],
    })

    log_pred = model.predict(df)[0]
    salario  = float(np.expm1(log_pred))
    salario  = max(salario, 0)
    mae      = meta.get("mae_teste", salario * 0.15)

    resultado = f"""
## 💰 Salário Estimado

**{f"R$ {salario:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")}**

**Intervalo estimado:** R$ {max(0,salario-mae):,.0f} — R$ {salario+mae:,.0f}

---

**Modelo:** {meta.get('nome', 'N/A')}  
**R² (teste):** {meta.get('r2_teste', 0):.4f}  
**MAE (teste):** R$ {meta.get('mae_teste', 0):,.0f}

---

> ⚠️ **Este conteúdo é destinado apenas para fins educacionais.**  
> Os dados exibidos são ilustrativos e podem não corresponder a situações reais.
""".replace(",", "X").replace(".", ",").replace("X", ".")
    return resultado


# ---------------------------------------------------------------------------
# Interface Gradio
# ---------------------------------------------------------------------------
with gr.Blocks(title="Predição de Salários — Brasil", theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
    # 🏦 Predição de Salários — Brasil
    ### UC Aprendizado de Máquina 2026/1

    Preencha o perfil profissional e clique em **Calcular** para obter a estimativa salarial.

    > ⚠️ **Este conteúdo é destinado apenas para fins educacionais. Os dados exibidos são
    > ilustrativos e podem não corresponder a situações reais.**
    """)

    with gr.Row():
        with gr.Column():
            profissao    = gr.Dropdown(choices=PROFISSOES,    label="Profissão",    value="Engenheiro")
            escolaridade = gr.Dropdown(choices=ESCOLARIDADES, label="Escolaridade", value="Superior")
            regiao       = gr.Dropdown(choices=REGIOES,       label="Região",       value="Sudeste")

        with gr.Column():
            idade    = gr.Slider(minimum=14, maximum=80, value=35, step=1, label="Idade")
            anos_exp = gr.Slider(minimum=0,  maximum=60, value=10, step=1, label="Anos de Experiência")
            seg_ling = gr.Dropdown(choices=LINGUAS, label="Segunda Língua", value="Inglês")
            ter_ling = gr.Dropdown(choices=LINGUAS, label="Terceira Língua", value="Nenhuma")

    btn = gr.Button("💡 Calcular Salário", variant="primary")
    saida = gr.Markdown()

    btn.click(
        fn=predizer,
        inputs=[profissao, escolaridade, regiao, idade, anos_exp, seg_ling, ter_ling],
        outputs=saida,
    )

    gr.Examples(
        examples=[
            ["Médico",      "Doutorado",   "Sudeste",      42, 15, "Inglês",  "Nenhuma"],
            ["Vendedor",    "Médio",        "Nordeste",     28,  4, "Nenhuma", "Nenhuma"],
            ["Engenheiro",  "Superior",     "Sul",          35, 10, "Inglês",  "Espanhol"],
            ["Programador", "Superior",     "Sudeste",      30,  7, "Inglês",  "Nenhuma"],
        ],
        inputs=[profissao, escolaridade, regiao, idade, anos_exp, seg_ling, ter_ling],
    )

if __name__ == "__main__":
    demo.launch()
