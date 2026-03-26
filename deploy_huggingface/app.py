from pathlib import Path

import gradio as gr
import joblib
import numpy as np
import pandas as pd

# Paths expected by Hugging Face Space
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_model.joblib"

model = joblib.load(MODEL_PATH)

ESCOLARIDADES = ["Fundamental", "Médio", "Técnico", "Superior", "Pós", "Mestrado", "Doutorado"]
REGIOES = ["Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"]
PROFISSOES = sorted([
    "Administrador", "Advogado", "Agrônomo", "Analista de Dados", "Arquiteto",
        "Cientista", "Consultor", "Designer", "Economista", "Enfermeiro",
            "Engenheiro", "Motorista", "Médico", "Pesquisador", "Policial",
                "Professor", "Programador", "Técnico", "Vendedor",
                ])
                LINGUAS = [
                    "Nenhuma", "Inglês", "Espanhol", "Francês", "Alemão", "Italiano",
                        "Mandarim", "Japonês", "Português",
                        ]


                        def format_brl(value: float) -> str:
                            # Keep formatting stable in containers where pt_BR locale may not be installed.
                                return f"R$ {value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


                                def predizer(profissao, escolaridade, regiao, idade, anos_exp, segunda_lingua, terceira_lingua):
                                    df = pd.DataFrame(
                                            {
                                                        "Idade": [float(idade)],
                                                                    "Anos_Experiencia": [float(anos_exp)],
                                                                                "Escolaridade": [escolaridade],
                                                                                            "Regiao": [regiao],
                                                                                                        "Profissao": [profissao],
                                                                                                                    "Segunda_Lingua": [segunda_lingua],
                                                                                                                                "Terceira_Lingua": [terceira_lingua],
                                                                                                                                            "tem_segunda_lingua": [int(segunda_lingua != "Nenhuma")],
                                                                                                                                                        "tem_terceira_lingua": [int(terceira_lingua != "Nenhuma")],
                                                                                                                                                                    "total_linguas": [
                                                                                                                                                                                    int(segunda_lingua != "Nenhuma") + int(terceira_lingua != "Nenhuma")
                                                                                                                                                                                                ],
                                                                                                                                                                                                            "dif_idade_exp": [float(idade) - float(anos_exp)],
                                                                                                                                                                                                                    }
                                                                                                                                                                                                                        )

                                                                                                                                                                                                                            log_pred = model.predict(df)[0]
                                                                                                                                                                                                                                salario = max(float(np.expm1(log_pred)), 0.0)
                                                                                                                                                                                                                                    margem = salario * 0.15

                                                                                                                                                                                                                                        return (
                                                                                                                                                                                                                                                "## Salário Estimado\n\n"
                                                                                                                                                                                                                                                        f"**{format_brl(salario)}**\n\n"
                                                                                                                                                                                                                                                                f"Intervalo estimado: {format_brl(max(0, salario - margem))} - {format_brl(salario + margem)}\n\n"
                                                                                                                                                                                                                                                                        "---\n\n"
                                                                                                                                                                                                                                                                                "Este app e os valores exibidos são para fins educacionais."
                                                                                                                                                                                                                                                                                    )


                                                                                                                                                                                                                                                                                    with gr.Blocks(title="Predição de Salários - Brasil", theme=gr.themes.Soft()) as demo:
                                                                                                                                                                                                                                                                                        gr.Markdown(
                                                                                                                                                                                                                                                                                                """
                                                                                                                                                                                                                                                                                                # Predição de Salários - Brasil
                                                                                                                                                                                                                                                                                                Preencha o perfil profissional e clique em **Calcular** para estimar o salário.
                                                                                                                                                                                                                                                                                                """
                                                                                                                                                                                                                                                                                                    )

                                                                                                                                                                                                                                                                                                        with gr.Row():
                                                                                                                                                                                                                                                                                                                with gr.Column():
                                                                                                                                                                                                                                                                                                                            profissao = gr.Dropdown(choices=PROFISSOES, label="Profissão", value="Engenheiro")
                                                                                                                                                                                                                                                                                                                                        escolaridade = gr.Dropdown(choices=ESCOLARIDADES, label="Escolaridade", value="Superior")
                                                                                                                                                                                                                                                                                                                                                    regiao = gr.Dropdown(choices=REGIOES, label="Região", value="Sudeste")

                                                                                                                                                                                                                                                                                                                                                            with gr.Column():
                                                                                                                                                                                                                                                                                                                                                                        idade = gr.Slider(minimum=14, maximum=80, value=35, step=1, label="Idade")
                                                                                                                                                                                                                                                                                                                                                                                    anos_exp = gr.Slider(minimum=0, maximum=60, value=10, step=1, label="Anos de Experiência")
                                                                                                                                                                                                                                                                                                                                                                                                seg_ling = gr.Dropdown(choices=LINGUAS, label="Segunda Língua", value="Inglês")
                                                                                                                                                                                                                                                                                                                                                                                                            ter_ling = gr.Dropdown(choices=LINGUAS, label="Terceira Língua", value="Nenhuma")

                                                                                                                                                                                                                                                                                                                                                                                                                btn = gr.Button("Calcular Salário", variant="primary")
                                                                                                                                                                                                                                                                                                                                                                                                                    saida = gr.Markdown()

                                                                                                                                                                                                                                                                                                                                                                                                                        btn.click(
                                                                                                                                                                                                                                                                                                                                                                                                                                fn=predizer,
                                                                                                                                                                                                                                                                                                                                                                                                                                        inputs=[profissao, escolaridade, regiao, idade, anos_exp, seg_ling, ter_ling],
                                                                                                                                                                                                                                                                                                                                                                                                                                                outputs=saida,
                                                                                                                                                                                                                                                                                                                                                                                                                                                    )

                                                                                                                                                                                                                                                                                                                                                                                                                                                    if __name__ == "__main__":
                                                                                                                                                                                                                                                                                                                                                                                                                                                        demo.launch()
                                                                                                                                                                                                                                                                                                                                                                                                                                                        