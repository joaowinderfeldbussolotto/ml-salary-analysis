# %% [markdown]
# # 📊 Análise de Resultados dos Modelos
# ## UC Aprendizado de Máquina 2026/1
#
# Este notebook analisa e visualiza os resultados do treinamento de 8 modelos
# candidatos para predição de salários. Todos os experimentos foram rastreados
# via MLflow e os resultados estão em `results/model_comparison.csv`.
#
# ---

# %% [markdown]
# ## 1. Carregamento dos Resultados

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import json
from pathlib import Path

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams["figure.dpi"] = 110
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["axes.titleweight"] = "bold"

RESULTS_PATH = Path("../results/model_comparison.csv")
META_PATH    = Path("../models/best_model_meta.json")

df_res = pd.read_csv(RESULTS_PATH)

with open(META_PATH, encoding="utf-8") as f:
    meta = json.load(f)

print("Modelos avaliados:", len(df_res))
df_res[["nome","rmse","mae","r2","mape","tempo"]].sort_values("rmse")

# %% [markdown]
# ## 2. Comparação Visual dos Modelos

# %%
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Comparação de Modelos — Dataset Salários Brasil", fontsize=14, fontweight="bold")

df_sorted_rmse = df_res.sort_values("rmse")
cores = sns.color_palette("RdYlGn_r", len(df_res))
cores_inv = sns.color_palette("RdYlGn", len(df_res))

# RMSE
axes[0,0].barh(df_sorted_rmse["nome"], df_sorted_rmse["rmse"],
               color=cores, edgecolor="white")
axes[0,0].set_title("RMSE (R$) — menor é melhor")
axes[0,0].set_xlabel("RMSE (R$)")
axes[0,0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,p: f"R${x/1000:.0f}k"))
axes[0,0].axvline(df_sorted_rmse["rmse"].iloc[0], color="red", linestyle="--", alpha=0.5)

# MAE
df_sorted_mae = df_res.sort_values("mae")
axes[0,1].barh(df_sorted_mae["nome"], df_sorted_mae["mae"],
               color=sns.color_palette("RdYlGn_r", len(df_res)), edgecolor="white")
axes[0,1].set_title("MAE (R$) — menor é melhor")
axes[0,1].set_xlabel("MAE (R$)")
axes[0,1].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,p: f"R${x/1000:.1f}k"))

# R²
df_sorted_r2 = df_res.sort_values("r2")
axes[1,0].barh(df_sorted_r2["nome"], df_sorted_r2["r2"],
               color=sns.color_palette("RdYlGn", len(df_res)), edgecolor="white")
axes[1,0].set_title("R² — maior é melhor")
axes[1,0].set_xlabel("R²")
axes[1,0].axvline(0, color="red", linestyle="--", linewidth=1.5)

# MAPE
df_sorted_mape = df_res.sort_values("mape")
axes[1,1].barh(df_sorted_mape["nome"], df_sorted_mape["mape"],
               color=sns.color_palette("RdYlGn_r", len(df_res)), edgecolor="white")
axes[1,1].set_title("MAPE (%) — menor é melhor")
axes[1,1].set_xlabel("MAPE (%)")

plt.tight_layout()
plt.savefig("../results/model_comparison_chart.png", dpi=130, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 3. Análise de Overfitting

# %%
fig, ax = plt.subplots(figsize=(10, 5))

x = np.arange(len(df_res))
w = 0.35
nomes = df_res.sort_values("rmse")["nome"].values
rmse_teste  = df_res.sort_values("rmse")["rmse"].values
rmse_treino = df_res.sort_values("rmse")["rmse_treino"].values

bars1 = ax.bar(x - w/2, rmse_treino, w, label="RMSE Treino", color=sns.color_palette("muted")[0], edgecolor="white")
bars2 = ax.bar(x + w/2, rmse_teste,  w, label="RMSE Teste",  color=sns.color_palette("muted")[3], edgecolor="white")

ax.set_xticks(x)
ax.set_xticklabels(nomes, rotation=25, ha="right")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,p: f"R${x/1000:.0f}k"))
ax.set_title("RMSE Treino vs Teste — Checagem de Overfitting")
ax.set_ylabel("RMSE (R$)")
ax.legend()

# Anotar diferença %
for i, (tr, te) in enumerate(zip(rmse_treino, rmse_teste)):
    diff_pct = (te - tr) / tr * 100
    ax.text(i, max(tr, te) + 200, f"+{diff_pct:.0f}%",
            ha="center", fontsize=8, color="gray")

# Destaque do KNN (overfitting extremo)
knn_idx = list(nomes).index("KNN") if "KNN" in nomes else -1
if knn_idx >= 0:
    ax.annotate("KNN: overfitting\nextremo (R²≈1 no treino)",
                xy=(knn_idx + w/2, rmse_teste[knn_idx]),
                xytext=(knn_idx + 1.5, rmse_teste[knn_idx] + 1000),
                arrowprops=dict(arrowstyle="->", color="red"),
                fontsize=9, color="red")

plt.tight_layout()
plt.savefig("../results/overfitting_analysis.png", dpi=130, bbox_inches="tight")
plt.show()

print("→ KNN tem R²=0.9999 no treino e apenas 0.13 no teste — overfitting severo")
print("→ Ridge: gap treino/teste pequeno — boa generalização")

# %% [markdown]
# ## 4. Análise de Predições do Melhor Modelo

# %%
import sys
sys.path.insert(0, "../src")
import joblib

model = joblib.load("../models/best_model.joblib")

# Reconstruir dados de teste
from pipeline.preprocessing import carregar_dados, limpar_dados, engenharia_features, criar_target_log
from sklearn.model_selection import train_test_split

df_raw   = carregar_dados("../data/dataset_salarios_brasil.csv")
df_clean = limpar_dados(df_raw)
df_feat  = engenharia_features(df_clean)
target   = criar_target_log(df_feat)

_, df_test, _, y_test = train_test_split(
    df_feat, target, test_size=0.2, random_state=42, stratify=df_feat["Profissao"]
)

y_pred_log = model.predict(df_test)
y_true = np.expm1(y_test.values)
y_pred = np.clip(np.expm1(y_pred_log), 0, None)
residuos = y_true - y_pred

# %%
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle(f"Análise de Predições — {meta['nome']}", fontsize=13, fontweight="bold")

# Scatter: Real vs Predito
lim = max(y_true.max(), y_pred.max())
axes[0].scatter(y_true, y_pred, alpha=0.3, s=10, color=sns.color_palette("muted")[0])
axes[0].plot([0, lim], [0, lim], "r--", linewidth=1.5, label="Perfeito")
axes[0].set_xlabel("Salário Real (R$)")
axes[0].set_ylabel("Salário Predito (R$)")
axes[0].set_title("Real vs Predito")
axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,p: f"R${x/1000:.0f}k"))
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,p: f"R${x/1000:.0f}k"))
axes[0].legend()

# Distribuição de resíduos
axes[1].hist(residuos, bins=60, color=sns.color_palette("muted")[2], edgecolor="white")
axes[1].axvline(0, color="red", linestyle="--", linewidth=1.5)
axes[1].set_title("Distribuição dos Resíduos")
axes[1].set_xlabel("Resíduo (R$ real – R$ predito)")
axes[1].set_ylabel("Frequência")
axes[1].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,p: f"R${x/1000:.0f}k"))

# Resíduos vs Predito (homocedasticidade)
axes[2].scatter(y_pred, residuos, alpha=0.3, s=10, color=sns.color_palette("muted")[4])
axes[2].axhline(0, color="red", linestyle="--", linewidth=1.5)
axes[2].set_xlabel("Salário Predito (R$)")
axes[2].set_ylabel("Resíduo (R$)")
axes[2].set_title("Resíduos vs Predito")
axes[2].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,p: f"R${x/1000:.0f}k"))
axes[2].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,p: f"R${x/1000:.0f}k"))

plt.tight_layout()
plt.savefig("../results/prediction_analysis.png", dpi=130, bbox_inches="tight")
plt.show()

print(f"Resíduo médio  : R$ {residuos.mean():,.0f} (bias)")
print(f"Resíduo mediano: R$ {np.median(residuos):,.0f}")
print(f"% dentro ± R$5k: {(np.abs(residuos) < 5000).mean()*100:.1f}%")

# %% [markdown]
# ## 5. Erro por Profissão e Região

# %%
df_analise = df_test.copy()
df_analise["y_true"] = y_true
df_analise["y_pred"] = y_pred
df_analise["erro_abs"] = np.abs(y_true - y_pred)

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Erro por Profissão
mae_prof = df_analise.groupby("Profissao")["erro_abs"].mean().sort_values()
axes[0].barh(mae_prof.index, mae_prof.values,
             color=sns.color_palette("coolwarm_r", len(mae_prof)), edgecolor="white")
axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,p: f"R${x/1000:.0f}k"))
axes[0].set_title("MAE por Profissão")
axes[0].set_xlabel("MAE Médio (R$)")

# Erro por Região
mae_reg = df_analise.groupby("Regiao")["erro_abs"].mean().sort_values()
axes[1].bar(mae_reg.index, mae_reg.values,
            color=sns.color_palette("Set2"), edgecolor="white")
axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,p: f"R${x/1000:.0f}k"))
axes[1].set_title("MAE por Região")
axes[1].set_ylabel("MAE Médio (R$)")

plt.tight_layout()
plt.savefig("../results/error_by_group.png", dpi=130, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 6. Resumo Executivo

# %%
print("="*60)
print("  RESUMO EXECUTIVO — MODELAGEM DE SALÁRIOS")
print("="*60)
print()
print(f"  Melhor modelo    : {meta['nome']}")
print(f"  RMSE (teste)     : R$ {meta['rmse_teste']:,.2f}")
print(f"  MAE  (teste)     : R$ {meta['mae_teste']:,.2f}")
print(f"  R²   (teste)     : {meta['r2_teste']:.4f}")
print(f"  MAPE (teste)     : {meta['mape_teste']:.1f}%")
print()
print("  Interpretação do R²=0.21:")
print("  O modelo explica ~21% da variância salarial observada.")
print("  Isso é ESPERADO — salário depende fortemente de fatores")
print("  não capturados: empresa, negociação individual, benefícios,")
print("  localização específica dentro da região, senioridade.")
print()
print("  O MAE de R$ 1.879 significa que, em média, a predição")
print("  erra ~R$ 1.9k — aceitável como estimativa orientativa.")
print()
print("  Para melhorar o R²:")
print("  → Adicionar variáveis: porte da empresa, cidade, área de atuação")
print("  → Coletar mais dados (dataset atual: 14.4k registros)")
print("  → Feature interactions: Profissão × Escolaridade, Região × Profissão")
