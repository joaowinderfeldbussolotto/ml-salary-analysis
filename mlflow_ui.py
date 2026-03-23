"""
mlflow_ui.py
============
Abre o MLflow UI apontando para o mlruns correto, independente
de qual pasta você estiver quando rodar.

Uso:
    python mlflow_ui.py

Equivale a:
    mlflow ui --backend-store-uri /caminho/absoluto/para/mlruns
"""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
MLRUNS_PATH  = PROJECT_ROOT / "mlruns"

if not MLRUNS_PATH.exists():
    print(f"ERRO: pasta '{MLRUNS_PATH}' não encontrada.")
    print("Execute primeiro: python src/pipeline/train.py")
    sys.exit(1)

print(f"Abrindo MLflow UI...")
print(f"Backend: {MLRUNS_PATH}")
print(f"Acesse : http://localhost:5000\n")

subprocess.run([
    sys.executable, "-m", "mlflow", "ui",
    "--backend-store-uri", str(MLRUNS_PATH),
    "--host", "0.0.0.0",
    "--port", "5000",
])
