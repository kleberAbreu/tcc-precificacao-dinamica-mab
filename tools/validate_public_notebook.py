from pathlib import Path
import json
import shutil
import subprocess

import joblib
import numpy as np
import pandas as pd


ROOT = Path("/Users/kleberabreu/Desktop/tcc/tcc-copy-note")
DADOS = ROOT / "Dados"
NB_PATH = DADOS / "04_Reproducao_Publica_TCC.ipynb"
EXECUTED_PATH = DADOS / "04_Reproducao_Publica_TCC.executed.ipynb"
REFERENCE_DIR = DADOS / "_validation_reference_public"
REFERENCE_DIR.mkdir(exist_ok=True)


REFERENCE_FILES = [
    "base_final_para_modelagem.csv",
    "modelo_ocupacao_final.pkl",
]


EXPECTED_FIGURES = [
    "Histogramas Antes e depois.png",
    "correlacao_subnotas.png",
    "diagnostico_modelo_reduzido.png",
    "curva_demanda_estimada.png",
    "simulacao_mab_receita_estacionario.png",
    "simulacao_mab_bracos_estacionario.png",
    "simulacao_mab_receita_multi_estacao.png",
    "heatmap_thompson_multi_estacao.png",
]


def backup_references():
    for filename in REFERENCE_FILES:
        shutil.copy2(DADOS / filename, REFERENCE_DIR / filename)


def execute_notebook():
    subprocess.run(
        [
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            str(NB_PATH),
            "--output",
            EXECUTED_PATH.name,
        ],
        cwd=DADOS,
        check=True,
    )


def compare_base():
    new = pd.read_csv(DADOS / "base_final_para_modelagem.csv")
    ref = pd.read_csv(REFERENCE_DIR / "base_final_para_modelagem.csv")
    if list(new.columns) != list(ref.columns):
        raise AssertionError("Colunas da base final divergiram.")
    if new.shape != ref.shape:
        raise AssertionError("Shape da base final divergiu.")
    numeric_cols = new.select_dtypes(include=[np.number]).columns
    if not np.allclose(new[numeric_cols], ref[numeric_cols], equal_nan=True):
        raise AssertionError("Valores numéricos da base final divergiram.")
    object_cols = [c for c in new.columns if c not in numeric_cols]
    if object_cols and not new[object_cols].equals(ref[object_cols]):
        raise AssertionError("Valores não numéricos da base final divergiram.")


def compare_model():
    new = joblib.load(DADOS / "modelo_ocupacao_final.pkl")
    ref = joblib.load(REFERENCE_DIR / "modelo_ocupacao_final.pkl")
    if new["features"] != ref["features"]:
        raise AssertionError("Features do modelo final divergiram.")
    if new["target"] != ref["target"] or new["tipo"] != ref["tipo"]:
        raise AssertionError("Metadados do modelo final divergiram.")
    if not np.isclose(new["r2"], ref["r2"]):
        raise AssertionError("R² do modelo final divergiu.")
    if not np.allclose(
        new["model"].params.sort_index().values,
        ref["model"].params.sort_index().values,
    ):
        raise AssertionError("Coeficientes do modelo final divergiram.")


def validate_summary():
    summary_path = DADOS / "resultados_reproducao_publica.json"
    if not summary_path.exists():
        raise AssertionError("Arquivo resultados_reproducao_publica.json não foi gerado.")
    summary = json.loads(summary_path.read_text())

    if summary["amostra"]["n_imoveis"] != 571:
        raise AssertionError("Número de imóveis divergente no resumo.")
    if summary["amostra"]["n_observacoes"] != 2284:
        raise AssertionError("Número de observações divergente no resumo.")
    if not np.isclose(summary["modelo_final"]["r2"], 0.0998, atol=5e-4):
        raise AssertionError("R² do resumo divergiu do esperado.")

    expected_stationary = {
        "ε-Greedy": 37223.7,
        "UCB": 40304.1,
        "Thompson Sampling": 35321.1,
    }
    expected_multi = {
        "ε-Greedy": 27838.7,
        "UCB": 29671.3,
        "Thompson Sampling": 27159.1,
    }

    for key, value in expected_stationary.items():
        if not np.isclose(summary["cenario_estacionario"][key], value, atol=1e-6):
            raise AssertionError(f"Resumo do cenário estacionário divergiu para {key}.")
    for key, value in expected_multi.items():
        if not np.isclose(summary["cenario_multi_estacao"][key], value, atol=1e-6):
            raise AssertionError(f"Resumo do cenário multi-estação divergiu para {key}.")


def validate_figures():
    for figure in EXPECTED_FIGURES:
        path = DADOS / figure
        if not path.exists():
            raise AssertionError(f"Figura não encontrada: {figure}")


def main():
    backup_references()
    execute_notebook()
    compare_base()
    compare_model()
    validate_summary()
    validate_figures()
    print("VALIDACAO_PUBLICA_OK")


if __name__ == "__main__":
    main()
