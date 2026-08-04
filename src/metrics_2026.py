#!/usr/bin/env python3
"""
Agregación de métricas 2026.

Módulo nuevo, no una edición de metrics.py: el legacy está congelado porque
tests/test_metricas.py fija contra él los números publicados del paper
original. Dos diferencias de fondo con aquel: acá hay dos métricas de
exactitud (estricta y laxa) en vez de una, y la taxonomía tiene 7 etiquetas
con alucinacion_valor_unidad y valor_numerico_incorrecto separadas, como en
la Tabla 4 publicada (el legacy las fusiona).

Uso:
    python src/metrics_2026.py
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from models_2026 import roster_activo
from taxonomia_2026 import (
    CATEGORIAS_LINGUISTICAS,
    ETIQUETAS_EQUIVALENTES,
    ETIQUETAS_ERROR,
    SEP_ETIQUETAS,
)

RAIZ = Path(__file__).resolve().parent.parent
DIR_2026 = RAIZ / "data" / "2026"
PATH_DETALLE = DIR_2026 / "detalle_2026.csv"
PATH_ETIQUETAS = DIR_2026 / "etiquetas_errores.csv"
PATH_CATEGORIAS = DIR_2026 / "categorias_comandos.csv"
PATH_RESUMEN = DIR_2026 / "resumen_2026.json"
PATH_TAXONOMIA = DIR_2026 / "taxonomia_2026.csv"
PATH_POR_CATEGORIA = DIR_2026 / "exactitud_por_categoria.csv"

CLAVES_RESUMEN = [
    "modelo", "hf_repo_id", "params_b", "tier", "modo_prompting",
    "transformers_pin", "n", "json_valido_pct", "exact_match_pct",
    "exact_match_laxo_pct", "avg_latencia_s", "acc_intent_pct",
    "acc_dispositivo_pct", "acc_ubicacion_pct", "acc_valor_pct",
    "acc_unidad_pct",
]


def _como_bool(serie: pd.Series) -> pd.Series:
    """Booleano robusto para columnas leidas de CSV.

    pandas infiere dtype bool cuando el CSV trae los literales True/False sin
    comillas, pero eso no esta garantizado (comillas, NaN mezclados, u otra
    fuente que serialice como texto bajan la columna a dtype object). Castear
    ese texto con `.astype(bool)` es un bug: `bool("False")` es `True` porque
    cualquier string no vacio es verdadero en Python, y eso empuja el
    porcentaje al 100% sin importar el contenido real.
    """
    if pd.api.types.is_bool_dtype(serie):
        return serie
    return serie.astype(str).str.strip().str.lower().map(
        {"true": True, "false": False}
    ).astype(bool)


def _mapa_equivalentes(df_etiquetas: pd.DataFrame) -> set[tuple[str, int]]:
    """(modelo, idx) que el juez consideró semánticamente equivalentes."""
    equivalentes = set()
    for _, fila in df_etiquetas.iterrows():
        partes = set(str(fila["etiquetas"]).split(SEP_ETIQUETAS))
        if partes & ETIQUETAS_EQUIVALENTES:
            equivalentes.add((fila["modelo"], int(fila["idx"])))
    return equivalentes


def calcular_resumen_2026(df_detalle: pd.DataFrame,
                          df_etiquetas: pd.DataFrame) -> list[dict]:
    """Resumen por modelo con exactitud estricta y laxa (RF10)."""
    equivalentes = _mapa_equivalentes(df_etiquetas)
    filas = []
    for modelo in roster_activo():
        df_m = df_detalle[df_detalle["modelo"] == modelo.nombre]
        if df_m.empty:
            continue
        estricta = _como_bool(df_m["match_exact"])
        laxa = estricta | df_m["idx"].map(
            lambda i: (modelo.nombre, int(i)) in equivalentes
        )
        modos = df_m["modo_prompting"].unique()
        filas.append({
            "modelo": modelo.nombre,
            "hf_repo_id": modelo.hf_repo_id,
            "params_b": modelo.params_b,
            "tier": modelo.tier,
            "modo_prompting": modos[0] if len(modos) == 1 else "mixto",
            "transformers_pin": modelo.transformers_pin,
            "n": len(df_m),
            "json_valido_pct": round(100 * _como_bool(df_m["json_valido"]).mean(), 1),
            "exact_match_pct": round(100 * estricta.mean(), 1),
            "exact_match_laxo_pct": round(100 * laxa.mean(), 1),
            "avg_latencia_s": round(df_m["latencia_s"].mean(), 3),
            "acc_intent_pct": round(100 * _como_bool(df_m["match_intent"]).mean(), 1),
            "acc_dispositivo_pct": round(100 * _como_bool(df_m["match_dispositivo"]).mean(), 1),
            "acc_ubicacion_pct": round(100 * _como_bool(df_m["match_ubicacion"]).mean(), 1),
            "acc_valor_pct": round(100 * _como_bool(df_m["match_valor"]).mean(), 1),
            "acc_unidad_pct": round(100 * _como_bool(df_m["match_unidad"]).mean(), 1),
        })
    return [{c: f[c] for c in CLAVES_RESUMEN} for f in filas]


def calcular_taxonomia_errores_2026(df_detalle: pd.DataFrame,
                                    df_etiquetas: pd.DataFrame) -> pd.DataFrame:
    """Tabla 4 con las 7 etiquetas, no excluyentes (RF11)."""
    filas = []
    for modelo in roster_activo():
        df_m = df_detalle[df_detalle["modelo"] == modelo.nombre]
        if df_m.empty:
            continue
        incorrectas = df_m[~_como_bool(df_m["match_exact"])]
        eti_m = df_etiquetas[df_etiquetas["modelo"] == modelo.nombre]
        conteos = {e: 0 for e in ETIQUETAS_ERROR}
        for valor in eti_m["etiquetas"]:
            for etiqueta in str(valor).split(SEP_ETIQUETAS):
                if etiqueta in conteos:
                    conteos[etiqueta] += 1
        filas.append({
            "modelo": modelo.nombre,
            "total_incorrectas": len(incorrectas),
            **conteos,
        })
    return pd.DataFrame(filas, columns=["modelo", "total_incorrectas"] + ETIQUETAS_ERROR)


def calcular_exactitud_por_categoria(df_detalle: pd.DataFrame,
                                     df_categorias: pd.DataFrame) -> pd.DataFrame:
    """Tabla 3: exactitud estricta por categoría lingüística x modelo (RF9)."""
    categoria_por_idx = dict(
        zip(df_categorias["idx"].astype(int), df_categorias["categoria"])
    )
    df = df_detalle.copy()
    df["categoria"] = df["idx"].astype(int).map(categoria_por_idx)

    presentes = [c for c in CATEGORIAS_LINGUISTICAS if c in set(df["categoria"])]
    modelos = [m.nombre for m in roster_activo() if m.nombre in set(df["modelo"])]

    filas = []
    for categoria in presentes:
        df_c = df[df["categoria"] == categoria]
        fila = {"categoria": categoria, "n": int(df_c["idx"].nunique())}
        for nombre in modelos:
            df_cm = df_c[df_c["modelo"] == nombre]
            fila[nombre] = (
                round(100 * _como_bool(df_cm["match_exact"]).mean(), 1)
                if not df_cm.empty else 0.0
            )
        filas.append(fila)
    return pd.DataFrame(filas, columns=["categoria", "n"] + modelos)


def main() -> int:
    parser = argparse.ArgumentParser(description="Métricas 2026")
    parser.add_argument("--detalle", default=str(PATH_DETALLE))
    parser.add_argument("--etiquetas", default=str(PATH_ETIQUETAS))
    parser.add_argument("--categorias", default=str(PATH_CATEGORIAS))
    args = parser.parse_args()

    det = pd.read_csv(args.detalle)
    eti = pd.read_csv(args.etiquetas)
    cat = pd.read_csv(args.categorias)

    resumen = calcular_resumen_2026(det, eti)
    PATH_RESUMEN.parent.mkdir(parents=True, exist_ok=True)
    PATH_RESUMEN.write_text(
        json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    calcular_taxonomia_errores_2026(det, eti).to_csv(PATH_TAXONOMIA, index=False)
    calcular_exactitud_por_categoria(det, cat).to_csv(PATH_POR_CATEGORIA, index=False)

    print(f"{'modelo':28s} {'estricta':>9s} {'laxa':>7s} {'brecha':>7s} {'lat(s)':>8s}")
    for f in resumen:
        brecha = round(f["exact_match_laxo_pct"] - f["exact_match_pct"], 1)
        print(f"{f['modelo']:28s} {f['exact_match_pct']:8.1f}% "
              f"{f['exact_match_laxo_pct']:6.1f}% {brecha:6.1f}pp "
              f"{f['avg_latencia_s']:8.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
