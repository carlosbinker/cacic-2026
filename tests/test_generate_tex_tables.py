"""Tests del generador de fragmentos .tex de tablas (F8 del índice)."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from generate_tex_tables import (  # noqa: E402
    ARCHIVOS_TABLAS,
    _escapar,
    _MOTIVO_EXPLICACION_PAPER,
    _VERSION_RESUELTA,
    tabla1_modelos,
    tabla2_resultados,
    tabla3_por_categoria,
    tabla4_taxonomia,
)
from models_2026 import (  # noqa: E402
    BASELINE_TRANSFORMERS,
    MODELOS_2026,
    TRANSFORMERS_5X,
    roster_activo,
)
from taxonomia_2026 import (  # noqa: E402
    CATEGORIAS_DISPLAY,
    ETIQUETAS_ERROR,
    ETIQUETAS_ERROR_DISPLAY,
)


def test_escapa_los_caracteres_especiales_de_latex():
    assert _escapar("a_b") == r"a\_b"
    assert _escapar("100%") == r"100\%"
    assert _escapar("a&b") == r"a\&b"
    assert _escapar("x#1") == r"x\#1"
    assert _escapar("${}") == r"\$\{\}"


def test_escapa_los_guiones_bajos_de_los_nombres_de_etiqueta():
    assert _escapar("confusion_intencion") == r"confusion\_intencion"


def test_escapar_no_rompe_texto_limpio():
    assert _escapar("granite-4.0-h-1b") == "granite-4.0-h-1b"


def test_los_cuatro_archivos_estan_declarados():
    # Bajó de 5 a 4 (Delta 2026-08-05): la Tabla 5/`tab:versiones` se fusionó
    # como columna de la Tabla 1 (versión de `transformers` + motivos de pin
    # en el caption) -- ningún dato se perdió, sólo el contenedor `table`.
    assert sorted(ARCHIVOS_TABLAS) == [
        "tabla1_modelos.tex", "tabla2_resultados_globales.tex",
        "tabla3_por_categoria.tex", "tabla4_taxonomia.tex",
    ]


def _resumen():
    return [{
        "modelo": m.nombre, "hf_repo_id": m.hf_repo_id, "params_b": m.params_b,
        "tier": m.tier, "modo_prompting": "chat_template",
        "transformers_pin": m.transformers_pin, "n": 32, "json_valido_pct": 100.0,
        "exact_match_pct": 40.0 + i, "exact_match_laxo_pct": 45.0 + i,
        "avg_latencia_s": 10.0 + i, "acc_intent_pct": 80.0,
        "acc_dispositivo_pct": 70.0, "acc_ubicacion_pct": 75.0,
        "acc_valor_pct": 85.0, "acc_unidad_pct": 90.0,
    } for i, m in enumerate(roster_activo())]


def _es_bloque_table(tex: str) -> bool:
    return (tex.startswith(r"\begin{table}") and tex.rstrip().endswith(r"\end{table}")
            and r"\documentclass" not in tex and tex.count(r"\begin{tabular}") == 1)


def test_tabla1_lista_los_doce_modelos_del_roster_activo_con_su_label():
    tex = tabla1_modelos()
    assert _es_bloque_table(tex)
    assert r"\label{tab:modelos}" in tex
    for m in roster_activo():
        assert _escapar(m.nombre) in tex
    assert tex.count(r"\\") >= 12


def test_tabla1_no_lista_los_modelos_excluidos():
    tex = tabla1_modelos()
    excluidos = [m for m in MODELOS_2026 if not m.activo]
    assert len(excluidos) == len(MODELOS_2026) - len(roster_activo())
    for m in excluidos:
        assert _escapar(m.nombre) not in tex


def test_tabla1_incluye_version_de_transformers_y_motivos_de_pin():
    """Re-base (Delta 2026-08-05) de lo que antes exigía
    `test_tabla5_marca_cuales_pines_son_necesarios` sobre la extinta Tabla 5
    (`tab:versiones`): la versión resuelta de `transformers` ahora es una
    columna de la Tabla 1 y los motivos de pin son una cláusula del caption.
    Mismo rigor sustantivo que el contrato anterior: los 12 nombres del
    roster (ya cubierto arriba), los dos valores de versión resuelta y los 4
    motivos reales de pin, cada uno junto a su modelo."""
    tex = tabla1_modelos()
    assert _escapar(_VERSION_RESUELTA[BASELINE_TRANSFORMERS]) in tex
    assert _escapar(_VERSION_RESUELTA[TRANSFORMERS_5X]) in tex
    for nombre, motivo in _MOTIVO_EXPLICACION_PAPER.items():
        assert _escapar(nombre) in tex
        assert _escapar(motivo) in tex


def test_tabla2_trae_estricta_y_laxa():
    tex = tabla2_resultados(_resumen())
    assert _es_bloque_table(tex)
    assert r"\label{tab:globales}" in tex
    assert "Estricta" in tex and "Laxa" in tex
    assert "40.0" in tex and "45.0" in tex


def test_tabla2_tiene_exactamente_cinco_columnas_sin_continuidad_2025():
    # La ronda 2025 fue un borrador de este mismo paper, nunca publicado, y sus
    # cifras se produjeron con el entorno sin fijar: no hay comparación válida
    # que hacer contra ellas, así que la tabla no lleva columnas de
    # continuidad (ver Delta 04, item 6).
    tex = tabla2_resultados(_resumen())
    encabezado = next(l for l in tex.splitlines() if "Modelo" in l)
    assert "2025" not in tex
    assert encabezado.count("&") == 4  # 5 columnas: Modelo, JSON válido, Estricta, Laxa, Latencia


def test_tabla2_no_incluye_al_modelo_excluido_del_roster_2026():
    tex = tabla2_resultados(_resumen())
    assert "Qwen2.5-0.5B-Instruct" not in tex
    assert _escapar("Qwen2.5-0.5B-Instruct") not in tex


def test_tabla3_una_fila_por_categoria():
    df = pd.DataFrame([
        {"categoria": "encendido_apagado_simple", "n": 8,
         roster_activo()[0].nombre: 50.0, roster_activo()[1].nombre: 62.5},
        {"categoria": "consulta_de_estado", "n": 3,
         roster_activo()[0].nombre: 33.3, roster_activo()[1].nombre: 66.7},
    ])
    tex = tabla3_por_categoria(df)
    assert _es_bloque_table(tex)
    assert r"\label{tab:categorias}" in tex
    assert CATEGORIAS_DISPLAY["encendido_apagado_simple"] in tex
    assert "(8)" in tex and "(3)" in tex          # el n va con la categoría
    assert "62.5" in tex


def test_tabla4_tiene_las_siete_etiquetas_con_nombre_legible():
    """Con las siete etiquetas instanciadas, las siete llevan columna."""
    df = pd.DataFrame([
        {"modelo": roster_activo()[0].nombre, "total_incorrectas": 10,
         **{e: i + 1 for i, e in enumerate(ETIQUETAS_ERROR)}},
    ])
    tex = tabla4_taxonomia(df)
    assert _es_bloque_table(tex)
    assert r"\label{tab:taxonomia}" in tex
    for e in ETIQUETAS_ERROR:
        # Case-insensitive: el nombre completo vive en el caption como
        # leyenda de siglas, con minúscula inicial (tipografía castellana
        # correcta tras los dos puntos); el test garantiza la presencia del
        # nombre, no su capitalización.
        assert ETIQUETAS_ERROR_DISPLAY[e].lower() in tex.lower()
    assert "alucinación de valor/unidad" in tex.lower()
    assert "valor numérico incorrecto" in tex.lower()


def test_tabla4_omite_la_etiqueta_que_quedo_en_cero_pero_no_la_de_una_ocurrencia():
    """El paper sólo muestra lo que llega a los resultados: una etiqueta que el
    juez pudo usar y nunca usó no gasta una columna entera de ceros en una
    tabla apretada (el hecho se reporta en prosa, §4.3). Una etiqueta con una
    sola ocurrencia sí lleva columna: es una medición, no un vacío."""
    nunca, una_vez = ETIQUETAS_ERROR[0], ETIQUETAS_ERROR[1]
    conteos = {e: 2 for e in ETIQUETAS_ERROR}
    conteos[nunca] = 0
    conteos[una_vez] = 1
    df = pd.DataFrame([
        {"modelo": m.nombre, "total_incorrectas": 10, **conteos}
        for m in roster_activo()[:2]
    ])
    df.loc[:, nunca] = 0
    tex = tabla4_taxonomia(df)
    assert ETIQUETAS_ERROR_DISPLAY[nunca] not in tex
    # Case-insensitive por el mismo motivo que en el test anterior: el
    # nombre completo aparece en el caption con minúscula inicial.
    assert ETIQUETAS_ERROR_DISPLAY[una_vez].lower() in tex.lower()
    encabezado = next(l for l in tex.splitlines() if "Modelo" in l)
    # Modelo + Incorrectas + 6 etiquetas con al menos una ocurrencia
    assert encabezado.count("&") == 1 + len(ETIQUETAS_ERROR) - 1


def test_ninguna_tabla_deja_guiones_bajos_sin_escapar():
    df_tax = pd.DataFrame([{"modelo": roster_activo()[0].nombre, "total_incorrectas": 1,
                            **{e: 0 for e in ETIQUETAS_ERROR}}])
    df_cat = pd.DataFrame([{"categoria": "encendido_apagado_simple", "n": 8,
                            roster_activo()[0].nombre: 50.0}])
    for tex in (tabla1_modelos(), tabla2_resultados(_resumen()),
                tabla3_por_categoria(df_cat), tabla4_taxonomia(df_tax)):
        for i, ch in enumerate(tex):
            if ch == "_":
                assert tex[i - 1] == "\\", f"guion bajo sin escapar cerca de: {tex[i-30:i+10]!r}"
