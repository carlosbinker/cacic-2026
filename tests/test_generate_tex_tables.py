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
    tabla2_resultados_taxonomia,
    tabla3_por_categoria,
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


def test_los_tres_archivos_estan_declarados():
    # Bajó de 5 a 3 (Delta 2026-08-05): la Tabla 5/`tab:versiones` se fusionó
    # como columna de la Tabla 1 (versión de `transformers` + motivos de pin
    # en el caption) y la Tabla 4/`tab:taxonomia` se fusionó con la Tabla 2
    # (mismo índice por modelo, mismas 12 filas) -- ningún dato se perdió,
    # sólo el contenedor `table` de cada una.
    assert sorted(ARCHIVOS_TABLAS) == [
        "tabla1_modelos.tex", "tabla2_resultados_globales.tex",
        "tabla3_por_categoria.tex",
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


def _taxonomia_df(conteos_por_modelo: dict[str, dict] | None = None) -> pd.DataFrame:
    """Una fila por modelo del roster activo, `total_incorrectas` + las 7
    etiquetas en 0 por defecto -- alineada con `_resumen()` para que
    `tabla2_resultados_taxonomia` pueda unir ambas por nombre de modelo."""
    conteos_por_modelo = conteos_por_modelo or {}
    filas = []
    for m in roster_activo():
        fila = {"modelo": m.nombre, "total_incorrectas": 10,
                **{e: 0 for e in ETIQUETAS_ERROR}}
        fila.update(conteos_por_modelo.get(m.nombre, {}))
        filas.append(fila)
    return pd.DataFrame(filas)


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
    tex = tabla2_resultados_taxonomia(_resumen(), _taxonomia_df())
    assert _es_bloque_table(tex)
    assert r"\label{tab:resultados}" in tex
    assert "Estr." in tex and "Laxa" in tex
    assert "40.0" in tex and "45.0" in tex


def test_tabla2_tiene_cinco_columnas_de_metricas_globales_sin_continuidad_2025():
    # La ronda 2025 fue un borrador de este mismo paper, nunca publicado, y sus
    # cifras se produjeron con el entorno sin fijar: no hay comparación válida
    # que hacer contra ellas, así que la tabla no lleva columnas de
    # continuidad (ver Delta 04, item 6).
    tex = tabla2_resultados_taxonomia(_resumen(), _taxonomia_df())
    encabezado = next(l for l in tex.splitlines() if "Modelo" in l)
    assert "2025" not in tex
    # Modelo + JSON + Estr. + Laxa + Lat. + I + hasta 6 siglas de etiqueta.
    # Con `_taxonomia_df()` (todas las etiquetas en 0) ninguna etiqueta suma
    # >0, así que sólo quedan las 5 columnas de métricas + I.
    assert encabezado.count("&") == 5


def test_tabla2_no_incluye_al_modelo_excluido_del_roster_2026():
    tex = tabla2_resultados_taxonomia(_resumen(), _taxonomia_df())
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


def test_tabla2_tiene_las_siete_etiquetas_con_nombre_legible_en_el_caption():
    """Con las siete etiquetas instanciadas, las siete llevan columna y
    aparecen en la leyenda del caption (fusión de lo que antes cubría
    `test_tabla4_tiene_las_siete_etiquetas_con_nombre_legible`)."""
    conteos = {roster_activo()[0].nombre: {e: i + 1 for i, e in enumerate(ETIQUETAS_ERROR)}}
    tex = tabla2_resultados_taxonomia(_resumen(), _taxonomia_df(conteos))
    assert _es_bloque_table(tex)
    assert r"\label{tab:resultados}" in tex
    for e in ETIQUETAS_ERROR:
        # Case-insensitive: el nombre completo vive en el caption como
        # leyenda de siglas, con minúscula inicial (tipografía castellana
        # correcta tras los dos puntos); el test garantiza la presencia del
        # nombre, no su capitalización.
        assert ETIQUETAS_ERROR_DISPLAY[e].lower() in tex.lower()
    assert "alucinación de valor/unidad" in tex.lower()
    assert "valor numérico incorrecto" in tex.lower()


def test_tabla2_omite_la_etiqueta_que_quedo_en_cero_pero_no_la_de_una_ocurrencia():
    """El paper sólo muestra lo que llega a los resultados: una etiqueta que el
    juez pudo usar y nunca usó no gasta una columna entera de ceros en una
    tabla apretada (el hecho se reporta en prosa, §4.3). Una etiqueta con una
    sola ocurrencia sí lleva columna: es una medición, no un vacío."""
    nunca, una_vez = ETIQUETAS_ERROR[0], ETIQUETAS_ERROR[1]
    conteos = {e: 2 for e in ETIQUETAS_ERROR}
    conteos[nunca] = 0
    conteos[una_vez] = 1
    modelos_afectados = roster_activo()[:2]
    tex = tabla2_resultados_taxonomia(
        _resumen(), _taxonomia_df({m.nombre: conteos for m in modelos_afectados})
    )
    assert ETIQUETAS_ERROR_DISPLAY[nunca] not in tex
    # Case-insensitive por el mismo motivo que en el test anterior: el
    # nombre completo aparece en el caption con minúscula inicial.
    assert ETIQUETAS_ERROR_DISPLAY[una_vez].lower() in tex.lower()
    encabezado = next(l for l in tex.splitlines() if "Modelo" in l)
    # Modelo + JSON + Estr. + Laxa + Lat. + I + 6 etiquetas con >0 ocurrencias
    assert encabezado.count("&") == 5 + len(ETIQUETAS_ERROR) - 1


def test_tabla2_desambigua_se_y_us_en_la_leyenda_con_orden_i_primero():
    """Tarea 2 (SE/US se leían como hermanas y no lo son -- ver
    `judge_2026.py`, definiciones del prompt del juez) + ajuste de la Tarea 5
    (orden de leyenda: I primero, en el mismo orden de las columnas)."""
    conteos = {roster_activo()[0].nombre: {e: 1 for e in ETIQUETAS_ERROR}}
    tex = tabla2_resultados_taxonomia(_resumen(), _taxonomia_df(conteos))
    assert ("SE: sin error semántico (equivalente por otro motivo)" in tex)
    assert ("US: uso de sinónimos (difiere solo por un sinónimo)" in tex)
    assert "I: incorrectas" in tex
    pos_i = tex.index("I: incorrectas")
    pos_ci = tex.index("CI: ")
    assert pos_i < pos_ci


def test_tabla2_no_usa_resizebox_y_mantiene_footnotesize():
    """Restricción dura de ancho de la Tarea 5: `\\resizebox` prohibido,
    `\\footnotesize` como piso para la tabla fusionada."""
    conteos = {roster_activo()[0].nombre: {e: 1 for e in ETIQUETAS_ERROR}}
    tex = tabla2_resultados_taxonomia(_resumen(), _taxonomia_df(conteos))
    assert r"\resizebox" not in tex
    assert r"\footnotesize" in tex


def test_ninguna_tabla_deja_guiones_bajos_sin_escapar():
    df_cat = pd.DataFrame([{"categoria": "encendido_apagado_simple", "n": 8,
                            roster_activo()[0].nombre: 50.0}])
    for tex in (tabla1_modelos(), tabla2_resultados_taxonomia(_resumen(), _taxonomia_df()),
                tabla3_por_categoria(df_cat)):
        for i, ch in enumerate(tex):
            if ch == "_":
                assert tex[i - 1] == "\\", f"guion bajo sin escapar cerca de: {tex[i-30:i+10]!r}"
