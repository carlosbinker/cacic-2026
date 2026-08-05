"""Tests de las métricas 2026: estricta vs laxa, taxonomía de 7, Tabla 3 (F6)."""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from metrics_2026 import (  # noqa: E402
    CLAVES_RESUMEN,
    PATH_DETALLE,
    PATH_ETIQUETAS,
    calcular_exactitud_por_categoria,
    calcular_resumen_2026,
    calcular_taxonomia_errores_2026,
)
from models_2026 import roster_activo  # noqa: E402
from run_sweep_2026 import COLUMNAS_DETALLE  # noqa: E402
from taxonomia_2026 import ETIQUETAS_ERROR  # noqa: E402

M1, M2 = roster_activo()[0].nombre, roster_activo()[1].nombre

PATH_ETAPA1 = Path(__file__).resolve().parent.parent / "data" / "2026" / "resumen_etapa1.json"


def _det(modelo, patron, latencia=1.0, modo="chat_template"):
    """patron: lista de bools de match_exact, una por comando."""
    filas = []
    for i, ok in enumerate(patron):
        filas.append({
            "modelo": modelo, "idx": i, "comando": f"c{i}", "gt": "{}",
            "pred_raw": "{}", "pred_json": "{}", "json_valido": True,
            "parse_note": "", "latencia_s": latencia,
            "match_intent": ok, "match_dispositivo": ok, "match_ubicacion": ok,
            "match_valor": ok, "match_unidad": ok, "match_exact": ok,
            "modo_prompting": modo, "transformers_version": "4.57.0",
        })
    return pd.DataFrame(filas, columns=COLUMNAS_DETALLE)


def _eti(pares):
    """pares: lista de (modelo, idx, etiquetas)."""
    return pd.DataFrame(
        [{"modelo": m, "idx": i, "etiquetas": e, "juez_raw": e, "juez_parse_ok": True}
         for m, i, e in pares],
        columns=["modelo", "idx", "etiquetas", "juez_raw", "juez_parse_ok"],
    )


def test_resumen_tiene_las_claves_congeladas_y_en_orden():
    det = _det(M1, [True, False, True, False])
    eti = _eti([(M1, 1, "confusion_intencion"), (M1, 3, "uso_de_sinonimos")])
    filas = calcular_resumen_2026(det, eti)
    assert len(filas) == 1
    assert list(filas[0].keys()) == CLAVES_RESUMEN


def test_la_laxa_rescata_sinonimos_y_equivalentes():
    det = _det(M1, [True, False, False, False])
    eti = _eti([
        (M1, 1, "uso_de_sinonimos"),
        (M1, 2, "sin_error_semantico"),
        (M1, 3, "confusion_intencion"),
    ])
    fila = calcular_resumen_2026(det, eti)[0]
    assert fila["exact_match_pct"] == 25.0        # 1 de 4
    assert fila["exact_match_laxo_pct"] == 75.0   # 1 + 2 rescatadas


def test_la_laxa_rescata_aunque_la_etiqueta_venga_combinada():
    det = _det(M1, [False])
    eti = _eti([(M1, 0, "confusion_ubicacion;uso_de_sinonimos")])
    assert calcular_resumen_2026(det, eti)[0]["exact_match_laxo_pct"] == 100.0


def test_la_laxa_nunca_es_menor_que_la_estricta():
    det = pd.concat([_det(M1, [True, False] * 8), _det(M2, [False] * 16)])
    eti = _eti([(M1, i, "confusion_intencion") for i in range(1, 16, 2)]
               + [(M2, i, "sin_error_semantico") for i in range(16)])
    for fila in calcular_resumen_2026(det, eti):
        assert fila["exact_match_laxo_pct"] >= fila["exact_match_pct"]
        assert 0.0 <= fila["exact_match_pct"] <= 100.0
        assert 0.0 <= fila["exact_match_laxo_pct"] <= 100.0


def test_sin_etiquetas_de_equivalencia_ambas_metricas_coinciden():
    det = _det(M1, [True, False])
    eti = _eti([(M1, 1, "confusion_dispositivo")])
    fila = calcular_resumen_2026(det, eti)[0]
    assert fila["exact_match_laxo_pct"] == fila["exact_match_pct"] == 50.0


def test_json_valido_no_confia_en_bool_de_string():
    """Regresion: bool("False") es True en Python. Si json_valido (o
    cualquier columna booleana) llega como texto en vez de bool nativo
    -p.ej. por una lectura de CSV que no infirio el dtype-, el computo no
    debe tratar todas las filas como validas."""
    det = _det(M1, [True, False, False, False])
    det["json_valido"] = ["True", "False", "False", "True"]  # 2 de 4 validas, no 4 de 4
    fila = calcular_resumen_2026(det, _eti([]))[0]
    assert fila["json_valido_pct"] == 50.0


def test_match_exact_no_confia_en_bool_de_string():
    """Misma trampa que json_valido pero sobre match_exact: si llegara como
    texto, bool("False") == True inflaria tanto la estricta como la laxa."""
    det = _det(M1, [True, False, False, False])
    det["match_exact"] = ["True", "False", "False", "False"]
    det["match_intent"] = det["match_exact"]
    det["match_dispositivo"] = det["match_exact"]
    det["match_ubicacion"] = det["match_exact"]
    det["match_valor"] = det["match_exact"]
    det["match_unidad"] = det["match_exact"]
    fila = calcular_resumen_2026(det, _eti([]))[0]
    assert fila["exact_match_pct"] == 25.0
    assert fila["acc_intent_pct"] == 25.0


@pytest.mark.skipif(
    not (PATH_DETALLE.exists() and PATH_ETIQUETAS.exists() and PATH_ETAPA1.exists()),
    reason="requiere los artefactos reales de data/2026/ (etapas 1 y 2)",
)
def test_json_valido_coincide_con_resumen_etapa1_para_los_12_modelos():
    """AC de Bug 1: detalle_2026.csv es la fuente autoritativa; el
    json_valido_pct que agrega metrics_2026 tiene que coincidir exactamente
    con el de resumen_etapa1.json modelo a modelo, y no pueden ser todos
    100% (LFM2.5-230M y SmolLM2-360M-Instruct tienen fallos reales)."""
    det = pd.read_csv(PATH_DETALLE)
    eti = pd.read_csv(PATH_ETIQUETAS)
    etapa1 = {
        f["modelo"]: f["json_valido_pct"]
        for f in json.loads(PATH_ETAPA1.read_text(encoding="utf-8"))
    }
    resumen = calcular_resumen_2026(det, eti)
    assert len(resumen) == 12
    for fila in resumen:
        assert fila["json_valido_pct"] == etapa1[fila["modelo"]], fila["modelo"]
    assert {f["json_valido_pct"] for f in resumen} != {100.0}


def test_laxa_credita_filas_con_json_invalido_si_el_juez_las_marco_equivalentes():
    """Semantica fijada de la laxa (Bug 2, verdicto b): 'laxa' es
    estricta OR etiqueta-de-equivalencia del juez, sin condicionar por
    json_valido. Es comportamiento intencional -no un bug de computo-, pero
    debe quedar fijado para que no derive en silencio, y el conteo de filas
    creditadas con JSON invalido tiene que aparecer en el resumen."""
    det = _det(M1, [False, False, False])
    det.loc[det["idx"] == 0, "json_valido"] = False  # invalida, igual rescatada
    det.loc[det["idx"] == 1, "json_valido"] = False  # invalida, no rescatada
    det.loc[det["idx"] == 2, "json_valido"] = True   # valida, rescatada
    eti = _eti([(M1, 0, "uso_de_sinonimos"), (M1, 2, "sin_error_semantico")])
    fila = calcular_resumen_2026(det, eti)[0]
    assert fila["exact_match_pct"] == 0.0
    assert fila["exact_match_laxo_pct"] == round(100 * 2 / 3, 1)
    assert fila["laxo_json_invalido_n"] == 1  # solo idx 0: invalida y rescatada


def test_laxo_json_invalido_n_es_cero_cuando_toda_json_valida_es_rescatada():
    det = _det(M1, [True, False])
    eti = _eti([(M1, 1, "uso_de_sinonimos")])
    fila = calcular_resumen_2026(det, eti)[0]
    assert fila["exact_match_laxo_pct"] == 100.0
    assert fila["laxo_json_invalido_n"] == 0


def test_el_resumen_trae_metadatos_del_registro():
    det = _det(M1, [True], modo="raw_completion")
    fila = calcular_resumen_2026(det, _eti([]))[0]
    assert fila["hf_repo_id"] == roster_activo()[0].hf_repo_id
    assert fila["tier"] == roster_activo()[0].tier
    assert fila["transformers_pin"] == roster_activo()[0].transformers_pin
    assert fila["modo_prompting"] == "raw_completion"


def test_el_resumen_respeta_el_orden_del_roster():
    det = pd.concat([_det(M2, [True]), _det(M1, [True])])  # al revés a propósito
    assert [f["modelo"] for f in calcular_resumen_2026(det, _eti([]))] == [M1, M2]


def test_taxonomia_tiene_las_siete_columnas_separadas():
    det = _det(M1, [False, False])
    eti = _eti([(M1, 0, "alucinacion_valor_unidad"), (M1, 1, "valor_numerico_incorrecto")])
    tax = calcular_taxonomia_errores_2026(det, eti)
    assert list(tax.columns) == ["modelo", "total_incorrectas"] + ETIQUETAS_ERROR
    fila = tax.iloc[0]
    assert fila["alucinacion_valor_unidad"] == 1
    assert fila["valor_numerico_incorrecto"] == 1
    assert "confusion_valor_o_unidad" not in tax.columns   # el legacy las fusionaba


def test_taxonomia_cuenta_etiquetas_no_excluyentes():
    det = _det(M1, [False, False, False])
    eti = _eti([
        (M1, 0, "confusion_intencion;confusion_dispositivo"),
        (M1, 1, "confusion_intencion"),
        (M1, 2, "uso_de_sinonimos"),
    ])
    fila = calcular_taxonomia_errores_2026(det, eti).iloc[0]
    assert fila["total_incorrectas"] == 3
    assert fila["confusion_intencion"] == 2
    assert fila["confusion_dispositivo"] == 1
    assert fila["uso_de_sinonimos"] == 1
    assert fila["confusion_ubicacion"] == 0


def test_taxonomia_incluye_una_fila_por_modelo_en_orden_de_roster():
    det = pd.concat([_det(M2, [False]), _det(M1, [False])])
    eti = _eti([(M1, 0, "confusion_intencion"), (M2, 0, "confusion_ubicacion")])
    tax = calcular_taxonomia_errores_2026(det, eti)
    assert tax["modelo"].tolist() == [M1, M2]


def test_exactitud_por_categoria_agrupa_bien():
    # idx 0,1 -> encendido simple ; idx 2,3 -> ajuste con valor
    det = _det(M1, [True, False, True, True])
    cat = pd.DataFrame([
        {"idx": 0, "comando": "c0", "categoria": "encendido_apagado_simple",
         "juez_raw": "", "juez_parse_ok": True},
        {"idx": 1, "comando": "c1", "categoria": "encendido_apagado_simple",
         "juez_raw": "", "juez_parse_ok": True},
        {"idx": 2, "comando": "c2", "categoria": "ajuste_con_valor_numerico",
         "juez_raw": "", "juez_parse_ok": True},
        {"idx": 3, "comando": "c3", "categoria": "ajuste_con_valor_numerico",
         "juez_raw": "", "juez_parse_ok": True},
    ])
    tabla = calcular_exactitud_por_categoria(det, cat)
    assert list(tabla.columns) == ["categoria", "n", M1]
    por_cat = tabla.set_index("categoria")
    assert por_cat.loc["encendido_apagado_simple", "n"] == 2
    assert por_cat.loc["encendido_apagado_simple", M1] == 50.0
    assert por_cat.loc["ajuste_con_valor_numerico", M1] == 100.0


def test_exactitud_por_categoria_omite_categorias_sin_comandos():
    det = _det(M1, [True])
    cat = pd.DataFrame([{"idx": 0, "comando": "c0", "categoria": "consulta_de_estado",
                         "juez_raw": "", "juez_parse_ok": True}])
    tabla = calcular_exactitud_por_categoria(det, cat)
    assert tabla["categoria"].tolist() == ["consulta_de_estado"]


def test_exactitud_por_categoria_respeta_el_orden_canonico():
    det = _det(M1, [True, True])
    cat = pd.DataFrame([
        {"idx": 0, "comando": "c0", "categoria": "consulta_de_estado",
         "juez_raw": "", "juez_parse_ok": True},
        {"idx": 1, "comando": "c1", "categoria": "encendido_apagado_simple",
         "juez_raw": "", "juez_parse_ok": True},
    ])
    tabla = calcular_exactitud_por_categoria(det, cat)
    # encendido_apagado_simple va antes que consulta_de_estado en el vocabulario
    assert tabla["categoria"].tolist() == ["encendido_apagado_simple", "consulta_de_estado"]
