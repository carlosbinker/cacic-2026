"""Tests de la etapa 2 (juez LLM) con un juez falso: sin pesos, sin red."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from judge_2026 import (  # noqa: E402
    COLUMNAS_CATEGORIAS,
    COLUMNAS_ETIQUETAS,
    etiqueta_de_respaldo,
    etiquetar_errores,
    etiquetar_categorias,
    prompt_categoria,
    prompt_error,
)
from taxonomia_2026 import CATEGORIAS_LINGUISTICAS, ETIQUETAS_ERROR  # noqa: E402


def _fila(idx=0, modelo="M", comando="Apagá la tele del living.",
          gt='{"intent": "apagar", "dispositivo": "tv", "ubicacion": "living", '
             '"valor": null, "unidad": null}',
          pred='{"intent": "apagar", "dispositivo": "tele", "ubicacion": "living", '
               '"valor": null, "unidad": null}',
          **overrides):
    base = {
        "modelo": modelo, "idx": idx, "comando": comando, "gt": gt,
        "pred_raw": pred, "pred_json": pred, "json_valido": True,
        "parse_note": "", "latencia_s": 1.0,
        "match_intent": True, "match_dispositivo": False, "match_ubicacion": True,
        "match_valor": True, "match_unidad": True, "match_exact": False,
        "modo_prompting": "chat_template", "transformers_version": "4.57.0",
    }
    base.update(overrides)
    return base


def test_el_prompt_de_error_lista_las_siete_etiquetas_y_nada_mas():
    p = prompt_error(_fila())
    for e in ETIQUETAS_ERROR:
        assert e in p
    assert "categorias cerradas" in p.lower() or "unicas etiquetas" in p.lower()


def test_el_prompt_de_error_incluye_comando_esperado_y_obtenido():
    p = prompt_error(_fila())
    assert "Apagá la tele del living." in p
    assert '"dispositivo": "tv"' in p      # esperado
    assert '"dispositivo": "tele"' in p    # obtenido


def test_el_prompt_de_categoria_lista_las_seis_categorias():
    p = prompt_categoria("Poné el aire de la cocina en 20 grados.")
    for c in CATEGORIAS_LINGUISTICAS:
        assert c in p
    assert "Poné el aire de la cocina en 20 grados." in p
    assert "exactamente una" in p.lower()


def test_respaldo_toma_el_primer_campo_que_no_coincide():
    assert etiqueta_de_respaldo(_fila(match_intent=False)) == "confusion_intencion"
    assert etiqueta_de_respaldo(_fila()) == "confusion_dispositivo"
    assert etiqueta_de_respaldo(
        _fila(match_dispositivo=True, match_ubicacion=False)
    ) == "confusion_ubicacion"


def test_respaldo_cuando_los_tres_campos_categoricos_coinciden():
    assert etiqueta_de_respaldo(
        _fila(match_dispositivo=True, match_valor=False)
    ) == "valor_numerico_incorrecto"


class JuezFalso:
    """Devuelve respuestas programadas; registra cuántas veces se lo llamó."""

    def __init__(self, respuestas):
        self.respuestas = list(respuestas)
        self.llamadas = []

    def __call__(self, prompt: str, max_new_tokens: int) -> str:
        self.llamadas.append((prompt, max_new_tokens))
        return self.respuestas.pop(0) if self.respuestas else ""


def test_etiquetar_errores_solo_procesa_las_incorrectas():
    df = pd.DataFrame([
        _fila(idx=0, match_exact=True, match_dispositivo=True),
        _fila(idx=1),
        _fila(idx=2, match_exact=True, match_dispositivo=True),
    ])
    juez = JuezFalso(["uso_de_sinonimos"])
    out = etiquetar_errores(df, juez)
    assert list(out.columns) == COLUMNAS_ETIQUETAS
    assert len(out) == 1 and out.iloc[0]["idx"] == 1
    assert len(juez.llamadas) == 1


def test_etiquetar_errores_normaliza_y_marca_parse_ok():
    df = pd.DataFrame([_fila(idx=1)])
    out = etiquetar_errores(df, JuezFalso(["  Uso_De_Sinonimos ;confusion_dispositivo "]))
    assert out.iloc[0]["etiquetas"] == "uso_de_sinonimos;confusion_dispositivo"
    assert bool(out.iloc[0]["juez_parse_ok"]) is True


def test_reintenta_una_vez_con_el_doble_de_tokens():
    df = pd.DataFrame([_fila(idx=1)])
    juez = JuezFalso(["blah blah no es una etiqueta", "sin_error_semantico"])
    out = etiquetar_errores(df, juez)
    assert len(juez.llamadas) == 2
    assert juez.llamadas[1][1] == 2 * juez.llamadas[0][1]
    assert out.iloc[0]["etiquetas"] == "sin_error_semantico"
    assert bool(out.iloc[0]["juez_parse_ok"]) is True


def test_cae_al_respaldo_tras_dos_fallos_y_marca_parse_ok_false():
    df = pd.DataFrame([_fila(idx=1)])
    out = etiquetar_errores(df, JuezFalso(["ni idea", "tampoco"]))
    assert out.iloc[0]["etiquetas"] == "confusion_dispositivo"
    assert bool(out.iloc[0]["juez_parse_ok"]) is False
    assert out.iloc[0]["juez_raw"] == "tampoco"


def test_toda_fila_incorrecta_recibe_al_menos_una_etiqueta_valida():
    df = pd.DataFrame([_fila(idx=i) for i in range(5)])
    out = etiquetar_errores(df, JuezFalso(["basura"] * 20))
    assert len(out) == 5
    for etiquetas in out["etiquetas"]:
        partes = etiquetas.split(";")
        assert partes and all(p in ETIQUETAS_ERROR for p in partes)


def test_etiquetar_categorias_cubre_los_comandos_una_sola_vez():
    df = pd.DataFrame([
        _fila(idx=0, modelo="A", comando="Prendé la luz."),
        _fila(idx=1, modelo="A", comando="Poné el aire en 20."),
        _fila(idx=0, modelo="B", comando="Prendé la luz."),   # mismo comando, otro modelo
    ])
    juez = JuezFalso(["encendido_apagado_simple", "ajuste_con_valor_numerico"])
    out = etiquetar_categorias(df, juez)
    assert list(out.columns) == COLUMNAS_CATEGORIAS
    assert out["idx"].tolist() == [0, 1]          # una vez por comando, no por corrida
    assert len(juez.llamadas) == 2


def test_categoria_invalida_cae_al_respaldo():
    df = pd.DataFrame([_fila(idx=0, comando="Prendé la luz.")])
    out = etiquetar_categorias(df, JuezFalso(["inventada", "tampoco"]))
    assert out.iloc[0]["categoria"] == "encendido_apagado_simple"
    assert bool(out.iloc[0]["juez_parse_ok"]) is False


def test_categoria_con_dos_valores_no_parsea():
    df = pd.DataFrame([_fila(idx=0, comando="Prendé la luz.")])
    out = etiquetar_categorias(
        df, JuezFalso(["consulta_de_estado;multiples_dispositivos", "consulta_de_estado"])
    )
    assert out.iloc[0]["categoria"] == "consulta_de_estado"
    assert bool(out.iloc[0]["juez_parse_ok"]) is True


def test_etiquetar_errores_invoca_persistir_una_vez_por_modelo_con_acumulado_creciente():
    """F12.3: una llamada por modelo, cada una con más filas que la anterior."""
    df = pd.DataFrame([
        _fila(idx=0, modelo="A"),
        _fila(idx=1, modelo="A"),
        _fila(idx=0, modelo="B"),
    ])
    juez = JuezFalso(["confusion_dispositivo"] * 3)
    llamados = []
    etiquetar_errores(df, juez, persistir=lambda acc: llamados.append(acc.copy()))
    assert len(llamados) == 2
    assert len(llamados[0]) == 2 and llamados[0]["modelo"].tolist() == ["A", "A"]
    assert len(llamados[1]) == 3
    assert all(list(acc.columns) == COLUMNAS_ETIQUETAS for acc in llamados)


def test_etiquetar_categorias_invoca_persistir_una_vez_por_comando():
    df = pd.DataFrame([
        _fila(idx=0, comando="Prendé la luz."),
        _fila(idx=1, comando="Poné el aire en 20."),
    ])
    juez = JuezFalso(["encendido_apagado_simple", "ajuste_con_valor_numerico"])
    llamados = []
    etiquetar_categorias(df, juez, persistir=lambda acc: llamados.append(acc.copy()))
    assert len(llamados) == 2
    assert len(llamados[0]) == 1 and len(llamados[1]) == 2
    assert all(list(acc.columns) == COLUMNAS_CATEGORIAS for acc in llamados)


def test_persistir_none_reproduce_el_resultado_de_antes_del_contrato():
    df = pd.DataFrame([_fila(idx=0, modelo="A"), _fila(idx=1, modelo="B")])
    sin_callback = etiquetar_errores(df, JuezFalso(["confusion_dispositivo"] * 2))
    con_persistir_none = etiquetar_errores(df, JuezFalso(["confusion_dispositivo"] * 2), persistir=None)
    pd.testing.assert_frame_equal(sin_callback, con_persistir_none)


from judge_2026 import (  # noqa: E402
    construir_persistidor_atomico,
    filtrar_pendientes_categorias,
    filtrar_pendientes_errores,
)


def test_escritor_atomico_no_deja_tmp_y_el_csv_parsea_tras_cada_llamada(tmp_path):
    destino = tmp_path / "salida.csv"
    escribir = construir_persistidor_atomico(destino)
    for n in (1, 2, 3):
        escribir(pd.DataFrame({"idx": range(n), "valor": range(n)}))
        assert destino.exists()
        assert not (tmp_path / "salida.csv.tmp").exists()
        assert len(pd.read_csv(destino)) == n


def test_escritor_atomico_con_base_antepone_lo_ya_persistido(tmp_path):
    destino = tmp_path / "salida.csv"
    base = pd.DataFrame({"idx": [0, 1], "valor": [10, 11]})
    escribir = construir_persistidor_atomico(destino, base=base)
    escribir(pd.DataFrame({"idx": [2], "valor": [12]}))
    assert pd.read_csv(destino)["idx"].tolist() == [0, 1, 2]


def test_reanudar_salta_los_pares_modelo_idx_ya_presentes():
    df = pd.DataFrame([
        _fila(idx=0, modelo="A"), _fila(idx=1, modelo="A"), _fila(idx=0, modelo="B"),
    ])
    incorrectas = df[~df["match_exact"].astype(bool)]
    pendientes = filtrar_pendientes_errores(incorrectas, ya_hechos={("A", 0)})
    assert sorted(map(tuple, pendientes[["modelo", "idx"]].values)) == [("A", 1), ("B", 0)]


def test_reanudar_categorias_salta_los_idx_ya_presentes():
    df = pd.DataFrame([_fila(idx=i, comando=f"c{i}") for i in range(4)])
    comandos = df[["idx", "comando"]].drop_duplicates(subset="idx").sort_values("idx")
    pendientes = filtrar_pendientes_categorias(comandos, ya_hechos={0, 2})
    assert pendientes["idx"].tolist() == [1, 3]
