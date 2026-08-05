"""Tests de la lógica pura del harness de barrido 2026 (F5 del índice).

No se ejerce inferencia real: se testea el contrato de columnas, el armado
de filas y la decisión de reanudación, que es lo que puede romperse en
silencio durante un barrido de varias horas.
"""

import os
import re
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from models_2026 import por_nombre  # noqa: E402
from run_sweep_2026 import (  # noqa: E402
    COLUMNAS_DETALLE,
    N_COMANDOS_ESPERADO,
    _credencial_hf,
    armar_fila,
    debe_saltear,
    ruta_detalle,
    verificar_credencial_gated,
)


def test_columnas_congeladas_en_orden():
    assert COLUMNAS_DETALLE == [
        "modelo", "idx", "comando", "gt", "pred_raw", "pred_json",
        "json_valido", "parse_note", "latencia_s",
        "match_intent", "match_dispositivo", "match_ubicacion",
        "match_valor", "match_unidad", "match_exact",
        "modo_prompting", "transformers_version",
    ]


def test_las_quince_primeras_columnas_son_las_del_csv_legacy():
    legacy = pd.read_csv(
        Path(__file__).resolve().parent.parent / "data" / "resultados_experimento_detalle.csv",
        nrows=1,
    )
    assert COLUMNAS_DETALLE[:15] == list(legacy.columns)


def test_ruta_detalle_usa_el_slug():
    ruta = ruta_detalle("Qwen3.5-0.8B")
    assert ruta.name == "qwen3-5-0-8b.csv"
    assert ruta.parent.name == "detalle" and ruta.parent.parent.name == "2026"


def test_verificar_credencial_gated_lanza_valueerror_sin_token(monkeypatch):
    """F11: falla temprano y en español si falta $HF_TOKEN para un modelo gated."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    modelo = por_nombre("gemma-3-270m-it")
    with pytest.raises(ValueError) as exc:
        verificar_credencial_gated(modelo)
    mensaje = str(exc.value)
    assert modelo.nombre in mensaje
    assert ".env" in mensaje
    assert not re.search(r"hf_[A-Za-z0-9]{20,}", mensaje), "no debe imprimir el valor del token"


def test_verificar_credencial_gated_no_hace_nada_para_no_gated(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    modelo = por_nombre("SmolLM2-360M-Instruct")
    verificar_credencial_gated(modelo)  # no debe lanzar: este modelo no es gated


def _csv_completo(tmp_path: Path, n: int) -> Path:
    ruta = tmp_path / "m.csv"
    pd.DataFrame(
        {c: [None] * n for c in COLUMNAS_DETALLE} | {"idx": list(range(n))}
    ).to_csv(ruta, index=False)
    return ruta


def test_debe_saltear_si_el_csv_esta_completo(tmp_path):
    assert debe_saltear(_csv_completo(tmp_path, N_COMANDOS_ESPERADO), force=False) is True


def test_no_saltea_si_falta_alguna_fila(tmp_path):
    assert debe_saltear(_csv_completo(tmp_path, N_COMANDOS_ESPERADO - 1), force=False) is False


def test_no_saltea_si_no_existe(tmp_path):
    assert debe_saltear(tmp_path / "no-existe.csv", force=False) is False


def test_force_nunca_saltea(tmp_path):
    assert debe_saltear(_csv_completo(tmp_path, N_COMANDOS_ESPERADO), force=True) is False


def test_no_saltea_si_el_csv_esta_corrupto(tmp_path):
    """Un barrido interrumpido a mitad de escritura no debe darse por bueno."""
    ruta = tmp_path / "corrupto.csv"
    ruta.write_text("modelo,idx\nesto,no,tiene,el,esquema\n", encoding="utf-8")
    assert debe_saltear(ruta, force=False) is False


def test_no_saltea_si_hay_un_idx_duplicado(tmp_path):
    """32 filas pero con un idx repetido significa que falta un comando."""
    ruta = _csv_completo(tmp_path, N_COMANDOS_ESPERADO)
    df = pd.read_csv(ruta)
    df.loc[N_COMANDOS_ESPERADO - 1, "idx"] = 0
    df.to_csv(ruta, index=False)
    assert debe_saltear(ruta, force=False) is False


def test_no_saltea_si_un_idx_esta_fuera_de_rango(tmp_path):
    """Un idx fuera de 0..31 delata un CSV armado con otro dataset."""
    ruta = _csv_completo(tmp_path, N_COMANDOS_ESPERADO)
    df = pd.read_csv(ruta)
    df.loc[0, "idx"] = 99
    df.to_csv(ruta, index=False)
    assert debe_saltear(ruta, force=False) is False


def test_armar_fila_produce_exactamente_las_columnas_y_los_tipos():
    modelo = por_nombre("SmolLM2-360M-Instruct")
    gt = {"intent": "encender", "dispositivo": "luz", "ubicacion": "living",
          "valor": None, "unidad": None}
    fila = armar_fila(
        modelo=modelo,
        idx=0,
        comando="Prendé la luz del living.",
        gt=gt,
        texto_generado='{"intent": "encender", "dispositivo": "luz", '
                       '"ubicacion": "living", "valor": null, "unidad": null}',
        latencia_s=1.23456,
        modo="chat_template",
        transformers_version="4.57.0",
    )
    assert list(fila.keys()) == COLUMNAS_DETALLE
    assert fila["modelo"] == "SmolLM2-360M-Instruct"
    assert fila["json_valido"] is True
    assert fila["match_exact"] is True
    assert fila["latencia_s"] == 1.235
    assert fila["modo_prompting"] == "chat_template"
    assert fila["transformers_version"] == "4.57.0"


def test_armar_fila_marca_incorrecta_una_respuesta_con_sinonimo():
    """'tele' en lugar de 'tv' es incorrecta en la etapa 1 (la laxa la rescata luego)."""
    modelo = por_nombre("SmolLM2-360M-Instruct")
    gt = {"intent": "apagar", "dispositivo": "tv", "ubicacion": "living",
          "valor": None, "unidad": None}
    fila = armar_fila(
        modelo=modelo, idx=5, comando="Apagá la tele del living.", gt=gt,
        texto_generado='{"intent": "apagar", "dispositivo": "tele", '
                       '"ubicacion": "living", "valor": null, "unidad": null}',
        latencia_s=2.0, modo="chat_template", transformers_version="4.57.0",
    )
    assert fila["json_valido"] is True
    assert fila["match_dispositivo"] is False
    assert fila["match_exact"] is False


def test_armar_fila_con_salida_no_parseable():
    modelo = por_nombre("LFM2.5-230M")
    gt = {"intent": "encender", "dispositivo": "luz", "ubicacion": "living",
          "valor": None, "unidad": None}
    fila = armar_fila(
        modelo=modelo, idx=1, comando="Prendé la luz.", gt=gt,
        texto_generado="No entiendo el comando.", latencia_s=0.5,
        modo="raw_completion", transformers_version="4.57.0",
    )
    assert fila["json_valido"] is False
    assert fila["pred_json"] == ""
    assert fila["parse_note"] == "no_parseable_como_json"
    assert fila["match_exact"] is False
    assert fila["modo_prompting"] == "raw_completion"


def test_armar_fila_rechaza_un_modo_invalido():
    modelo = por_nombre("LFM2.5-230M")
    with pytest.raises(ValueError, match="modo_prompting"):
        armar_fila(
            modelo=modelo, idx=1, comando="x", gt={}, texto_generado="{}",
            latencia_s=0.1, modo="inventado", transformers_version="4.57.0",
        )


def test_el_modulo_se_importa_sin_torch_ni_transformers():
    """Los imports pesados son locales a `evaluar_modelo`: los contenedores
    mínimos y la suite de tests deben poder importar el harness sin ellos."""
    fuente = (
        Path(__file__).resolve().parent.parent / "src" / "run_sweep_2026.py"
    ).read_text(encoding="utf-8")
    for linea in fuente.splitlines():
        if linea.startswith(("import ", "from ")):
            assert "torch" not in linea and "transformers" not in linea, linea


def test_el_harness_no_contiene_ningun_token_literal():
    """RF17: el valor del token vive solo en .env, nunca en un archivo versionado."""
    fuente = (
        Path(__file__).resolve().parent.parent / "src" / "run_sweep_2026.py"
    ).read_text(encoding="utf-8")
    assert not re.search(r"hf_[A-Za-z0-9]{20,}", fuente)
    # El token se lee solo del entorno, en runtime; nunca por un flujo de login.
    assert not re.search(r"^\s*(import|from)\s+huggingface_hub", fuente, re.MULTILINE)
    assert 'os.environ["HF_TOKEN"]' in fuente


def test_solo_los_modelos_gated_reciben_el_token(monkeypatch):
    """F11: el token viaja por `token=` de from_pretrained y solo para gated."""
    monkeypatch.setitem(os.environ, "HF_TOKEN", "hf_token_de_prueba")
    assert _credencial_hf(por_nombre("gemma-3-270m-it")) == {"token": "hf_token_de_prueba"}
    assert _credencial_hf(por_nombre("SmolLM2-360M-Instruct")) == {}
