"""Tests del prompt 2026 y del despacho de modo de prompting (F2 del índice)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from prompt import SYSTEM_PROMPT_PAPER  # noqa: E402
from prompt_2026 import (  # noqa: E402
    CLAUSULA_TAXATIVA,
    RAW_TEMPLATE,
    SYSTEM_PROMPT_2026,
    construir_entrada,
    detectar_modo,
)


def test_el_prompt_2026_extiende_al_del_paper_sin_alterarlo():
    assert SYSTEM_PROMPT_2026.startswith(SYSTEM_PROMPT_PAPER)
    assert len(SYSTEM_PROMPT_2026) > len(SYSTEM_PROMPT_PAPER)


def test_la_clausula_taxativa_esta_incluida():
    assert CLAUSULA_TAXATIVA in SYSTEM_PROMPT_2026


def test_la_clausula_declara_los_campos_categoricos_cerrados():
    for campo in ('"intent"', '"dispositivo"', '"ubicacion"', '"unidad"'):
        assert campo in CLAUSULA_TAXATIVA
    assert "CATEGORICOS CERRADOS" in CLAUSULA_TAXATIVA


def test_la_clausula_declara_los_sinonimos_violacion_de_formato():
    texto = CLAUSULA_TAXATIVA.lower()
    assert "unicos outputs aceptados" in texto
    assert "sinonimo" in texto
    assert "violacion del formato" in texto


def test_la_clausula_da_un_ejemplo_concreto_de_sinonimo_prohibido():
    assert "televisor" in CLAUSULA_TAXATIVA and "tv" in CLAUSULA_TAXATIVA


def test_el_segundo_ejemplo_fewshot_cubre_ajuste_sin_valor():
    assert "Ejemplo 2" in SYSTEM_PROMPT_2026
    assert '"valor": null, "unidad": null' in SYSTEM_PROMPT_2026


def test_raw_template_tiene_los_dos_huecos_y_termina_pidiendo_json():
    assert "{system}" in RAW_TEMPLATE and "{comando}" in RAW_TEMPLATE
    assert RAW_TEMPLATE.rstrip().endswith("JSON:")


class TokenizerConChat:
    """Doble mínimo de un tokenizer instruct: expone chat_template."""

    chat_template = "{% for m in messages %}{{ m.content }}{% endfor %}"

    def __init__(self):
        self.mensajes_recibidos = None
        self.kwargs_recibidos = None

    def apply_chat_template(self, mensajes, **kwargs):
        self.mensajes_recibidos = mensajes
        self.kwargs_recibidos = kwargs
        return {"input_ids": [[1, 2, 3]], "attention_mask": [[1, 1, 1]]}

    def __call__(self, texto, **kwargs):  # no debe usarse en modo chat
        raise AssertionError("no se debe tokenizar en crudo un modelo con chat_template")


class TokenizerBase:
    """Doble mínimo de un modelo base: chat_template es None."""

    chat_template = None

    def __init__(self):
        self.texto_recibido = None

    def apply_chat_template(self, mensajes, **kwargs):  # no debe usarse
        raise AssertionError("no se debe usar chat_template en un modelo base")

    def __call__(self, texto, **kwargs):
        self.texto_recibido = texto
        return {"input_ids": [[4, 5]]}


def test_detectar_modo_con_chat_template():
    assert detectar_modo(TokenizerConChat()) == "chat_template"


def test_detectar_modo_base_por_none():
    assert detectar_modo(TokenizerBase()) == "raw_completion"


def test_detectar_modo_base_por_cadena_vacia():
    tok = TokenizerBase()
    tok.chat_template = ""
    assert detectar_modo(tok) == "raw_completion"


def test_detectar_modo_base_por_atributo_ausente():
    class SinAtributo:
        pass

    assert detectar_modo(SinAtributo()) == "raw_completion"


def test_construir_entrada_modo_chat_arma_system_y_user():
    tok = TokenizerConChat()
    entrada, modo = construir_entrada(tok, "Prendé la luz del living.")
    assert modo == "chat_template"
    assert "input_ids" in entrada
    assert tok.mensajes_recibidos[0]["role"] == "system"
    assert tok.mensajes_recibidos[0]["content"] == SYSTEM_PROMPT_2026
    assert tok.mensajes_recibidos[1]["role"] == "user"
    assert tok.mensajes_recibidos[1]["content"] == 'Comando: "Prendé la luz del living."'
    # return_dict=True es obligatorio: sin él generate() falla con AttributeError
    assert tok.kwargs_recibidos["return_dict"] is True
    assert tok.kwargs_recibidos["add_generation_prompt"] is True
    assert tok.kwargs_recibidos["return_tensors"] == "pt"


def test_construir_entrada_modo_raw_usa_la_plantilla():
    tok = TokenizerBase()
    entrada, modo = construir_entrada(tok, "Apagá la tele.")
    assert modo == "raw_completion"
    assert "input_ids" in entrada
    assert tok.texto_recibido.startswith(SYSTEM_PROMPT_2026)
    assert 'Comando: "Apagá la tele."' in tok.texto_recibido
    assert tok.texto_recibido.rstrip().endswith("JSON:")


def test_construir_entrada_rechaza_comando_vacio():
    with pytest.raises(ValueError, match="comando vacío"):
        construir_entrada(TokenizerConChat(), "   ")
