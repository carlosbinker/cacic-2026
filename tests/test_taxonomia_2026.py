"""Tests de los vocabularios cerrados de la etapa 2 (F4 del índice)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from taxonomia_2026 import (  # noqa: E402
    CATEGORIAS_DISPLAY,
    CATEGORIAS_LINGUISTICAS,
    ETIQUETAS_EQUIVALENTES,
    ETIQUETAS_ERROR,
    ETIQUETAS_ERROR_DISPLAY,
    parsear_categoria,
    parsear_etiquetas,
)


def test_vocabularios_tienen_el_tamano_congelado():
    assert len(ETIQUETAS_ERROR) == 7
    assert len(CATEGORIAS_LINGUISTICAS) == 6
    assert ETIQUETAS_EQUIVALENTES == {"sin_error_semantico", "uso_de_sinonimos"}


def test_las_cinco_primeras_etiquetas_son_la_tabla4_publicada():
    assert ETIQUETAS_ERROR[:5] == [
        "confusion_intencion",
        "confusion_dispositivo",
        "confusion_ubicacion",
        "alucinacion_valor_unidad",
        "valor_numerico_incorrecto",
    ]


def test_todo_valor_tiene_nombre_para_mostrar():
    assert set(ETIQUETAS_ERROR_DISPLAY) == set(ETIQUETAS_ERROR)
    assert set(CATEGORIAS_DISPLAY) == set(CATEGORIAS_LINGUISTICAS)


def test_parsear_etiquetas_acepta_lista_separada_por_punto_y_coma():
    assert parsear_etiquetas("confusion_intencion;uso_de_sinonimos") == [
        "confusion_intencion",
        "uso_de_sinonimos",
    ]


def test_parsear_etiquetas_normaliza_espacios_mayusculas_y_duplicados():
    assert parsear_etiquetas("  Confusion_Intencion ; confusion_intencion ") == [
        "confusion_intencion"
    ]


def test_parsear_etiquetas_rechaza_vacio():
    with pytest.raises(ValueError, match="al menos una etiqueta"):
        parsear_etiquetas("   ")


def test_parsear_etiquetas_rechaza_fuera_de_vocabulario_y_nombra_al_ofensor():
    with pytest.raises(ValueError, match="error_raro"):
        parsear_etiquetas("confusion_intencion;error_raro")


def test_parsear_categoria_acepta_una_sola():
    assert parsear_categoria(" Consulta_De_Estado ") == "consulta_de_estado"


def test_parsear_categoria_rechaza_dos():
    with pytest.raises(ValueError, match="exactamente una"):
        parsear_categoria("consulta_de_estado;multiples_dispositivos")


def test_parsear_categoria_rechaza_fuera_de_vocabulario():
    with pytest.raises(ValueError, match="otra_cosa"):
        parsear_categoria("otra_cosa")
