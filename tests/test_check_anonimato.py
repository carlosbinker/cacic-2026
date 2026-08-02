"""Tests del verificador de anonimato para envío ciego (F10 / RF15)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from check_anonimato import (  # noqa: E402
    PATRONES_PROHIBIDOS,
    Hallazgo,
    revisar_carpeta,
    revisar_tex,
)


def _tex(tmp_path: Path, contenido: str, nombre: str = "s.tex") -> Path:
    ruta = tmp_path / nombre
    ruta.write_text(contenido, encoding="utf-8")
    return ruta


def test_un_tex_limpio_no_da_hallazgos(tmp_path):
    ruta = _tex(tmp_path, r"""
\section{Introducción}
Los modelos de lenguaje pequeños permiten inferencia local en hardware modesto.
El código y los datos se publicarán tras la revisión.
""")
    assert revisar_tex([ruta]) == []


def test_detecta_un_author_no_vacio(tmp_path):
    ruta = _tex(tmp_path, r"\author{Juan Pérez}")
    hallazgos = revisar_tex([ruta])
    assert len(hallazgos) == 1
    assert isinstance(hallazgos[0], Hallazgo)
    assert hallazgos[0].linea == 1
    assert "Juan Pérez" in hallazgos[0].fragmento


def test_no_marca_un_author_anonimo(tmp_path):
    assert revisar_tex([_tex(tmp_path, r"\author{Anonimizado por revisión ciega}")]) == []
    assert revisar_tex([_tex(tmp_path, r"\author{}")]) == []


def test_detecta_institute_y_afiliacion(tmp_path):
    ruta = _tex(tmp_path, r"\institute{Universidad Nacional de Río Negro}")
    assert len(revisar_tex([ruta])) >= 1


def test_detecta_correo_electronico(tmp_path):
    ruta = _tex(tmp_path, "Contacto: alguien@institucion.edu.ar")
    hallazgos = revisar_tex([ruta])
    assert any("correo" in h.motivo.lower() for h in hallazgos)


def test_detecta_url_de_repositorio(tmp_path):
    ruta = _tex(tmp_path, r"Código en \url{https://github.com/usuario/cacic-2026}")
    hallazgos = revisar_tex([ruta])
    assert any("repositorio" in h.motivo.lower() for h in hallazgos)


def test_detecta_orcid(tmp_path):
    ruta = _tex(tmp_path, r"\orcidID{0000-0002-1825-0097}")
    assert revisar_tex([ruta]) != []


def test_detecta_agradecimientos_y_financiamiento(tmp_path):
    for texto in (r"\begin{acknowledgements}Gracias a...\end{acknowledgements}",
                  r"\section*{Agradecimientos}",
                  "Este trabajo fue financiado por el proyecto PICT-2024-1234."):
        assert revisar_tex([_tex(tmp_path, texto)]) != [], texto


def test_detecta_autorreferencia_deanonimizante(tmp_path):
    ruta = _tex(tmp_path, "En nuestro trabajo previo [3] mostramos que...")
    hallazgos = revisar_tex([ruta])
    assert any("previo" in h.motivo.lower() or "autorreferencia" in h.motivo.lower()
               for h in hallazgos)


def test_reporta_archivo_y_numero_de_linea(tmp_path):
    ruta = _tex(tmp_path, "linea 1\nlinea 2\n\\author{Alguien Real}\n", "seccion.tex")
    h = revisar_tex([ruta])[0]
    assert h.archivo.endswith("seccion.tex")
    assert h.linea == 3


def test_revisar_carpeta_recorre_los_tex_recursivamente(tmp_path):
    (tmp_path / "secciones").mkdir()
    _tex(tmp_path, r"\author{}", "main.tex")
    _tex(tmp_path / "secciones", r"\institute{Facultad de Ingeniería}", "01.tex")
    assert revisar_carpeta(tmp_path) != []


def test_revisar_carpeta_ignora_la_clase_y_el_bst(tmp_path):
    """llncs.cls trae el nombre de Springer y no es un dato del autor."""
    (tmp_path / "llncs.cls").write_text(r"% Springer Verlag, Heidelberg", encoding="utf-8")
    (tmp_path / "splncs03.bst").write_text("% by Springer", encoding="utf-8")
    _tex(tmp_path, r"\author{}", "main.tex")
    assert revisar_carpeta(tmp_path) == []


def test_hay_patrones_para_todas_las_familias_de_fuga():
    motivos = " ".join(m for _, m in PATRONES_PROHIBIDOS).lower()
    for familia in ("autor", "afilia", "correo", "repositorio", "orcid",
                    "agradec", "financ"):
        assert familia in motivos, familia
