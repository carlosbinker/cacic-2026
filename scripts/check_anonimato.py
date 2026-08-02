#!/usr/bin/env python3
"""
Verificador de envío ciego.

CACIC 2026 se envía de forma ciega: el PDF no puede contener ningún dato que
identifique a los autores. Este script es la forma ejecutable de esa
restricción --- revisar a ojo no escala y falla justo en lo menos visible,
que son los metadatos del PDF: un \\author{} vacío en el .tex no impide que
pdflatex escriba /Author con el usuario del sistema operativo.

Uso:
    python scripts/check_anonimato.py paper/02_reescrito
Sale con código 1 si encuentra algo.
"""

import re
import sys
from dataclasses import dataclass
from pathlib import Path

ARCHIVOS_IGNORADOS = {".cls", ".bst", ".sty", ".bbl", ".aux", ".log", ".out", ".fls"}

_PLACEHOLDERS_ACEPTADOS = re.compile(
    r"^\s*(|anonimizado.*|an[oó]nimo.*|omitido.*|oculto.*|"
    r"\[.*\]|autor[ae]?s?\s+an[oó]nimos?.*)\s*$",
    re.IGNORECASE,
)

PATRONES_PROHIBIDOS: list[tuple[str, str]] = [
    (r"\\author\s*\{(?P<v>[^}]*)\}", "author no anonimizado"),
    (r"\\institute\s*\{(?P<v>[^}]*)\}", "afiliación / institute no anonimizado"),
    (r"\\thanks\s*\{(?P<v>[^}]*)\}", "nota de agradecimiento o financiamiento"),
    (r"\\orcidID\s*\{(?P<v>[^}]*)\}", "ORCID identifica al autor"),
    (r"\b[\w.\-+]+@[\w\-]+\.[\w.\-]+\b", "correo electrónico"),
    (r"https?://(?:www\.)?(?:github|gitlab|bitbucket|zenodo|osf)\.\S+",
     "URL de repositorio o dataset identificable"),
    (r"\b\d{4}-\d{4}-\d{4}-\d{3}[\dX]\b", "ORCID identifica al autor"),
    (r"\\(?:section|subsection)\*?\s*\{\s*Agradecimientos?\s*\}",
     "sección de agradecimientos"),
    (r"\\begin\{acknowledge?ments?\}", "bloque de agradecimientos"),
    (r"\b(?:financiad[oa]|subsidiad[oa])\s+por\b", "declaración de financiamiento"),
    (r"\b(?:PICT|PIP|UBACYT|CONICET|ANPCyT)[-\s]?\d*", "subsidio identificable"),
    (r"\b(?:nuestro|nuestra)s?\s+(?:trabajo|investigaci[oó]n|estudio|art[ií]culo)s?\s+"
     r"(?:previo|anterior)", "autorreferencia deanonimizante a un trabajo previo"),
    (r"\bUniversidad\s+(?:Nacional\s+)?[A-ZÁÉÍÓÚÑ]\w+", "afiliación institucional"),
    (r"\bFacultad\s+(?:Regional\s+|de\s+)[A-ZÁÉÍÓÚÑ]\w+", "afiliación institucional"),
]

_CON_VALOR = {"author no anonimizado", "afiliación / institute no anonimizado",
              "nota de agradecimiento o financiamiento"}


@dataclass(frozen=True)
class Hallazgo:
    archivo: str
    linea: int
    patron: str
    motivo: str
    fragmento: str


def _buscar_en_texto(texto: str, origen: str, linea_base: int = 0) -> list[Hallazgo]:
    hallazgos = []
    for numero, linea in enumerate(texto.splitlines(), start=1):
        for patron, motivo in PATRONES_PROHIBIDOS:
            for m in re.finditer(patron, linea):
                # Los comandos con valor (\author{...}) solo son fuga si el
                # valor no es un placeholder anónimo.
                if motivo in _CON_VALOR:
                    valor = (m.groupdict().get("v") or "")
                    if _PLACEHOLDERS_ACEPTADOS.match(valor):
                        continue
                hallazgos.append(Hallazgo(
                    archivo=origen,
                    linea=linea_base or numero,
                    patron=patron,
                    motivo=motivo,
                    fragmento=m.group(0)[:120],
                ))
    return hallazgos


def revisar_tex(rutas: list[Path]) -> list[Hallazgo]:
    hallazgos = []
    for ruta in rutas:
        if ruta.suffix in ARCHIVOS_IGNORADOS:
            continue
        texto = ruta.read_text(encoding="utf-8", errors="ignore")
        hallazgos.extend(_buscar_en_texto(texto, str(ruta)))
    return hallazgos


def revisar_pdf(ruta_pdf: Path) -> list[Hallazgo]:
    """Revisa el texto extraído y, sobre todo, los metadatos del PDF."""
    from pypdf import PdfReader

    lector = PdfReader(str(ruta_pdf))
    hallazgos = []

    meta = lector.metadata or {}
    for clave in ("/Author", "/Creator", "/Producer", "/Title", "/Subject", "/Keywords"):
        valor = str(meta.get(clave, "") or "")
        if clave in ("/Author", "/Subject", "/Keywords") and valor.strip():
            if not _PLACEHOLDERS_ACEPTADOS.match(valor):
                hallazgos.append(Hallazgo(
                    archivo=str(ruta_pdf), linea=0, patron=clave,
                    motivo=f"metadato {clave} del PDF con contenido identificatorio",
                    fragmento=valor[:120],
                ))
        hallazgos.extend(_buscar_en_texto(valor, f"{ruta_pdf} [{clave}]", linea_base=0))

    texto = "\n".join(pagina.extract_text() or "" for pagina in lector.pages)
    hallazgos.extend(_buscar_en_texto(texto, f"{ruta_pdf} [texto]"))
    return hallazgos


def revisar_carpeta(carpeta: Path) -> list[Hallazgo]:
    tex = [p for p in sorted(carpeta.rglob("*.tex"))
           if p.suffix not in ARCHIVOS_IGNORADOS]
    hallazgos = revisar_tex(tex)
    pdf = carpeta / "main.pdf"
    if pdf.exists():
        hallazgos.extend(revisar_pdf(pdf))
    return hallazgos


def main() -> int:
    if len(sys.argv) != 2:
        print("uso: python scripts/check_anonimato.py <carpeta-del-paper>",
              file=sys.stderr)
        return 2
    carpeta = Path(sys.argv[1])
    if not carpeta.is_dir():
        print(f"No es una carpeta: {carpeta}", file=sys.stderr)
        return 2

    hallazgos = revisar_carpeta(carpeta)
    if not hallazgos:
        print(f"OK: sin datos identificatorios en {carpeta}")
        return 0

    print(f"{len(hallazgos)} hallazgo(s) de anonimato en {carpeta}:\n")
    for h in hallazgos:
        ubicacion = f"{h.archivo}:{h.linea}" if h.linea else h.archivo
        print(f"  [{h.motivo}] {ubicacion}\n      {h.fragmento!r}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
