---
id: 12
title: Verificador de anonimato para envío ciego
depends_on: []
files:
  - scripts/check_anonimato.py
  - tests/test_check_anonimato.py
  - requirements-dev.txt
---

## Spec

Implementar **F10** / **RF15**: la forma ejecutable de la restricción de envío ciego. El usuario fue explícito — *"Es ciego, no hay que poner ningún dato identificatorio en esta etapa"* — y pidió que exista un paso de verificación, no una inspección a ojo.

El script revisa los `.tex` de una carpeta de paper y, si hay un `main.pdf` construido, también su texto y sus **metadatos** (que son la fuga más fácil de pasar por alto: `\author{}` vacío en el `.tex` no impide que el PDF lleve `/Author` del sistema).

Sale con código 1 si encuentra algo, 0 si está limpio, para poder usarse como puerta en el `Verify` del subtask 15.

No tiene dependencias con el resto del DAG: puede implementarse en cualquier momento, en paralelo. Requiere `pypdf`, que se agrega en un `requirements-dev.txt` nuevo — el `requirements.txt` de producción está congelado y no debe crecer con herramientas de autoría.

## Implementation plan

### Tarea 1 — Dependencia de desarrollo

> Exención de TDD: edición de configuración.

- [ ] Crear `requirements-dev.txt` (no tocar `requirements.txt`):

```
# Herramientas de autoría y verificación, no necesarias para correr el
# experimento. El requirements.txt de produccion se mantiene minimo porque
# se instala dentro de cada imagen de modelo.
-r requirements.txt
pypdf>=5.0.0
```

- [ ] Instalar: `pip install -r requirements-dev.txt`

### Tarea 2 — Revisión de archivos `.tex` (TDD)

- [ ] Escribir el test que falla, `tests/test_check_anonimato.py`:

```python
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
```

- [ ] Correr y confirmar que **falla**: `pytest -q tests/test_check_anonimato.py`
- [ ] Implementar `scripts/check_anonimato.py`:

```python
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
```

- [ ] Correr y confirmar **verde**: `pytest -q tests/test_check_anonimato.py`

### Tarea 3 — Prueba de extremo a extremo contra un PDF real

> Exención de TDD: verificación de integración con el toolchain.

- [ ] Construir dos PDF de prueba en el scratch y confirmar que el script distingue:

```bash
TMP=$(mktemp -d)
cp "LaTeX2e (1)/llncs.cls" "$TMP/"
cat > "$TMP/main.tex" <<'EOF'
\documentclass[runningheads]{llncs}
\usepackage[T1]{fontenc}\usepackage[utf8]{inputenc}
\usepackage[spanish,es-noquoting]{babel}
\begin{document}
\title{Título de prueba}
\author{Anonimizado por revisión ciega}
\institute{Anonimizado por revisión ciega}
\maketitle
\begin{abstract}Resumen de prueba.\end{abstract}
\section{Intro} Texto sin datos identificatorios.
\end{document}
EOF
(cd "$TMP" && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex >/dev/null)
python scripts/check_anonimato.py "$TMP" && echo "LIMPIO detectado bien (exit 0)"

sed -i 's/Anonimizado por revisión ciega/Juan Pérez/' "$TMP/main.tex"
(cd "$TMP" && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex >/dev/null)
python scripts/check_anonimato.py "$TMP"; test $? -eq 1 && echo "SUCIO detectado bien (exit 1)"
```

### Tarea 4 — commit

- [ ] `pytest -q`
- [ ] `git add scripts/check_anonimato.py tests/test_check_anonimato.py requirements-dev.txt`
- [ ] `git commit -m "feat(paper): verificador de anonimato para envio ciego (tex y metadatos del PDF)"`

## Verify

```bash
# 1. Suite verde
pytest -q

# 2. Códigos de salida correctos sobre casos sintéticos
TMP=$(mktemp -d); mkdir -p "$TMP/limpio" "$TMP/sucio"
printf '\\author{}\n\\section{Intro}\nTexto neutro.\n' > "$TMP/limpio/main.tex"
printf '\\author{Juan Pérez}\nContacto: j@uni.edu.ar\n' > "$TMP/sucio/main.tex"
python scripts/check_anonimato.py "$TMP/limpio"; test $? -eq 0 && echo "limpio -> 0 OK"
python scripts/check_anonimato.py "$TMP/sucio";  test $? -eq 1 && echo "sucio  -> 1 OK"
python scripts/check_anonimato.py "$TMP/no-existe"; test $? -eq 2 && echo "error  -> 2 OK"

# 3. requirements.txt de producción sin tocar (F0 de facto: entra en las imágenes)
git diff --exit-code main -- requirements.txt && echo "requirements.txt intacto"

# 4. pypdf disponible y revisar_pdf importable
python -c "
import sys; sys.path.insert(0,'scripts')
from check_anonimato import revisar_pdf
import pypdf; print('pypdf', pypdf.__version__)
"

# 5. E2E contra PDF real (ver Tarea 3)
```

## Acceptance criteria

- **Dado** un `.tex` sin datos identificatorios, **entonces** `revisar_tex` devuelve lista vacía y el CLI sale con código 0.
- **Dado** `\author{Juan Pérez}` o `\institute{Universidad Nacional de ...}`, **entonces** se reporta un `Hallazgo` con el archivo, el número de línea correcto y el fragmento ofensor.
- **Dado** `\author{}` o `\author{Anonimizado por revisión ciega}`, **entonces** **no** se reporta nada: los placeholders anónimos son válidos.
- **Dado** un correo electrónico, una URL de GitHub/GitLab/Zenodo, un ORCID (por comando o por su formato numérico), una sección o bloque de agradecimientos, una declaración de financiamiento, un código de subsidio, o una autorreferencia del tipo "en nuestro trabajo previo", **entonces** cada uno produce al menos un hallazgo con su motivo.
- **Dado** una carpeta de paper, **cuando** se corre `revisar_carpeta`, **entonces** recorre los `.tex` recursivamente e **ignora** `.cls`, `.bst`, `.sty` y los auxiliares del build (que traen nombres de Springer ajenos al autor).
- **Dado** un `main.pdf` presente en la carpeta, **entonces** también se revisan su texto extraído y sus metadatos `/Author`, `/Creator`, `/Producer`, `/Title`, `/Subject`, `/Keywords`, reportando cualquier contenido identificatorio con `linea == 0`.
- **Dado** el CLI, **entonces** sale con 0 si está limpio, 1 si hay hallazgos, y 2 ante un uso incorrecto o una carpeta inexistente, imprimiendo cada hallazgo con motivo, ubicación y fragmento.
- **Dado** el commit, **entonces** `requirements.txt` sigue byte-idéntico a `main` y la dependencia nueva vive solo en `requirements-dev.txt`.
