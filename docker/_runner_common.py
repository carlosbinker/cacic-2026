#!/usr/bin/env python3
"""Utilidades compartidas por los runners host-side (`run_sweep.py`,
`run_control.py`, `run_baseline.py`): el mismo envolvente de recursos de
`docker run` (RNF1/RNF6), el mismo fix de decodificación UTF-8 de stderr
(F12.1), el mismo esquema de fila de fallo, y el mismo chequeo liviano de
"CSV ya completo" antes de levantar un contenedor.

Extraído tras la implementación (`refactor: deduplicate after
implementation`): los tres runners lo reimplementaban de forma casi
idéntica, cada uno con su propio comentario explicando el mismo fix. No
cambia comportamiento: cada función de acá es la lógica que antes vivía
repetida, sin ninguna variación.
"""

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from run_sweep_2026 import COLUMNAS_DETALLE, N_COMANDOS_ESPERADO


def docker_run_base(cpuset: str) -> list[str]:
    """El prefijo `docker run` congelado (F11), común a los tres runners.

    `--memory=8g --cpus=2` no son negociables (RNF1) y `--cpuset-cpus`
    tampoco (RNF6): `--cpus` es una cuota del planificador CFS y permite
    migración entre núcleos, así que sin pinning la latencia arrastra
    variabilidad que no es del modelo. El valor tiene que ser el MISMO en
    las 12 corridas del barrido, en las de control y en las de baseline.
    """
    return [
        "docker", "run", "--rm",
        "--memory=8g", "--cpus=2", f"--cpuset-cpus={cpuset}",
    ]


def mounts_datos_y_cache(raiz: Path) -> list[str]:
    """Los dos `-v` que montan los tres runners, siempre en este orden:
    los datos compartidos (`data/`) y la cache de HuggingFace (`.hf_cache/`)."""
    return [
        "-v", f"{raiz / 'data'}:/app/data",
        "-v", f"{raiz / '.hf_cache'}:/app/.hf_cache",
    ]


def correr_contenedor(cmd: list[str], dry_run: bool) -> tuple[int, str]:
    """Corre un `docker run ...` ya armado; devuelve (código de salida, stderr).

    F12.1: el contenedor emite stderr en UTF-8 (barras de progreso, texto en
    español). Sin `encoding` explícito, `text=True` decodifica con la
    codificación preferida del locale del HOST (cp1252 en Windows), que no
    tiene mapeo para bytes como 0x8d y revienta con UnicodeDecodeError,
    abortando la corrida entera. `errors="replace"` garantiza además que
    ningún byte de ningún contenedor pueda tirar abajo la corrida.
    """
    if dry_run:
        return 0, ""
    completado = subprocess.run(
        cmd, stderr=subprocess.PIPE, encoding="utf-8", errors="replace"
    )
    if completado.stderr:
        print(completado.stderr, file=sys.stderr)
    return completado.returncode, completado.stderr or ""


def registrar_fallo(modelo: Any, codigo: int, error: str) -> dict:
    """La fila que documenta un fallo irrecuperable de un modelo/anclaje."""
    return {
        "modelo": modelo.nombre,
        "hf_repo_id": modelo.hf_repo_id,
        "transformers_pin": modelo.transformers_pin,
        "codigo_salida": codigo,
        "error_textual": error,
        "momento_iso": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def csv_ya_completo(ruta: Path) -> bool:
    """Chequeo host-side liviano: evita levantar un contenedor si el CSV de
    ese modelo/anclaje ya está completo y bien formado (mismo criterio que
    `run_sweep_2026.debe_saltear`, sin necesitar `--force` a nivel host)."""
    if not ruta.exists():
        return False
    try:
        df = pd.read_csv(ruta)
    except Exception:
        return False
    if list(df.columns) != COLUMNAS_DETALLE or len(df) != N_COMANDOS_ESPERADO:
        return False
    try:
        indices = sorted(int(i) for i in df["idx"])
    except (TypeError, ValueError):
        return False
    return indices == list(range(N_COMANDOS_ESPERADO))
