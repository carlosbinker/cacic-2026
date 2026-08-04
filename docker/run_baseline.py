#!/usr/bin/env python3
"""Corre la corrida de BASELINE-COMPLETION: el/los modelo(s) del paper
original que todavía faltan medir bajo el prompt/harness 2026 (Delta 05,
subtask 19). Hoy es exactamente 1: `Qwen2.5-0.5B-Instruct`.

Mirroring de `docker/run_control.py` a propósito, con los dos mismos fixes
ya probados: `subprocess.run(..., encoding="utf-8", errors="replace")` (un
byte no-cp1252 en stderr abortó el barrido principal antes de este fix) y
continuación-ante-fallo con `data/2026/baseline_original/fallos_baseline.json`
(lista, siempre existe salvo `--dry-run`). Mismo envolvente de recursos que
todo el proyecto (`--memory=8g --cpus=2 --cpuset-cpus=0-1`), secuencial.

A diferencia de `docker/run_control.py`, esta corrida SÍ puede necesitar
construir una imagen nueva (`docker/build_all.py --modelo
"Qwen2.5-0.5B-Instruct"`, no acá: este script nunca invoca `docker build`).
Monta `src/` de solo lectura por el mismo motivo que `run_control.py`: las
imágenes `slm-domotica-2026:*` hornean `src/` al build time, así que un
entrypoint nuevo (`src/run_baseline_original.py`) es invisible dentro del
contenedor sin este montaje -- ese error exacto ya costó una corrida
fallida en este proyecto.

Uso:
    python docker/run_baseline.py [--force] [--dry-run] [--cpuset PAR]
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(AQUI))

from build_all import tag_imagen  # noqa: E402
from models_2026 import ModeloEvaluado2026  # noqa: E402
from run_baseline_original import modelos_baseline_completion, ruta_baseline  # noqa: E402

CPUSET_POR_DEFECTO = "0-1"
DIR_BASELINE = RAIZ / "data" / "2026" / "baseline_original"
FALLOS_PATH = DIR_BASELINE / "fallos_baseline.json"


def modelos_baseline() -> list[ModeloEvaluado2026]:
    """Los modelos de baseline-completion (hoy, 1: `Qwen2.5-0.5B-Instruct`)."""
    return modelos_baseline_completion()


def comando_run_baseline(modelo: ModeloEvaluado2026, raiz: Path, force: bool = False,
                         cpuset: str = CPUSET_POR_DEFECTO) -> list[str]:
    """El `docker run` de un modelo de baseline-completion contra el entrypoint
    de baseline. Mismo envolvente de recursos que el resto del proyecto
    (RNF1/RNF6). Ninguno de los modelos de baseline-completion es gated, así
    que no hay `--env-file`."""
    cmd = [
        "docker", "run", "--rm",
        "--memory=8g", "--cpus=2", f"--cpuset-cpus={cpuset}",
        "-v", f"{raiz / 'data'}:/app/data",
        "-v", f"{raiz / '.hf_cache'}:/app/.hf_cache",
        # Ver docstring del módulo: mismo fix que docker/run_control.py.
        "-v", f"{raiz / 'src'}:/app/src:ro",
        tag_imagen(modelo),
        "python",
    ]
    if force:
        cmd += ["src/run_baseline_original.py", "--force", "--modelo", modelo.nombre]
    else:
        cmd += ["src/run_baseline_original.py", "--modelo", modelo.nombre]
    return cmd


def _csv_ya_completo(ruta: Path) -> bool:
    """Chequeo host-side liviano, mismo criterio que `run_sweep_2026.debe_saltear`."""
    if not ruta.exists():
        return False
    import pandas as pd

    from run_sweep_2026 import COLUMNAS_DETALLE, N_COMANDOS_ESPERADO
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


def _correr_modelo(modelo: ModeloEvaluado2026, posicion: str, force: bool,
                   dry_run: bool, cpuset: str) -> tuple[int, str]:
    """Corre el contenedor de un modelo; devuelve (código de salida, stderr)."""
    cmd = comando_run_baseline(modelo, RAIZ, force, cpuset)
    print(f"\n=== [{posicion}] baseline::{modelo.nombre} ===")
    print(" ".join(cmd))
    if dry_run:
        return 0, ""
    # Mismo fix que docker/run_sweep.py y docker/run_control.py: sin `encoding`
    # explícito, `text=True` decodifica con la codificación preferida del
    # locale del host (cp1252 en Windows), que revienta con bytes UTF-8 fuera
    # de su rango.
    completado = subprocess.run(
        cmd, stderr=subprocess.PIPE, encoding="utf-8", errors="replace"
    )
    if completado.stderr:
        print(completado.stderr, file=sys.stderr)
    return completado.returncode, completado.stderr or ""


def _registrar_fallo(modelo: ModeloEvaluado2026, codigo: int, error: str) -> dict:
    """La fila que documenta un fallo irrecuperable de un modelo."""
    return {
        "modelo": modelo.nombre,
        "hf_repo_id": modelo.hf_repo_id,
        "transformers_pin": modelo.transformers_pin,
        "codigo_salida": codigo,
        "error_textual": error,
        "momento_iso": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def ejecutar_baseline(modelos: list[ModeloEvaluado2026], force: bool, dry_run: bool,
                      cpuset: str = CPUSET_POR_DEFECTO) -> int:
    """De a un contenedor por vez; no corta al primer fallo (mismo contrato
    F12.1 que `docker/run_sweep.py` y `docker/run_control.py`)."""
    fallos: list[dict] = []
    try:
        for i, modelo in enumerate(modelos, 1):
            ruta = ruta_baseline(modelo.nombre)
            if not force and not dry_run and _csv_ya_completo(ruta):
                print(f"[{i}/{len(modelos)}] {modelo.nombre}: ya completo en {ruta}, se saltea "
                      f"(usá --force)")
                continue
            try:
                codigo, error = _correr_modelo(modelo, f"{i}/{len(modelos)}", force,
                                               dry_run, cpuset)
            except Exception as exc:
                codigo, error = 1, f"{type(exc).__name__}: {exc}"
            if codigo != 0:
                fallos.append(_registrar_fallo(modelo, codigo, error))
                print(f"FALLO {modelo.nombre} (código {codigo}). Se registra y se "
                      f"CONTINÚA con el siguiente modelo.\n{error}", file=sys.stderr)
    finally:
        if not dry_run:
            FALLOS_PATH.parent.mkdir(parents=True, exist_ok=True)
            FALLOS_PATH.write_text(
                json.dumps(fallos, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )

    if fallos:
        print(f"\nCorrida de baseline-completion terminada con {len(fallos)} fallo(s): "
              f"{', '.join(f['modelo'] for f in fallos)}. Detalle en {FALLOS_PATH}.",
              file=sys.stderr)
        return 1
    print("\nCorrida de baseline-completion completa, sin fallos.")
    return 0


def _parsear_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Corrida de baseline-completion: el/los modelo(s) del paper "
                     "original que faltan medir bajo el prompt 2026"
    )
    parser.add_argument("--force", action="store_true",
                        help="rehacer aunque el CSV ya esté completo")
    parser.add_argument("--dry-run", action="store_true",
                        help="imprimir los comandos sin ejecutarlos")
    parser.add_argument("--cpuset", default=CPUSET_POR_DEFECTO,
                        help="par de núcleos al que se fija cada contenedor "
                             "(--cpuset-cpus); el mismo que el resto del proyecto")
    return parser.parse_args()


def main() -> int:
    args = _parsear_argumentos()
    return ejecutar_baseline(modelos_baseline(), args.force, args.dry_run, args.cpuset)


if __name__ == "__main__":
    sys.exit(main())
