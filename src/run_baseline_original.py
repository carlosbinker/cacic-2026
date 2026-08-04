#!/usr/bin/env python3
"""
Corrida de BASELINE-COMPLETION: el modelo del paper original que todavía
falta medir bajo el harness/prompt 2026 (Delta 05, subtask 19).

Por qué existe -- leer antes de tocar este archivo:

Un control con el prompt ORIGINAL del paper (subtask 18) NO reprodujo la
Tabla 2 publicada para los tres anclajes de continuidad. La causa, con
evidencia: `requirements.txt` legacy fija `transformers>=4.46.0` SIN cota
superior y `data/resultados_experimento_detalle.csv` no tiene columna
`transformers_version` -- el entorno de la corrida original nunca quedó
fijado ni registrado, así que es irrecuperable. Es irreproducibilidad del
trabajo original, no un defecto del harness nuevo (`tests/test_metricas.py`
sigue verde: la ruta de scoring es fiel).

Consecuencia: las cifras publicadas pasan a referencia histórica no
reproducida. El paper compara en cambio contra un baseline RE-MEDIDO e
internamente consistente de los 4 modelos del paper original, bajo el
harness/prompt 2026, misma máquina, temperatura 0. Tres de esos 4
(`SmolLM2-360M-Instruct`, `SmolLM2-1.7B-Instruct`, `Qwen2.5-1.5B-Instruct`)
ya corren en el barrido principal (`data/2026/detalle/`, `activo=True`).
Este entrypoint cubre el cuarto, `Qwen2.5-0.5B-Instruct`
(`baseline_original=True`, `activo=False`).

DIFERENCIA con `src/run_control_prompt_original.py` (no confundir los dos):
la corrida de control sustituye el prompt de sistema por el ORIGINAL del
paper para aislar el efecto del prompt; esta corrida usa el prompt 2026 por
DEFECTO -- exactamente las mismas condiciones que el barrido principal --
porque el objetivo es medir el modelo, no aislar una variable del prompt.
Por eso reusa `run_sweep_2026.evaluar_modelo` SIN pasar un
`construir_entrada` alternativo.

Uso:
    python src/run_baseline_original.py --modelo "Qwen2.5-0.5B-Instruct" [--force]
"""

import argparse
import sys
from pathlib import Path

from models_2026 import ModeloEvaluado2026, por_nombre, roster_baseline_original, slug
from run_sweep_2026 import cargar_dataset, debe_saltear, escribir_detalle, evaluar_modelo

RAIZ = Path(__file__).resolve().parent.parent
DIR_BASELINE = RAIZ / "data" / "2026" / "baseline_original"


def modelos_baseline_completion() -> list[ModeloEvaluado2026]:
    """Los `baseline_original` que TODAVÍA no están en el roster activo.

    Hoy es exactamente 1 (`Qwen2.5-0.5B-Instruct`). Los otros 3
    `baseline_original` ya corrieron en el barrido principal (`activo=True`)
    y no se re-corren acá: `data/2026/detalle/*.csv` es de solo lectura, y
    re-medirlos ahí sería un dato duplicado, no uno nuevo.
    """
    return [m for m in roster_baseline_original() if not m.activo]


def ruta_baseline(nombre_modelo: str) -> Path:
    """CSV de baseline-completion que le corresponde a un modelo."""
    return DIR_BASELINE / f"{slug(nombre_modelo)}.csv"


def verificar_es_baseline_completion(modelo: ModeloEvaluado2026) -> None:
    """Esta corrida está restringida a los modelos de `modelos_baseline_completion()`."""
    nombres = [m.nombre for m in modelos_baseline_completion()]
    if modelo.nombre not in nombres:
        raise ValueError(
            f"{modelo.nombre} no es un modelo de baseline-completion "
            f"({', '.join(nombres)}); esta corrida está restringida a ellos."
        )


def _parsear_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Corrida de baseline-completion: prompt 2026 sobre el modelo "
                     "del paper original que falta medir"
    )
    parser.add_argument("--modelo", required=True,
                        help="uno de los modelos de baseline-completion (ver "
                             "src/run_baseline_original.py:modelos_baseline_completion)")
    parser.add_argument("--force", action="store_true",
                        help="rehacer aunque el CSV de baseline ya esté completo")
    return parser.parse_args()


def main() -> int:
    args = _parsear_argumentos()
    modelo = por_nombre(args.modelo)
    verificar_es_baseline_completion(modelo)

    salida = ruta_baseline(modelo.nombre)
    if debe_saltear(salida, args.force):
        print(f"{modelo.nombre}: baseline ya completo en {salida}, se saltea (usá --force)")
        return 0

    filas = evaluar_modelo(modelo, cargar_dataset())
    escribir_detalle(filas, salida)
    return 0


if __name__ == "__main__":
    sys.exit(main())
