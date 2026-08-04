#!/usr/bin/env python3
"""
Corrida de CONTROL: el barrido 2026 con el prompt ORIGINAL del paper.

Por qué existe (Delta 04, 2026-08-04) -- leer antes de tocar este archivo:

El barrido 2026 usó un prompt de sistema ENDURECIDO respecto del publicado
(`prompt_2026.SYSTEM_PROMPT_2026`, cláusula taxativa + segundo ejemplo). Los
tres anclajes de continuidad con la Tabla 2 publicada (`SmolLM2-360M-Instruct`,
`Qwen2.5-1.5B-Instruct`, `SmolLM2-1.7B-Instruct`) divergen del publicado con
signo OPUESTO entre sí (-9.4 pp, +15.6 pp, -9.4 pp). Como el signo varía por
modelo, cualquier comparación nuevo-vs-publicado confunde el efecto del
modelo con el efecto del prompt, y el sesgo no se puede argumentar en una
única dirección. Esta corrida de control aísla esa variable.

INVARIANTE DE DISEÑO (no negociable): esta corrida debe diferir del barrido
2026 en EXACTAMENTE UNA variable -- el prompt de sistema. Mismo dataset,
mismos 32 comandos en el mismo orden, misma decodificación (greedy,
temperatura 0), mismo código de scoring, mismas imágenes Docker, mismo
envolvente de recursos. Se logra reusando `run_sweep_2026.evaluar_modelo`
tal cual (mismo bucle de carga/generación/scoring) y pasándole únicamente un
constructor de entrada distinto (`construir_entrada_original`, que sustituye
`SYSTEM_PROMPT_2026` por `SYSTEM_PROMPT_PAPER`, importado tal cual de
`src/prompt.py`, F0 congelado). Si en algún momento hiciera falta cambiar
algo más para que esto corra, hay que PARAR: no es un ajuste, es que el
invariante se rompió.

Restringido a los tres anclajes (`ANCLAS_CONTROL`): son los únicos con una
cifra publicada contra la que comparar el efecto del prompt, y son además
los únicos tres cuyas imágenes Docker del barrido principal (grupo A,
`transformers==4.57.6`) se reusan sin reconstruir.

Uso:
    python src/run_control_prompt_original.py --modelo "SmolLM2-360M-Instruct" [--force]
"""

import argparse
import sys
from pathlib import Path

from models_2026 import ModeloEvaluado2026, por_nombre, slug
from prompt import SYSTEM_PROMPT_PAPER
from prompt_2026 import ModoPrompting, construir_entrada_con_prompt
from run_sweep_2026 import cargar_dataset, debe_saltear, escribir_detalle, evaluar_modelo

RAIZ = Path(__file__).resolve().parent.parent
DIR_CONTROL = RAIZ / "data" / "2026" / "control_prompt_original"

# Los tres modelos que también aparecen en la Tabla 2 publicada (RF19): son
# los únicos con una cifra publicada contra la que comparar, así que la
# corrida de control no tiene sentido -- ni imagen Docker construida -- para
# ningún otro modelo del roster.
ANCLAS_CONTROL: tuple[str, ...] = (
    "SmolLM2-360M-Instruct",
    "Qwen2.5-1.5B-Instruct",
    "SmolLM2-1.7B-Instruct",
)


def ruta_control(nombre_modelo: str) -> Path:
    """CSV de control que le corresponde a un anclaje."""
    return DIR_CONTROL / f"{slug(nombre_modelo)}.csv"


def verificar_es_ancla(modelo: ModeloEvaluado2026) -> None:
    """La corrida de control está restringida a los tres anclajes de continuidad."""
    if modelo.nombre not in ANCLAS_CONTROL:
        raise ValueError(
            f"{modelo.nombre} no es uno de los tres anclajes de continuidad "
            f"({', '.join(ANCLAS_CONTROL)}); la corrida de control con el "
            f"prompt original está restringida a ellos."
        )


def construir_entrada_original(tokenizer, comando: str) -> tuple[dict, ModoPrompting]:
    """La ÚNICA diferencia respecto del barrido 2026: `SYSTEM_PROMPT_PAPER` en
    vez de `SYSTEM_PROMPT_2026`. Reusa `construir_entrada_con_prompt` (misma
    lógica de despacho por capacidad que el barrido principal), así que el
    modo (`chat_template` / `raw_completion`) se decide exactamente igual."""
    return construir_entrada_con_prompt(tokenizer, comando, SYSTEM_PROMPT_PAPER)


def _parsear_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Corrida de control: prompt original del paper sobre un anclaje"
    )
    parser.add_argument("--modelo", required=True,
                        help=f"uno de los tres anclajes: {', '.join(ANCLAS_CONTROL)}")
    parser.add_argument("--force", action="store_true",
                        help="rehacer aunque el CSV de control ya esté completo")
    return parser.parse_args()


def main() -> int:
    args = _parsear_argumentos()
    modelo = por_nombre(args.modelo)
    verificar_es_ancla(modelo)

    salida = ruta_control(modelo.nombre)
    if debe_saltear(salida, args.force):
        print(f"{modelo.nombre}: control ya completo en {salida}, se saltea (usá --force)")
        return 0

    filas = evaluar_modelo(modelo, cargar_dataset(), construir_entrada=construir_entrada_original)
    escribir_detalle(filas, salida)
    return 0


if __name__ == "__main__":
    sys.exit(main())
