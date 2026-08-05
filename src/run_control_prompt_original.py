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

Restringido, en su origen, a los tres anclajes de continuidad
(`ANCLAS_CONTROL`): son los que tienen una cifra publicada contra la que
comparar el efecto del prompt, y además los únicos tres cuyas imágenes
Docker del barrido principal (grupo A, `transformers==4.57.6`) se reusaban
sin reconstruir.

Delta 06 (2026-08-04) amplía la validación de este entrypoint a los
**cuatro** modelos de `roster_baseline_original()`: falta medir
`Qwen2.5-0.5B-Instruct` bajo el prompt ORIGINAL para completar la matriz
4×2 (4 modelos × {prompt original, prompt 2026}) del brazo de generación
anterior -- su imagen ya existe (`slm-domotica-2026:qwen2-5-0-5b-instruct`,
construida en el Delta 05 para la baseline-completion bajo el prompt 2026) y
tampoco se reconstruye. `ANCLAS_CONTROL` se mantiene tal cual (los tres
anclajes con imagen ya reusada por `docker/run_control.py` en su barrido por
defecto); la validación de este módulo pasa a apoyarse en
`roster_baseline_original()`, no en una segunda lista hardcodeada.

Uso:
    python src/run_control_prompt_original.py --modelo "SmolLM2-360M-Instruct" [--force]
    python src/run_control_prompt_original.py --modelo "Qwen2.5-0.5B-Instruct" [--force]
"""

import argparse
import sys
from pathlib import Path

from models_2026 import ModeloEvaluado2026, por_nombre, roster_baseline_original, slug
from prompt import SYSTEM_PROMPT_PAPER
from prompt_2026 import construir_entrada_con_prompt
from run_sweep_2026 import cargar_dataset, debe_saltear, escribir_detalle, evaluar_modelo

RAIZ = Path(__file__).resolve().parent.parent
DIR_CONTROL = RAIZ / "data" / "2026" / "control_prompt_original"

# Los tres modelos que también aparecen en la Tabla 2 publicada y ya están en
# el roster activo (RF19): son el barrido por defecto de `docker/run_control.py`.
# NO es la validación de este módulo (ver `verificar_es_baseline_original`,
# Delta 06): esa se apoya en `roster_baseline_original()`, que agrega el cuarto
# modelo (`Qwen2.5-0.5B-Instruct`, baseline-completion, `activo=False`).
ANCLAS_CONTROL: tuple[str, ...] = (
    "SmolLM2-360M-Instruct",
    "Qwen2.5-1.5B-Instruct",
    "SmolLM2-1.7B-Instruct",
)


def ruta_control(nombre_modelo: str) -> Path:
    """CSV de control que le corresponde a un modelo del baseline original."""
    return DIR_CONTROL / f"{slug(nombre_modelo)}.csv"


def verificar_es_baseline_original(modelo: ModeloEvaluado2026) -> None:
    """La corrida de control está restringida a los 4 modelos de
    `roster_baseline_original()` (Delta 06): los tres anclajes de continuidad
    más `Qwen2.5-0.5B-Instruct`, que completa la matriz 4×2 bajo el prompt
    original. El conjunto permitido se deriva de esa función -- nunca de una
    segunda lista hardcodeada -- para que agregar o quitar un modelo del
    baseline original no requiera tocar dos lugares."""
    nombres_validos = {m.nombre for m in roster_baseline_original()}
    if modelo.nombre not in nombres_validos:
        raise ValueError(
            f"{modelo.nombre} no es uno de los 4 modelos de roster_baseline_original() "
            f"({', '.join(sorted(nombres_validos))}); la corrida de control con el "
            f"prompt original está restringida a ellos."
        )


def construir_entrada_original(tokenizer, comando: str) -> dict:
    """La ÚNICA diferencia respecto del barrido 2026: `SYSTEM_PROMPT_PAPER` en
    vez de `SYSTEM_PROMPT_2026`. Reusa `construir_entrada_con_prompt` (misma
    lógica que el barrido principal, chat_template únicamente)."""
    return construir_entrada_con_prompt(tokenizer, comando, SYSTEM_PROMPT_PAPER)


def _parsear_argumentos() -> argparse.Namespace:
    nombres_validos = [m.nombre for m in roster_baseline_original()]
    parser = argparse.ArgumentParser(
        description="Corrida de control: prompt original del paper sobre el baseline original"
    )
    parser.add_argument("--modelo", required=True,
                        help=f"uno de los 4 modelos de roster_baseline_original(): "
                             f"{', '.join(nombres_validos)}")
    parser.add_argument("--force", action="store_true",
                        help="rehacer aunque el CSV de control ya esté completo")
    return parser.parse_args()


def main() -> int:
    args = _parsear_argumentos()
    modelo = por_nombre(args.modelo)
    verificar_es_baseline_original(modelo)

    salida = ruta_control(modelo.nombre)
    if debe_saltear(salida, args.force):
        print(f"{modelo.nombre}: control ya completo en {salida}, se saltea (usá --force)")
        return 0

    filas = evaluar_modelo(modelo, cargar_dataset(), construir_entrada=construir_entrada_original)
    escribir_detalle(filas, salida)
    return 0


if __name__ == "__main__":
    sys.exit(main())
