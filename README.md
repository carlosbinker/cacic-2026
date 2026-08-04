# SLMs para interpretación de comandos de domótica en español rioplatense

Código de soporte para el paper *"Modelos de Lenguaje Pequeños para la Interpretación de Comandos de Domótica en Español: Un Estudio Comparativo"* (CACIC 2026), que evalúa cuatro Small Language Models open-source (0,36B–1,7B parámetros) interpretando 32 comandos de domótica en español rioplatense.

## ⚠️ Nota sobre el origen de este código — leer antes de usar

**Este repositorio es una reconstrucción documentada, no el código original.** El script que efectivamente cargó los cuatro modelos y generó los resultados publicados en el paper corrió en un entorno de sesión efímero (un contenedor cloud descartado al cerrar la sesión) y nunca se guardó como archivo — solo sobrevivieron sus salidas: el dataset, el CSV de detalle con las 128 corridas y el resumen agregado.

Este código reimplementa la evaluación siguiendo al pie de la letra la metodología descripta en el paper (modelos exactos, prompt, decodificación greedy, métricas), y **se validó** ejecutando la lógica de scoring (`src/scoring.py` + `src/metrics.py`) sobre el CSV de detalle real: reproduce exactamente el resumen agregado publicado (ver `src/validar_contra_resultados_originales.py` y `tests/`). Eso confirma que la lógica de evaluación es fiel a la que se usó. Lo que **no** se pudo validar es que el texto del prompt sea 100% idéntico al original: el paper aclara que el prompt real tenía dos ejemplos few-shot, pero solo transcribe el primero "por espacio" — el segundo ejemplo en `src/prompt.py` es una adición propia, marcada explícitamente como tal en el código.

Si corrés `src/run_evaluation.py` de punta a punta, vas a descargar y ejecutar los cuatro modelos reales — los resultados deberían ser muy similares a los publicados, pero no hay garantía de coincidencia exacta byte a byte (versiones de librerías, build de PyTorch y hardware pueden introducir variaciones menores, aunque la decodificación es determinista/greedy).

## Estructura del repositorio

```
.
├── data/
│   ├── dataset_comandos_domotica.csv        # 32 comandos + ground truth (5 campos)
│   ├── resultados_experimento_detalle.csv   # 128 corridas reales (4 modelos x 32 comandos)
│   └── resultados_experimento_resumen.json  # resumen agregado real, tal como en el paper
├── src/
│   ├── schema.py       # vocabulario cerrado del dominio (intents, dispositivos, ubicaciones)
│   ├── prompt.py        # prompt de sistema (ver nota de fidelidad en el docstring)
│   ├── models.py         # los 4 modelos evaluados, con su repo_id real de HuggingFace
│   ├── scoring.py         # parseo de JSON + comparación contra ground truth (sin deps pesadas)
│   ├── run_evaluation.py   # harness principal: carga modelo, corre inferencia, guarda detalle
│   ├── metrics.py           # agrega el detalle al resumen por modelo (Tabla 2 del paper)
│   ├── generate_figures.py   # regenera fig1 (exactitud/latencia) y fig2 (exactitud por campo)
│   └── validar_contra_resultados_originales.py  # confirma fidelidad del scoring vs. resultados reales
├── tests/
│   └── test_metricas.py    # tests unitarios de scoring.py y metrics.py
├── figures/                  # salida de generate_figures.py
├── Dockerfile                 # entorno reproducible, con límites de CPU/RAM configurables
├── requirements.txt
├── CITATION.cff
└── LICENSE                     # MIT
```

## Cómo correrlo

### Opción 1 — Python directo (recomendado si tenés RAM de sobra, p. ej. 64GB)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Corre los 4 modelos sobre los 32 comandos (descarga ~4GB de pesos la primera vez)
python src/run_evaluation.py --salida data/resultados_reproducidos.csv

# Agrega el detalle a un resumen por modelo
python src/metrics.py --detalle data/resultados_reproducidos.csv --salida data/resumen_reproducido.json

# Regenera las figuras del paper
python src/generate_figures.py
```

Para correr un solo modelo (por ejemplo, mientras probás cambios):
```bash
python src/run_evaluation.py --modelos "Qwen2.5-0.5B-Instruct"
```

### Opción 2 — Docker (aislamiento real: filesystem, red y recursos acotados)

```bash
docker build -t slm-domotica .
docker run --rm --memory=8g --cpus=2 \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/.hf_cache:/app/.hf_cache \
    slm-domotica
```

Los flags `--memory=8g --cpus=2` reproducen el hardware acotado del estudio original (VM con 2 núcleos, 7,8GB de RAM, sin GPU). Una vez que los pesos están cacheados en `.hf_cache/`, se le puede agregar `--network=none` a corridas siguientes para aislar completamente la ejecución de la red.

### Validar la fidelidad del scoring (no requiere descargar modelos)

```bash
python src/validar_contra_resultados_originales.py
# o, equivalente, con pytest:
pytest tests/ -v
```

## Resultados (del paper, Tabla 2)

| Modelo | Parámetros | JSON válido | Coincidencia exacta | Latencia (s) |
|---|---|---|---|---|
| SmolLM2-360M-Instruct | 0,36B | 100,0% | 18,8% | 15,6 |
| Qwen2.5-0.5B-Instruct | 0,49B | 100,0% | 43,8% | 14,7 |
| Qwen2.5-1.5B-Instruct | 1,54B | 100,0% | 50,0% | 43,2 |
| SmolLM2-1.7B-Instruct | 1,71B | 100,0% | 59,4% | 61,9 |

Hallazgo central: los cuatro modelos fallan sistemáticamente (0%–16,7% de exactitud) al interpretar comandos de ajuste relativo o cualitativo sin valor numérico explícito ("bajale un poco a la luz"), tendiendo a alucinar un valor o unidad inexistente en el comando original — un patrón que no mejora con el tamaño del modelo dentro del rango evaluado.

## Estudio 2026 — roster ampliado y evaluación automática

A partir de este estudio, el repositorio incorpora una segunda ronda de
evaluación, ampliada y automatizada, que convive con el código original de
2025 descripto arriba sin modificarlo.

**Roster.** El registro de candidatos (`src/models_2026.py:MODELOS_2026`)
considera **15** modelos abiertos de menos de 2 mil millones de parámetros;
el roster efectivamente evaluado (`roster_activo()`) son **12**, organizados
en dos tiers de seis modelos (sub-1000M y 1000–2000M de parámetros). Los
**3** restantes quedaron fuera por dos causas distintas, ninguna
metodológica: **2** por acceso restringido (repositorios `gated` de
HuggingFace cuya aprobación manual de descarga no llegó a otorgarse) y **1**
(`Qwen3.5-2B`) porque su compatibilidad de versión con la librería de
inferencia quedó sin verificar. El detalle completo, con las versiones de
`transformers` que fuerza cada grupo, está en `docker/README.md` y en la
Tabla 5 del paper reescrito.

**Pipeline de dos etapas.** La etapa 1 compara cada respuesta campo a campo
contra el ground truth (exactitud estricta, determinista). La etapa 2 usa
como juez automático al modelo con mayor exactitud estricta de la etapa 1
(empate hacia el de mayor tamaño), corrido con decodificación greedy, para
clasificar cada respuesta incorrecta en una taxonomía cerrada de siete
etiquetas y para asignar la categoría lingüística de cada uno de los 32
comandos — reemplazando el etiquetado manual del estudio original. Ver
`src/judge_2026.py`, `src/taxonomia_2026.py` y `src/metrics_2026.py`.

**Cómo correr el barrido:**
```bash
python docker/build_all.py   # construye una imagen por modelo del roster activo
python docker/run_sweep.py   # corre las 12 imágenes, de a una, en orden de registro
```
Ambos scripts operan por defecto sobre `roster_activo()` (12 modelos); pedir
explícitamente un modelo excluido falla con un `ValueError` que nombra su
motivo de exclusión.

**Cómo regenerar métricas, figuras y tablas:**
```bash
python src/metrics_2026.py           # agrega data/2026/detalle_2026.csv a resumen_2026.json y CSVs derivados
python src/generate_figures_2026.py  # regenera figures/2026/fig1_*.png y fig2_*.png
python src/generate_tex_tables.py    # regenera paper/02_reescrito/tablas/*.tex (5 fragmentos)
```

**Cómo compilar los dos papers:**
```bash
# transcripción del estudio original (4 modelos)
(cd paper/01_original && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex)
# paper reescrito con el roster ampliado de 12 modelos (envío ciego a CACIC 2026)
(cd paper/02_reescrito && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex)
python scripts/check_anonimato.py paper/02_reescrito   # puerta de envío ciego, código 0 esperado
```

## Cómo citar

Ver `CITATION.cff`. Citación en texto:

> [Autor/a]. "Modelos de Lenguaje Pequeños para la Interpretación de Comandos de Domótica en Español: Un Estudio Comparativo". XXXII Congreso Argentino de Ciencias de la Computación (CACIC 2026).

## Licencia

MIT — ver `LICENSE`.
