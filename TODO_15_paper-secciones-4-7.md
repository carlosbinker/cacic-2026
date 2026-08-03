---
id: 15
title: "Paper reescrito: secciones 4–7, build y anonimato"
depends_on: [14]
files:
  - paper/02_reescrito/secciones/04_resultados.tex
  - paper/02_reescrito/secciones/05_discusion.tex
  - paper/02_reescrito/secciones/06_amenazas.tex
  - paper/02_reescrito/secciones/07_conclusiones.tex
  - paper/02_reescrito/secciones/08_declaracion_ia.tex
  - paper/02_reescrito/refs.bib
  - README.md
---

## Spec

Cierre del paper reescrito: las secciones que dependen de los resultados (§4 Experimentación y resultados, §5 Discusión, §6 Amenazas a la validez, §7 Conclusiones) más la declaración sobre uso de IA, el build final y la verificación de envío ciego.

Aquí se materializan tres cosas que el índice exige y que no pueden faltar:

- **RF10** — la brecha entre exactitud estricta y laxa se reporta y se **discute** como resultado, porque es la respuesta directa a la amenaza de validez de constructo que el estudio original dejó planteada.
- **RF11** — §4.4 pasa de "se clasificó manualmente cada una de las 73 respuestas incorrectas… en una o más de cuatro categorías" a una clasificación automática con **siete** etiquetas; la mención a "cuatro categorías" del original debe desaparecer.
- **RF16** — §6 incorpora **tres amenazas nuevas**: (1) sesgo de auto-favorecimiento del juez; (2) versiones divergentes de la librería de inferencia entre modelos (dos grupos de versión mutuamente excluyentes sobre el roster) como confusor de la latencia; (3) exclusión de 2 de los 14 modelos del diseño original por acceso restringido no otorgado. **Se elimina** la amenaza de incomparabilidad de los dos modelos base prompteados por raw completion: no aplica, porque los 12 del roster activo usan `chat_template`.

Todos los números vienen de los fragmentos generados en el subtask 11; ninguno se escribe a mano.

## Implementation plan

### Tarea 1 — §4 Experimentación y resultados

- [ ] Antes de escribir, leer los valores reales para redactar sobre ellos:

```bash
python -c "
import json
r = json.load(open('data/2026/resumen_2026.json', encoding='utf-8'))
print(f'{\"modelo\":26s} {\"tier\":7s} {\"estr\":>6s} {\"laxa\":>6s} {\"brecha\":>7s} {\"lat\":>7s}')
for f in r:
    print(f'{f[\"modelo\"]:26s} {f[\"tier\"]:7s} {f[\"exact_match_pct\"]:5.1f}% '
          f'{f[\"exact_match_laxo_pct\"]:5.1f}% '
          f'{f[\"exact_match_laxo_pct\"]-f[\"exact_match_pct\"]:6.1f}pp '
          f'{f[\"avg_latencia_s\"]:6.2f}s')
"
python -c "
import pandas as pd
print(pd.read_csv('data/2026/taxonomia_2026.csv').to_string(index=False))
print()
print(pd.read_csv('data/2026/exactitud_por_categoria.csv').to_string(index=False))
"
```

- [ ] Escribir `04_resultados.tex` con `\section{Experimentación y resultados}` y cuatro subsecciones:

**4.1 Resultados globales** — `\input{tablas/tabla2_resultados_globales}` y la Fig. 1:

```latex
\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{figuras/fig1_exactitud_latencia_2026.png}
\caption{Coincidencia exacta estricta y laxa (izquierda) y latencia promedio de
inferencia en CPU (derecha), por modelo, agrupados por tamaño.}
\label{fig:globales}
\end{figure}
```

Comentar: el mejor y el peor modelo, si la ventaja del tier 1–2B sobre sub-1B se sostiene, el costo en latencia, y **la brecha estricta/laxa** con su interpretación. Referirse a las tablas y figuras con `\ref{}`, nunca con números de tabla escritos a mano.

**4.2 Exactitud por campo del esquema** — Fig. 2 (`figuras/fig2_exactitud_por_campo_2026.png`, `\label{fig:campos}`). Señalar qué campo concentra los errores.

**4.3 Exactitud por categoría lingüística** — `\input{tablas/tabla3_por_categoria}`. Aclarar de entrada que **las categorías las asignó el juez automáticamente**, no un anotador humano, y comparar la distribución obtenida contra la del estudio anterior (8/8/6/4/3/3), discutiendo las diferencias si las hubo en vez de ocultarlas.

**4.4 Análisis automático de errores** — `\input{tablas/tabla4_taxonomia}`. Esta subsección **reemplaza** el análisis manual del original. Debe: (a) decir explícitamente que la clasificación es automática, hecha por el juez seleccionado, y nombrar cuál fue y con qué exactitud de etapa 1 ganó; (b) usar **siete** etiquetas, con `alucinacion_valor_unidad` y `valor_numerico_incorrecto` separadas; (c) reportar cuántas respuestas quedaron con `juez_parse_ok = False` y cómo se resolvieron; (d) cuantificar cuántas respuestas rescató la métrica laxa.

- [ ] Verificar que **no queda** ninguna mención a clasificación manual ni a "cuatro categorías":

```bash
grep -niE "manualmente|a mano|cuatro categor" paper/02_reescrito/secciones/04_resultados.tex \
  && echo "FALLA: quedó lenguaje del etiquetado manual" || echo "sin etiquetado manual OK"
```

### Tarea 2 — §5 Discusión

- [ ] Escribir `05_discusion.tex`. Ejes obligatorios:
  1. **Qué significa la brecha estricta/laxa.** Es la medida directa de cuánto penaliza la coincidencia exacta a respuestas semánticamente correctas — exactamente la duda que el estudio original no podía cuantificar porque su verificación era manual y su métrica única.
  2. **Endurecer el prompt no elimina el problema.** El protocolo declara los sinónimos violación de formato y aun así aparecen; discutir qué dice eso sobre la capacidad de estos modelos de respetar vocabularios cerrados.
  3. **Tamaño vs. desempeño vs. latencia**, con los dos tiers, y si algún sub-1B compite con los 1–2B.
  4. **Viabilidad práctica**: ¿alguno de los 12 sirve para domótica local en CPU con 2 núcleos? Responder con las latencias medidas.

  No incluir un eje sobre "los dos modelos base prompteados por raw completion": esa amenaza ya no
  aplica, porque los 12 modelos evaluados resuelven a `chat_template` (ver 3.3 y 4.4). No es una
  omisión, es una simplificación real del panorama de comparabilidad.

### Tarea 3 — §6 Amenazas a la validez, con las tres nuevas (RF16)

- [ ] Escribir `06_amenazas.tex` conservando los cuatro subtítulos del original (validez de constructo, interna, externa, de conclusión) e incorporando las **tres** amenazas nuevas, cada una nombrada explícitamente:

**(a) Sesgo de auto-favorecimiento del juez** — en validez interna. El juez es uno de los modelos evaluados y **también juzga sus propias salidas**; puede favorecerse. Declarar además que es un modelo de menos de 2B juzgando equivalencia semántica, con la capacidad limitada que eso implica, y respaldar con el número de fallos de parseo de §4.4. Decir qué mitigaría esto en trabajo futuro (juez externo más grande, o validación humana sobre una muestra).

**(b) Versiones divergentes de la librería de inferencia** — en validez interna, y es el punto que el usuario pidió explícitamente que llegue al paper. Los 12 modelos evaluados corren bajo **dos grupos de versión de `transformers`, mutuamente excluyentes** (grupo A resuelve 4.57.6, grupo B resuelve 5.14.1 — ver Tabla~\ref{tab:versiones}), lo cual es un confusor para la comparación de **latencia**: parte de la diferencia observada puede deberse a la versión y no al modelo. Nombrar los modelos que forzaron cada grupo: `granite-4.0-350m` necesita el grupo A (falla bajo 5.14.1); `LFM2.5-230M`, `LFM2.5-350M`, `Qwen3.5-0.8B` y `Qwen3.5-2B` necesitan el grupo B (fallan bajo 4.57.6). Señalar la evidencia más nítida: el grupo B resuelve exactamente a la versión que rompe al modelo del grupo A, de modo que ninguna versión mayor única cubre el roster. **No** repetir la premisa superada de que una única versión elimina este confusor.

**(c) Exclusión de modelos por acceso restringido no otorgado** — en validez externa. El diseño original consideraba 14 modelos; **2** (`gemma-3-270m-it`, `Llama-3.2-1B-Instruct`) quedaron fuera del roster evaluado porque el acceso de descarga no fue otorgado —repositorios `gated` con aprobación manual pendiente—, **no por una decisión metodológica**. Esto acota el alcance de la comparación a los 12 modelos que sí se pudieron evaluar; no se puede afirmar nada sobre el desempeño relativo de los dos excluidos. Redactar sin datos identificatorios ni mención de ningún token concreto (envío ciego).

- [ ] **No** incluir una amenaza de "incomparabilidad de los dos modelos base prompteados por raw completion": queda **eliminada** porque no aplica (los 12 modelos evaluados usan `chat_template`). Confirmar que no sobrevive ninguna mención a que `LFM2.5-230M` o `LFM2.5-350M` se hayan prompteado por raw completion.
- [ ] Conservar del original las amenazas que siguen vigentes: dataset chico y desbalanceado (32 comandos, 3–8 por categoría), prompt único no optimizado, texto limpio sin ruido de ASR, una sola variante dialectal, sin significancia estadística formal, y VM compartida para medir latencia.
- [ ] **Actualizar** la amenaza de validez de constructo del original: la penalización por sinónimos ya no es solo una advertencia, ahora está **cuantificada** por la métrica laxa. Debe quedar redactada como resuelta-parcialmente, no como pendiente.

### Tarea 4 — §7 Conclusiones y declaración sobre uso de IA

- [ ] Escribir `07_conclusiones.tex`: qué se aprendió sobre los 12 modelos evaluados, qué aporta el pipeline de verificación automática (reproducible y escalable a rosters grandes, a diferencia del etiquetado manual), y trabajo futuro — juez externo, dataset más grande y balanceado, ruido de ASR, más variantes dialectales, y reintentar la evaluación de los 2 modelos excluidos si su acceso se otorga.
- [ ] Escribir `08_declaracion_ia.tex` adaptando la del original: mantener la declaración de uso de IA en diseño, código, redacción y figuras, y **agregar** que en este trabajo un modelo de lenguaje cumple además un rol **metodológico** como juez automático, descripto en §3.5. Redactarla **sin datos identificatorios** ("los autores" en impersonal, nunca nombres).

### Tarea 5 — Build final, anonimato y documentación

- [ ] Regenerar tablas y figuras por si cambió algo, y recopiar las figuras:

```bash
python src/metrics_2026.py
python src/generate_figures_2026.py
python src/generate_tex_tables.py
cp figures/2026/fig1_exactitud_latencia_2026.png paper/02_reescrito/figuras/
cp figures/2026/fig2_exactitud_por_campo_2026.png paper/02_reescrito/figuras/
```

- [ ] Build limpio (borrando auxiliares para que la bibliografía se resuelva de cero):

```bash
(cd paper/02_reescrito && latexmk -C >/dev/null; \
 latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex)
```

- [ ] Confirmar que no quedan referencias ni citas rotas:

```bash
grep -iE "undefined (reference|citation)|LaTeX Warning: Reference" paper/02_reescrito/main.log \
  && echo "FALLA: hay referencias sin resolver" || echo "referencias resueltas OK"
```

- [ ] **Puerta de envío ciego**: `python scripts/check_anonimato.py paper/02_reescrito` → código 0. Si falla, corregir el `.tex` señalado y **volver a compilar antes de reintentar** (los metadatos viven en el PDF, no en el fuente).
- [ ] Actualizar `README.md` con una sección nueva sobre el estudio 2026: roster activo de 12 modelos (de un registro de 14, con 2 excluidos por acceso no otorgado), pipeline de dos etapas, cómo correr el barrido (`python docker/build_all.py`, `python docker/run_sweep.py`, ambos por defecto sobre el roster activo), cómo regenerar métricas, figuras y tablas, y cómo compilar los dos papers. **Sin datos identificatorios que puedan filtrarse al paper**; el README sí puede tener la URL del repo, el paper no.
- [ ] `pytest -q`
- [ ] `git add paper/02_reescrito/ README.md`
- [ ] `git commit -m "docs(paper): secciones 4-7, amenazas nuevas y build ciego del paper reescrito"`

## Verify

```bash
# 1. Suite verde y build de los DOS papers
pytest -q
for d in paper/01_original paper/02_reescrito; do
  (cd "$d" && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex >/dev/null) \
    && echo "$d compila" || echo "$d FALLA"
done

# 2. PUERTA DE ENVÍO CIEGO
python scripts/check_anonimato.py paper/02_reescrito
test $? -eq 0 && echo "ENVIO CIEGO OK"

# 3. Las nueve secciones tienen contenido real (ningún stub sobreviviente)
python -c "
from pathlib import Path
for p in sorted(Path('paper/02_reescrito/secciones').glob('*.tex')):
    n = len(p.read_text(encoding='utf-8').split())
    assert n > 60, f'{p.name} tiene solo {n} palabras: parece un stub'
    print(f'{p.name:32s} {n:5d} palabras')
"

# 4. RF16: las TRES amenazas nuevas están presentes, y la de raw completion NO sobrevive
python -c "
import re
from pathlib import Path
t = Path('paper/02_reescrito/secciones/06_amenazas.tex').read_text(encoding='utf-8').lower()
faltan = []
if not any(k in t for k in ('auto-favorec','autofavorec','sus propias salidas','juzga sus propias')):
    faltan.append('sesgo del juez')
if not any(k in t for k in ('versiones', 'transformers')) or not any(k in t for k in ('4.57', '5.14')):
    faltan.append('versiones divergentes de libreria (dos grupos)')
if not any(k in t for k in ('acceso restringido', 'gated')) or 'metodologica' not in t:
    faltan.append('exclusion por acceso restringido no otorgado')
assert not faltan, faltan
prohibido = re.search(r'(lfm2\.5-230m|lfm2\.5-350m|modelos base)[^.]{0,80}raw completion', t)
assert not prohibido, 'sobrevive la amenaza eliminada: incomparabilidad por raw completion'
print('las 3 amenazas nuevas (RF16) estan presentes, ninguna de raw completion')
"

# 5. §4.4 es automática: no quedó lenguaje de etiquetado manual
grep -niE "manualmente|clasificó a mano|cuatro categor" \
  paper/02_reescrito/secciones/04_resultados.tex \
  && echo "FALLA: lenguaje manual" || echo "4.4 automatica OK"

# 6. Ningún número de resultado escrito a mano: todas las tablas por \input
python -c "
from pathlib import Path
t = Path('paper/02_reescrito/secciones/04_resultados.tex').read_text(encoding='utf-8')
for n in ('tabla2_resultados_globales','tabla3_por_categoria','tabla4_taxonomia'):
    assert f'input{{tablas/{n}}}' in t.replace(' ',''), n
assert 'begin{tabular}' not in t, 'hay una tabla escrita a mano en 04_resultados.tex'
print('tablas 2/3/4 por input OK')
"

# 7. Regenerar tablas no produce diff (paper y datos sincronizados)
python src/generate_tex_tables.py >/dev/null
git diff --exit-code -- paper/02_reescrito/tablas/ && echo "paper sincronizado con los datos"

# 8. F0 intacto y docx sin trackear
git diff --exit-code main -- src/models.py src/prompt.py src/scoring.py src/schema.py \
  src/metrics.py src/run_evaluation.py src/generate_figures.py tests/test_metricas.py \
  data/resultados_experimento_detalle.csv data/resultados_experimento_resumen.json \
  data/dataset_comandos_domotica.csv && echo "F0 intacto"
git ls-files | grep -i docx && echo "FALLA: docx trackeado" || echo "docx sin trackear OK"

# 9. Al menos un commit por nodo del DAG (16, tras el Delta 02) mas G1 y G2 (18 nodos en total).
# No es un numero exacto: subtasks de alcance amplio (04, 16) abarcan varios commits, igual que
# ya ocurrio historicamente en 04 (99c13aa, dfb8315, d6c7581).
git rev-list --count main..HEAD   # -> >= 18
```

## Acceptance criteria

- **Dado** `paper/02_reescrito/`, **cuando** se corre `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` tras un `latexmk -C`, **entonces** termina con código 0, produce `main.pdf`, y `main.log` no reporta referencias ni citas sin resolver.
- **Dado** `python scripts/check_anonimato.py paper/02_reescrito`, **entonces** sale con **código 0** con el `main.pdf` final presente: sin autores, afiliaciones, correos, ORCID, URL de repositorio, agradecimientos, financiamiento ni autorreferencias, y sin metadatos identificatorios en el PDF.
- **Dado** las nueve secciones, **entonces** ninguna es un stub: todas superan las 60 palabras de contenido real.
- **Dado** §4.1, **entonces** incorpora la Tabla 2 y la Fig. 1 por `\input`/`\includegraphics`, se refiere a ellas con `\ref`, y comenta explícitamente la brecha entre exactitud estricta y laxa.
- **Dado** §4.3, **entonces** declara que las categorías lingüísticas las asignó el juez automáticamente y compara la distribución obtenida contra la referencia 8/8/6/4/3/3 del estudio anterior.
- **Dado** §4.4, **entonces** incorpora la Tabla 4 con **siete** etiquetas, nombra al juez seleccionado y su exactitud de etapa 1, reporta la cantidad de respuestas con `juez_parse_ok = False`, cuantifica cuántas rescató la métrica laxa, y **no contiene** las expresiones "manualmente", "clasificó a mano" ni "cuatro categorías".
- **Dado** §4, **entonces** no contiene ningún `\begin{tabular}` propio: las Tablas 2, 3 y 4 entran exclusivamente por `\input{tablas/...}`.
- **Dado** §6, **entonces** contiene las **tres** amenazas de RF16 —sesgo de auto-favorecimiento del juez (incluido que juzga sus propias salidas), versiones divergentes de librería como confusor de la latencia (dos grupos mutuamente excluyentes, con referencia a `tab:versiones` y a los modelos que forzaron cada grupo), y exclusión de 2 de los 14 modelos del diseño original por acceso restringido no otorgado (sin datos identificatorios ni mención de ningún token concreto)— **sin** ninguna mención a que `LFM2.5-230M`/`LFM2.5-350M` se hayan prompteado por raw completion, y conserva las amenazas vigentes del original, con la de validez de constructo **actualizada** a "cuantificada por la métrica laxa".
- **Dado** §8, **entonces** declara el uso de IA incluyendo su rol **metodológico** como juez automático, redactada en impersonal y sin datos identificatorios.
- **Dado** que se regeneran las tablas sin cambiar los datos, **entonces** `git diff -- paper/02_reescrito/tablas/` sale vacío: el paper y los datos están sincronizados.
- **Dado** el repositorio al final, **entonces** `pytest -q` pasa, los dos papers compilan, todos los archivos de F0 siguen byte-idénticos a `main`, `paper_cacic_LNCS_word.docx` sigue sin trackear, y `git rev-list --count main..HEAD` es al menos 18 (16 nodos del DAG + G1 + G2, con subtasks de alcance amplio como 04 y 16 aportando varios commits).
