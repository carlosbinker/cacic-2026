# FIXES_1 — Defectos del tier `cheap` de TEST_PLAN.md

Contexto: se corrió el tier `cheap` de la regresión cross-task (`TEST_PLAN.md` §6). Dos
checks fallaban o daban un valor observado distinto del esperado. **Ambos son defectos del
check, no del producto**: no hay ningún token versionado, ningún `Dockerfile` viola F11 y el
barrido de Docker sigue construyendo exactamente las 12 imágenes del roster activo. No es un
hallazgo de seguridad — es una corrección de la instrumentación de test.

Recordatorio de F11 (TODO.md §4, "Docker — re-congelado"): está prohibido declarar
`ARG HF_TOKEN` / `ENV HF_TOKEN` en cualquier `Dockerfile`, pasar el token como `--build-arg`,
hacer `COPY .env`, escribirlo en una capa de imagen, o invocar `huggingface-cli login` /
`huggingface_hub.login()`. La única forma de consumirlo es `os.environ["HF_TOKEN"]` en
runtime. Ninguna de esas prohibiciones está violada en el árbol; lo que fallaba era el
*alcance* y el *conteo* de los checks que lo verifican.

---

## Defecto 1 — Check C4 se autofalla contra su propia documentación

**Tipo:** defecto de check (falso positivo). No es un hallazgo de seguridad.

### Symptom

El Check C4 ("El token nunca entra en una capa de imagen") corre dos `git grep` sobre **todo
el árbol trackeado**:

```bash
git grep -nE 'ARG +HF_TOKEN|ENV +HF_TOKEN|COPY +\.env|--build-arg[= ]*HF_TOKEN'
git grep -niE 'huggingface-cli login|huggingface_hub\.login\('
```

Ambos greps encontraban coincidencias y ninguna línea `... OK` se imprimía. El check
reportaba fallo.

### Root cause

Los patrones prohibidos también aparecen, textualmente, en los archivos que **documentan la
prohibición**: `TODO.md`, `TODO_03_harness-barrido-reanudable.md`,
`TODO_04_docker-por-modelo.md`, el propio `TEST_PLAN.md` (el bloque `Run` del check cita los
patrones) y los mensajes de assert de `tests/test_docker_matriz.py`. Además, el docstring de
`src/run_sweep_2026.py::_credencial_hf` citaba literalmente `huggingface_hub.login()` para
decir "esto nunca se invoca". Ninguno de estos archivos es un `Dockerfile` ni termina en una
capa de imagen (excepto `src/`, que sí se copia a la imagen vía
`COPY src/ /app/src/` en `docker/Dockerfile.modelo` — ver más abajo). El check tenía el
alcance correcto en intención ("que el token no entre en una capa de imagen") pero el alcance
implementado era el árbol completo, así que escribir la prohibición en prosa o en un mensaje
de assert hacía fallar al check que la documenta.

### Fix

1. **`TEST_PLAN.md`, Check C4**: acotar el `git grep` con pathspec a los archivos que
   realmente pueden terminar en una capa de imagen: `-- '*Dockerfile*' 'docker/**' 'src/**'
   'scripts/**'`. Se agregó un comentario en el bloque `Run` explicando por qué el alcance
   está acotado (para que nadie lo vuelva a ampliar al árbol completo). El "Covers AC", el
   cost tier (`cheap`) y el texto de "On failure indicates" se preservaron sin cambios; solo
   cambió el alcance.
2. **`src/run_sweep_2026.py`**: el docstring de `_credencial_hf` citaba literalmente
   `huggingface_hub.login()`. Como `src/` se copia a la imagen, ese archivo sí está dentro del
   nuevo alcance y el match era real (aunque inocuo: es un comentario, no una llamada). Se
   reformuló la frase para transmitir el mismo significado sin el patrón literal ("Nunca se
   hace login interactivo de huggingface_hub" en vez de citar la API). No cambia
   comportamiento; no está en F0.
3. **Cierre del hueco de cobertura**: el check manual acotado ya no mira `*.md` ni `tests/`,
   así que la garantía "ningún `Dockerfile` en TODO el árbol declara/copia el token" debía
   quedar cubierta en otro lado. `tests/test_docker_matriz.py::test_no_hay_credenciales_en_capas_ni_login_interactivo`
   ya existía pero solo iteraba `(RAIZ / "docker").rglob("*")` — no cubre el `Dockerfile`
   legacy de la raíz del repo (`git ls-files` lista `Dockerfile` y `docker/Dockerfile.modelo`).
   Se agregó `test_ningun_dockerfile_trackeado_tiene_credenciales_ni_login`, que itera
   `git ls-files '*Dockerfile*'` (cualquier ubicación) y assert-ea las mismas cinco
   prohibiciones. No se debilitó ningún assert existente.

### Files

- `TEST_PLAN.md` (Check C4)
- `src/run_sweep_2026.py` (docstring de `_credencial_hf`, no es F0)
- `tests/test_docker_matriz.py` (test nuevo `test_ningun_dockerfile_trackeado_tiene_credenciales_ni_login`)

### Verification command and expected output

```bash
git grep -nE 'ARG +HF_TOKEN|ENV +HF_TOKEN|COPY +\.env|--build-arg[= ]*HF_TOKEN' -- '*Dockerfile*' 'docker/**' 'src/**' 'scripts/**' ; test $? -eq 1 && echo "sin credenciales en capas OK"
git grep -niE 'huggingface-cli login|huggingface_hub\.login\(' -- '*Dockerfile*' 'docker/**' 'src/**' 'scripts/**' ; test $? -eq 1 && echo "sin login interactivo OK"
```

Salida real obtenida:

```
sin credenciales en capas OK
sin login interactivo OK
```

### Acceptance criterion

Dado el árbol trackeado actual, cuando se corren los dos `git grep` acotados del Check C4,
entonces ambos terminan con exit 1 (sin coincidencias) e imprimen `sin credenciales en capas
OK` y `sin login interactivo OK`; y `tests/test_docker_matriz.py::test_ningun_dockerfile_trackeado_tiene_credenciales_ni_login`
pasa iterando **todos** los Dockerfiles trackeados (`Dockerfile` y
`docker/Dockerfile.modelo`), no solo los de `docker/`.

---

## Defecto 2 — NF1 cuenta 24 líneas donde hay 12 imágenes (cosmético)

**Tipo:** defecto de check (conteo ambiguo). No es un hallazgo de seguridad ni un bug de build.

### Symptom

La fila de §6 del tier `cheap` para C6+NF1 espera `2, 12, 12, 0, sin GPU OK, 12`. El último
número —la cantidad de imágenes del plan de build, medido con
`python docker/build_all.py --dry-run | grep -c 'slm-domotica-2026:'`— observaba **24**, no
12.

### Root cause

`docker/build_all.py::construir` imprime cada tag de imagen **dos veces** por `--dry-run`:
una vez en el header (`print(f"\n=== build {tag_imagen(modelo)} | {modelo.transformers_pin}
===")`, línea 64) y otra vez en la línea del comando (`print(" ".join(cmd))`, línea 65, que
incluye `-t slm-domotica-2026:<slug>`). `grep -c 'slm-domotica-2026:'` cuenta **líneas que
mencionan el tag**, no imágenes distintas: con 12 modelos en `roster_activo()`, eso da 24
líneas. Las imágenes distintas siguen siendo 12 (confirmado con `sort -u`).

### Fix

Se cambió el comando de medición en `TEST_PLAN.md` (NF1, sección "Non-functional checks") de
contar líneas a contar **tags distintos**:

```bash
python docker/build_all.py --dry-run | grep -oE 'slm-domotica-2026:[A-Za-z0-9._-]+' | sort -u | wc -l
```

Se agregó un comentario explicando por qué (`--dry-run` imprime cada tag dos veces) para que
nadie vuelva a "arreglar" esto bajando la expectativa a 24. Se actualizó también la
descripción de la fila 4 de la tabla de regresión en §6 para decir explícitamente "doce
imágenes **distintas**", consistente con el cuerpo del check. El valor esperado numérico (12)
no cambió porque ya era el correcto — solo el comando que lo produce.

### Files

- `TEST_PLAN.md` (sección NF1 y fila 4 de la tabla del §6)

### Verification command and expected output

```bash
python docker/run_sweep.py --dry-run | grep -cE 'gpus|--device|nvidia' ; test $? -eq 1 && echo "sin GPU OK"
python docker/build_all.py --dry-run | grep -oE 'slm-domotica-2026:[A-Za-z0-9._-]+' | sort -u | wc -l
```

Salida real obtenida:

```
0
sin GPU OK
12
```

(`--dry-run` en ambos scripts: no se invocó ningún comando `docker` real.)

### Acceptance criterion

Dado `python docker/build_all.py --dry-run`, cuando se cuentan los tags de imagen distintos
con `grep -oE ... | sort -u | wc -l`, entonces el resultado es exactamente `12` — igual al
tamaño de `roster_activo()` — sin depender de que cada tag aparezca una cantidad fija de veces
en la salida del script.

---

## Verificaciones finales requeridas

1. **C4 reparado** (ver Defecto 1): `sin credenciales en capas OK` / `sin login interactivo
   OK` — ambas líneas impresas.
2. **NF1 reparado** (ver Defecto 2): `12`.
3. **`python -m pytest -q`**: `109 passed, 4 skipped in 4.51s`.
4. **`git grep -iE 'hf_[A-Za-z0-9]{20,}'`**: exit 1, sin salida (sigue vacío después de los
   cambios).

Ningún valor de token está presente en el árbol y ningún `Dockerfile` viola F11: ambos
defectos son de la instrumentación de test, no del producto.
