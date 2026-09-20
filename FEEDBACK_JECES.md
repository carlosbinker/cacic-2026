CACIC 2026 cacic@lidi.info.unlp.edu.ar a través de ingunlamedu.onmicrosoft.com 
mar, 8 sept, 15:35 (hace 10 días)
para cbinker, laulasorsa, htantignone, gburanits, eazurdo, mfrattini

Estimado Colega:
Su artículo 16775 - "Modelos de Lenguaje Pequeños para la Interpretación de Comandos de Domótica en Español: Evaluación Automática de Doce Modelos Sub-2B" ha sido Aceptado para su exposición y publicación en CACIC 2026 (Argentina).
Proximamente se abrira un periodo de recepcion de CAMERA READY, donde podra enviar una nueva version del articulo teniendo en cuenta las observaciones de los evaluadores. (si correspondiese)
Se requiere la inscripción al menos de 1 autor en el Congreso para su publicación en el CACIC 2026.

Fecha límite de Inscripción de autores 03/10/2026
Mas información en el sitio del evento
Saludos cordiales,
Comité Coordinador WS
CACIC 2026
Evaluación número 1:
IMPORTANCIA DEL TEMA
Interés Actual
Relevancia
Aplicabilidad
Bibliografía: Significativa
CONTRIBUCIÓN
Originalidad
Claridad del Resumen y Texto
Aporte de las Conclusiones
Calidad del Trabajo Experimental: Posible
CALIDAD DE PRESENTACIÓN
Organización
Claridad
Legibilidad
Calidad de Figuras y Tablas
Sintaxis: Buena
SELECCIÓN DE MEJORES TRABAJOS
¿Elegible Para Publicar Entre Los Mejores Trabajos?: No
Recomendación parcial: Aceptar (Weak Accept)
Calificación global: 7
Conocimiento del tema por el evaluador: 8
Comentarios: Este trabajo presenta una evaluación reproducible de doce SLMs abiertos sub-2B sobre la interpretación de comandos de domótica en español rioplatense, con un protocolo de dos etapas y decodificación determinista que elimina el etiquetado manual. La contribución es valiosa por centrarse en una variedad dialectal escasamente cubierta y por el diseño cuidadoso del entorno de ejecución y la reproducibilidad. Una fortaleza destacable es la evaluación de los modelos, que incluye un análisis de tiempos de latencia muy completo y directamente relevante para el despliegue en el borde: las comparaciones de pares bajo la misma topología y la discusión sobre el costo de la atención híbrida en CPU aportan conclusiones prácticas y bien fundamentadas para la selección de modelos en hubs domóticos. Como principal observación, la validación sobre un dataset de solo 32 comandos no justifica el uso de un modelo como juez. Dado que la referencia es conocida y acotada a cinco campos categóricos, la comparación determinista contra el ground truth ya provee la respuesta correcta; un juez de lenguaje añade incertidumbre sin beneficio adicional en un escenario donde la verdad es explícita y fácilmente contrastable. Sería preferible emplear directamente esa referencia conocida en lugar de delegar la decisión semántica a un modelo evaluado, especialmente cuando el juez es a su vez uno de los modelos bajo estudio. Finalmente, el manuscrito presenta varios errores tipográficos y de estilo que dan la impresión de haber sido generados por una herramienta de IA sin revisión cuidadosa. Se recomienda una corrección exhaustiva para la versión final. En conjunto, el artículo ofrece una evaluación pertinente y bien temporalizada de un conjunto de modelos poco explorados, y con una revisión de los puntos anteriores resulta adecuado para su aceptación tras revisiones menores.
Evaluación número 2:
IMPORTANCIA DEL TEMA
Interés Actual
Relevancia
Aplicabilidad
Bibliografía: Excelente
CONTRIBUCIÓN
Originalidad
Claridad del Resumen y Texto
Aporte de las Conclusiones
Calidad del Trabajo Experimental: Significativa
CALIDAD DE PRESENTACIÓN
Organización
Claridad
Legibilidad
Calidad de Figuras y Tablas
Sintaxis: Muy Buena
SELECCIÓN DE MEJORES TRABAJOS
¿Elegible Para Publicar Entre Los Mejores Trabajos?: Si
Recomendación parcial: Aceptar (Strong Accept)
Calificación global: 8
Conocimiento del tema por el evaluador: 9
Comentarios: El trabajo presenta una evaluación comparativa de doce Small Language Models sub-2B para la interpretación estructurada de comandos de domótica en español rioplatense. La temática es muy actual y relevante, particularmente por el creciente interés en modelos pequeños ejecutables localmente y por la escasez de evaluaciones específicas para variantes lingüísticas regionales. Se destacan la claridad y reproducibilidad del protocolo experimental, la comparación bajo un prompt común, la diferenciación entre exactitud estricta y semántica y el análisis explícito de las amenazas a la validez. Los resultados aportan evidencia interesante respecto de que un mayor número de parámetros no necesariamente implica mejor desempeño en esta tarea. Como principal limitación, el dataset de solo 32 comandos resulta reducido para sustentar comparaciones amplias entre doce modelos y algunas categorías lingüísticas quedan escasamente representadas. Asimismo, sería recomendable validar mediante evaluación humana o un juez externo una muestra de las decisiones del LLM utilizado como juez, especialmente dado que dicho modelo pertenece también al conjunto evaluado. Finalmente, debería enfatizarse que el experimento evalúa interpretación de texto limpio y no todavía el pipeline completo de un sistema domótico basado en voz. En conjunto, se considera que es un trabajo bien presentado, reproducible, actual y con resultados de interés, cuyas limitaciones están además correctamente reconocidas por los autores.