# Rimasia-web

Buscador de rimas tematicas apoyado en word embeddings.

## Arranque rapido

```powershell
.\setup.ps1
.\.venv\Scripts\python.exe run.py
```

La web queda en `http://127.0.0.1:5000`.

## Como funciona ahora

1. Construye un lexico de palabras frecuentes en espanol.
2. Genera embeddings con el modelo `jinaai/jina-embeddings-v2-base-es`.
3. Filtra candidatas por rima consonante o asonante con `pyverse`.
4. Ordena esas candidatas por cercania vectorial a los conceptos introducidos.
5. Proyecta una muestra del espacio vectorial a 2D para dibujar el fondo tipo constelacion.

## Notas

- La primera ejecucion de `setup.ps1` descarga el modelo de embeddings y precalcula el indice vectorial.
- El cache se guarda en `.cache/`.
- Si una rima no tiene suficiente afinidad semantica, la interfaz puede mostrarla como apoyo por sonido.
