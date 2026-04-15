from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from uuid import uuid4

from flask import Flask, jsonify, render_template, request

from rhyme_engine import (
    DEFAULT_RESULTS_PER_GROUP,
    MAX_RESULTS_PER_GROUP,
    MIN_RESULTS_PER_GROUP,
    SearchOutcome,
    get_constellation_payload,
    get_embedding_metadata,
    search_rhymes,
)


app = Flask(__name__)
app.config.update(
    TEMPLATES_AUTO_RELOAD=True,
    SEND_FILE_MAX_AGE_DEFAULT=0,
)

STATIC_DIR = Path(__file__).with_name('static')
SEARCH_JOBS: dict[str, dict] = {}
SEARCH_JOB_LOCK = threading.Lock()
SEARCH_JOB_TTL_SECONDS = 900


def asset_version(filename: str) -> int:
    try:
        return int((STATIC_DIR / filename).stat().st_mtime)
    except OSError:
        return 0


def base_context() -> dict:
    context = {
        'palabra': '',
        'temas': '',
        'conceptos': (),
        'consonante': (),
        'asonante': (),
        'error_message': None,
        'warning_message': None,
        'result_limit': DEFAULT_RESULTS_PER_GROUP,
        'result_page': 0,
        'page_count': 1,
        'consonant_total': 0,
        'assonant_total': 0,
        'graph_data': {'nodes': [], 'edges': []},
        'highlight_words': (),
        'query_constellation': {'nodes': [], 'edges': []},
        'embedding_metadata': {
            'model_name': '',
            'word_count': 0,
            'word_count_label': '0',
        },
        'asset_versions': {
            'index.css': asset_version('index.css'),
            'constellation.js': asset_version('constellation.js'),
            'query-form.js': asset_version('query-form.js'),
            'space-audio.js': asset_version('space-audio.js'),
            'sonido_espacio.mp3': asset_version('sonido_espacio.mp3'),
            'favicon.svg': asset_version('favicon.svg'),
        },
    }

    try:
        context['graph_data'] = get_constellation_payload()
        metadata = get_embedding_metadata()
        context['embedding_metadata'] = {
            **metadata,
            'word_count_label': f"{int(metadata['word_count']):,}".replace(',', '.'),
        }
    except Exception as exc:
        context['error_message'] = (
            'No se ha podido inicializar el indice de embeddings. '
            f'Detalle: {exc}'
        )

    return context


def build_highlight_words(palabra: str, conceptos: tuple[str, ...], outcome: SearchOutcome) -> tuple[str, ...]:
    del palabra, conceptos, outcome
    return ()


def parse_result_limit(raw_value: str | None) -> int:
    try:
        parsed = int((raw_value or '').strip())
    except (TypeError, ValueError, AttributeError):
        parsed = DEFAULT_RESULTS_PER_GROUP

    return max(MIN_RESULTS_PER_GROUP, min(MAX_RESULTS_PER_GROUP, parsed))


def parse_result_page(raw_value: str | None) -> int:
    try:
        parsed = int((raw_value or '').strip())
    except (TypeError, ValueError, AttributeError):
        parsed = 0

    return max(0, parsed)


def parse_query_input(source) -> tuple[str, str, tuple[str, ...], int, int, str | None]:
    palabra = source.get('palabra', '').strip().lower()
    temas_text = source.get('temas', '').strip()
    conceptos = tuple(tema.strip().lower() for tema in temas_text.split(',') if tema.strip())
    result_limit = parse_result_limit(source.get('cantidad'))
    result_page = parse_result_page(source.get('pagina'))

    if not palabra:
        return palabra, temas_text, conceptos, result_limit, result_page, 'Introduce una palabra objetivo.'
    if ' ' in palabra:
        return palabra, temas_text, conceptos, result_limit, result_page, 'La palabra objetivo debe ser una sola palabra.'
    if not conceptos:
        return palabra, temas_text, conceptos, result_limit, result_page, 'Introduce uno o varios conceptos separados por comas.'

    return palabra, temas_text, conceptos, result_limit, result_page, None


def apply_outcome_to_context(
    context: dict,
    palabra: str,
    temas_text: str,
    conceptos: tuple[str, ...],
    result_limit: int,
    outcome: SearchOutcome,
) -> dict:
    context['palabra'] = palabra
    context['temas'] = temas_text
    context['conceptos'] = conceptos
    context['result_limit'] = result_limit
    context['result_page'] = outcome.result_page
    context['page_count'] = outcome.page_count
    context['consonante'] = outcome.consonant
    context['asonante'] = outcome.assonant
    context['consonant_total'] = outcome.consonant_total
    context['assonant_total'] = outcome.assonant_total
    context['warning_message'] = outcome.warning_message
    context['highlight_words'] = build_highlight_words(palabra, conceptos, outcome)
    context['query_constellation'] = outcome.query_constellation
    return context


def serialize_outcome(
    palabra: str,
    temas_text: str,
    conceptos: tuple[str, ...],
    result_limit: int,
    outcome: SearchOutcome,
) -> dict:
    return {
        'palabra': palabra,
        'temas': temas_text,
        'conceptos': list(conceptos),
        'result_limit': result_limit,
        'result_page': outcome.result_page,
        'page_count': outcome.page_count,
        'consonant_total': outcome.consonant_total,
        'assonant_total': outcome.assonant_total,
        'warning_message': outcome.warning_message,
        'highlight_words': list(build_highlight_words(palabra, conceptos, outcome)),
        'query_constellation': outcome.query_constellation,
    }


def prune_search_jobs() -> None:
    cutoff = time.time() - SEARCH_JOB_TTL_SECONDS
    with SEARCH_JOB_LOCK:
        expired = [
            job_id
            for job_id, payload in SEARCH_JOBS.items()
            if float(payload.get('updated_at', 0.0)) < cutoff
        ]
        for job_id in expired:
            del SEARCH_JOBS[job_id]


def update_search_job(job_id: str, **changes) -> None:
    with SEARCH_JOB_LOCK:
        job = SEARCH_JOBS.get(job_id)
        if job is None:
            return
        job.update(changes)
        job['updated_at'] = time.time()


def start_search_job(
    palabra: str,
    temas_text: str,
    conceptos: tuple[str, ...],
    result_limit: int,
    result_page: int,
) -> str:
    prune_search_jobs()
    job_id = uuid4().hex
    now = time.time()

    with SEARCH_JOB_LOCK:
        SEARCH_JOBS[job_id] = {
            'status': 'queued',
            'progress': 0.03,
            'message': 'Consulta en cola',
            'result': None,
            'error_message': None,
            'created_at': now,
            'updated_at': now,
        }

    def worker() -> None:
        def progress_callback(progress: float, message: str) -> None:
            update_search_job(
                job_id,
                status='running',
                progress=round(progress, 4),
                message=message,
            )

        try:
            outcome = search_rhymes(
                palabra,
                list(conceptos),
                progress_callback=progress_callback,
                result_limit=result_limit,
                result_page=result_page,
            )
            update_search_job(
                job_id,
                status='complete',
                progress=1.0,
                message='Listo',
                result=serialize_outcome(palabra, temas_text, conceptos, result_limit, outcome),
            )
        except Exception as exc:
            update_search_job(
                job_id,
                status='error',
                progress=1.0,
                message='Error',
                error_message=(
                    'Ha fallado el motor de embeddings para la busqueda de rimas. '
                    f'Detalle: {exc}'
                ),
            )

    threading.Thread(target=worker, daemon=True).start()
    return job_id


@app.route('/')
@app.route('/home')
def home():
    return render_template('index.html', **base_context())


@app.route('/api/search/start', methods=['POST'])
def api_search_start():
    palabra, temas_text, conceptos, result_limit, result_page, error_message = parse_query_input(request.form)
    if error_message:
        return jsonify({'ok': False, 'error_message': error_message}), 400

    job_id = start_search_job(palabra, temas_text, conceptos, result_limit, result_page)
    return jsonify({'ok': True, 'job_id': job_id}), 202


@app.route('/api/search/status/<job_id>', methods=['GET'])
def api_search_status(job_id: str):
    with SEARCH_JOB_LOCK:
        job = SEARCH_JOBS.get(job_id)
        if job is None:
            return jsonify({'ok': False, 'error_message': 'La consulta ya no esta disponible.'}), 404

        payload = {
            'ok': True,
            'status': job['status'],
            'progress': job['progress'],
            'message': job['message'],
            'error_message': job['error_message'],
            'result': job['result'],
        }

    return jsonify(payload)


@app.route('/result', methods=['POST'])
def result():
    palabra, temas_text, conceptos, result_limit, result_page, error_message = parse_query_input(request.form)

    context = base_context()
    context['palabra'] = palabra
    context['temas'] = temas_text
    context['conceptos'] = conceptos
    context['result_limit'] = result_limit
    context['result_page'] = result_page

    if error_message:
        context['error_message'] = error_message
        return render_template('index.html', **context)

    try:
        outcome = search_rhymes(
            palabra,
            list(conceptos),
            result_limit=result_limit,
            result_page=result_page,
        )
    except Exception as exc:
        context['error_message'] = (
            'Ha fallado el motor de embeddings para la busqueda de rimas. '
            f'Detalle: {exc}'
        )
        return render_template('index.html', **context)

    apply_outcome_to_context(context, palabra, temas_text, conceptos, result_limit, outcome)
    return render_template('index.html', **context)


def main():
    app.run(
        host=os.environ.get('HOST', '127.0.0.1'),
        port=int(os.environ.get('PORT', '5000')),
        debug=os.environ.get('FLASK_DEBUG', '1') == '1',
    )


if __name__ == '__main__':
    main()
