from __future__ import annotations

import io
import json
import re
import unicodedata
import warnings
from contextlib import redirect_stdout
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable

import numpy as np
from fastembed import TextEmbedding
from pyverse import Pyverse
from wordfreq import top_n_list, zipf_frequency


BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / '.cache'
FASTEMBED_CACHE_DIR = CACHE_DIR / 'fastembed'
MODEL_NAME = 'jinaai/jina-embeddings-v2-base-es'
MODEL_SLUG = MODEL_NAME.replace('/', '__').replace('-', '_')
RHYME_INDEX_VERSION = 2
LEXICON_SIZE = 12000
GRAPH_LAYOUT_VERSION = 5
GRAPH_NODE_COUNT = 2000
GRAPH_BASIS_SAMPLE_COUNT = 900
GRAPH_EDGE_WINDOW = 2
GRAPH_EDGE_MAX_DISTANCE = 0.04
MIN_RESULTS_PER_GROUP = 2
DEFAULT_RESULTS_PER_GROUP = 6
MAX_RESULTS_PER_GROUP = 10
MATCH_THRESHOLD = 0.17
SEMANTIC_FALLBACK_THRESHOLD = 0.08
DIRECT_QUERY_MATCH_THRESHOLD = 0.1
WORD_RE = re.compile(r'^[a-z]+$')
STOPWORDS = {
    'ante', 'bajo', 'cabe', 'cada', 'como', 'contra', 'desde', 'donde', 'entre',
    'hacia', 'hasta', 'para', 'porque', 'sobre', 'tras', 'cuando', 'aunque', 'mismo',
    'misma', 'mismos', 'mismas', 'estas', 'estos', 'estar', 'seria', 'fuera', 'ellos',
    'ellas', 'aquel', 'aquella', 'aquellos', 'aquellas', 'dicho', 'dicha', 'dichos',
    'dichas', 'tener', 'hacer', 'puede', 'pueden', 'entre', 'alguna', 'alguno',
    'algunas', 'algunos', 'porque', 'quien', 'quienes', 'nunca', 'siempre', 'mucho',
    'mucha', 'muchos', 'muchas', 'poco', 'poca', 'pocos', 'pocas', 'mismo', 'misma',
    'mismos', 'mismas', 'todas', 'todos', 'toda', 'todo'
}
INDEX_CACHE_FILE = CACHE_DIR / f'embedding_index_v{RHYME_INDEX_VERSION}_{MODEL_SLUG}_{LEXICON_SIZE}.npz'
LEGACY_INDEX_CACHE_FILES = (
    CACHE_DIR / f'embedding_index_{MODEL_SLUG}_{LEXICON_SIZE}.npz',
)
GRAPH_CACHE_FILE = CACHE_DIR / f'embedding_graph_v{GRAPH_LAYOUT_VERSION}_{MODEL_SLUG}_{LEXICON_SIZE}.json'


@dataclass(frozen=True)
class RankedWord:
    word: str
    score: float
    matched_concepts: tuple[str, ...]
    is_fallback: bool = False


@dataclass(frozen=True)
class SearchOutcome:
    consonant: tuple[RankedWord, ...]
    assonant: tuple[RankedWord, ...]
    consonant_total: int
    assonant_total: int
    result_page: int
    page_count: int
    warning_message: str | None
    query_constellation: dict


_MODEL: TextEmbedding | None = None
_WORDS: np.ndarray | None = None
_DISPLAY_WORDS: np.ndarray | None = None
_EMBEDDINGS: np.ndarray | None = None
_FREQUENCIES: np.ndarray | None = None
_CONSONANT_INDEX: dict[str, np.ndarray] | None = None
_ASSONANT_INDEX: dict[str, np.ndarray] | None = None
_GRAPH_PAYLOAD: dict | None = None
_DISPLAY_LOOKUP: dict[str, str] | None = None


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize('NFD', text.lower().strip())
    normalized = ''.join(
        char for char in normalized if unicodedata.category(char) != 'Mn'
    )
    return re.sub(r'\s+', ' ', normalized)


def normalize_word(word: str) -> str:
    return normalize_text(word).replace('_', ' ')


def prepare_display_word(word: str) -> str:
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFC', word.strip().lower())).replace('_', ' ')


def has_explicit_accent(word: str) -> bool:
    return any(char in 'áéíóúÁÉÍÓÚ' for char in word)


def normalize_rhyme_code(code: str) -> str:
    return ''.join(char for char in normalize_text(code) if char.isalpha())


def is_valid_word(word: str) -> bool:
    return len(word) >= 4 and WORD_RE.fullmatch(word) is not None


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector) + 1e-12
    return vector / norm


def normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12
    return matrix / norms


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def report_progress(
    callback: Callable[[float, str], None] | None,
    progress: float,
    message: str,
) -> None:
    if callback is not None:
        callback(clamp(progress, 0.0, 1.0), message)


def get_model() -> TextEmbedding:
    global _MODEL

    if _MODEL is not None:
        return _MODEL

    CACHE_DIR.mkdir(exist_ok=True)
    FASTEMBED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings(
        'ignore',
        message='The model .* now uses mean pooling instead of CLS embedding.*',
    )
    _MODEL = TextEmbedding(
        model_name=MODEL_NAME,
        cache_dir=str(FASTEMBED_CACHE_DIR),
    )
    return _MODEL


@lru_cache(maxsize=50000)
def rhyme_signature(word: str) -> tuple[str, str]:
    with redirect_stdout(io.StringIO()):
        verse = Pyverse(prepare_display_word(word))

    return (
        normalize_rhyme_code(verse.consonant_rhyme),
        normalize_rhyme_code(verse.assonant_rhyme),
    )


def resolve_rhyme_surface(word: str) -> str:
    prepared_word = prepare_display_word(word)
    normalized_word = normalize_word(prepared_word)
    if has_explicit_accent(prepared_word):
        return prepared_word
    if _DISPLAY_LOOKUP is not None and normalized_word in _DISPLAY_LOOKUP:
        return _DISPLAY_LOOKUP[normalized_word]
    return prepared_word


def select_lexicon() -> tuple[list[str], list[str], np.ndarray]:
    words: list[str] = []
    display_words: list[str] = []
    frequencies: list[float] = []
    seen = set()

    for raw_word in top_n_list('es', LEXICON_SIZE):
        display_word = prepare_display_word(raw_word)
        word = normalize_word(display_word)
        if ' ' in word or not is_valid_word(word):
            continue
        if word in seen:
            continue

        seen.add(word)
        words.append(word)
        display_words.append(display_word)
        frequencies.append(max(zipf_frequency(display_word, 'es'), zipf_frequency(word, 'es'), 0.0))

    return words, display_words, np.asarray(frequencies, dtype=np.float32)


def restore_display_words(words: np.ndarray) -> np.ndarray:
    lookup: dict[str, str] = {}
    seen = set()

    for raw_word in top_n_list('es', LEXICON_SIZE):
        display_word = prepare_display_word(raw_word)
        normalized_word = normalize_word(display_word)
        if ' ' in normalized_word or not is_valid_word(normalized_word):
            continue
        if normalized_word in seen:
            continue

        seen.add(normalized_word)
        lookup[normalized_word] = display_word

    return np.asarray(
        [lookup.get(str(word), str(word)) for word in words.tolist()],
        dtype=words.dtype,
    )


def build_lookup_index(codes: np.ndarray) -> dict[str, np.ndarray]:
    lookup: dict[str, list[int]] = {}
    for position, code in enumerate(codes.tolist()):
        if not code:
            continue
        lookup.setdefault(code, []).append(position)
    return {
        code: np.asarray(indices, dtype=np.int32)
        for code, indices in lookup.items()
    }


def build_constellation_payload(
    words: np.ndarray,
    display_words: np.ndarray,
    embeddings: np.ndarray,
    frequencies: np.ndarray,
) -> dict:
    eligible = [
        index
        for index, word in enumerate(words.tolist())
        if 4 <= len(word) <= 12 and word not in STOPWORDS
    ]

    if not eligible:
        return {'nodes': [], 'edges': []}

    step = max(1, len(eligible) // GRAPH_NODE_COUNT)
    sampled = eligible[::step][:GRAPH_NODE_COUNT]
    if len(sampled) < GRAPH_NODE_COUNT:
        sampled = eligible[:GRAPH_NODE_COUNT]

    selected = np.asarray(sampled, dtype=np.int32)
    basis_step = max(1, len(selected) // GRAPH_BASIS_SAMPLE_COUNT)
    basis_indices = selected[::basis_step][:GRAPH_BASIS_SAMPLE_COUNT]
    if len(basis_indices) < 2:
        basis_indices = selected[: min(len(selected), 2)]

    sample_vectors = embeddings[selected]
    basis_vectors = embeddings[basis_indices]
    basis_center = basis_vectors.mean(axis=0, keepdims=True)
    centered_basis = basis_vectors - basis_center

    try:
        _, _, vt = np.linalg.svd(centered_basis, full_matrices=False)
        axes = vt[:2]
        if axes.shape[0] < 2:
            coords = (sample_vectors - basis_center)[:, :2]
        else:
            coords = (sample_vectors - basis_center) @ axes.T
    except np.linalg.LinAlgError:
        coords = (sample_vectors - basis_center)[:, :2]

    if coords.shape[1] == 1:
        coords = np.column_stack(
            (coords[:, 0], np.zeros(coords.shape[0], dtype=np.float32))
        )

    min_values = coords.min(axis=0)
    max_values = coords.max(axis=0)
    span = np.where((max_values - min_values) < 1e-9, 1.0, max_values - min_values)
    coords = (coords - min_values) / span

    nodes = []
    sizes = np.clip(1.05 + (frequencies[selected] - 2.7) * 0.28, 0.82, 2.35)
    for local_index, word_index in enumerate(selected.tolist()):
        nodes.append(
            {
                'id': local_index,
                'word': display_words[word_index],
                'x': round(float(0.08 + coords[local_index, 0] * 0.84), 6),
                'y': round(float(0.1 + coords[local_index, 1] * 0.8), 6),
                'size': round(float(sizes[local_index]), 4),
                'label': True,
            }
        )

    edges = {}
    order_x = np.argsort(coords[:, 0])
    order_y = np.argsort(coords[:, 1])
    candidates = [set() for _ in range(len(selected))]

    for ordering in (order_x, order_y):
        for position, source in enumerate(ordering.tolist()):
            left = max(0, position - GRAPH_EDGE_WINDOW)
            right = min(len(ordering), position + GRAPH_EDGE_WINDOW + 1)
            for candidate_position in range(left, right):
                if candidate_position == position:
                    continue
                candidates[source].add(int(ordering[candidate_position]))

    degrees = np.zeros(len(selected), dtype=np.int16)
    for source in range(len(selected)):
        source_candidates: list[tuple[float, int]] = []
        for target in candidates[source]:
            dx = float(coords[source, 0] - coords[target, 0])
            dy = float(coords[source, 1] - coords[target, 1])
            distance = float(np.hypot(dx, dy))
            if distance <= GRAPH_EDGE_MAX_DISTANCE:
                source_candidates.append((distance, target))

        source_candidates.sort(key=lambda item: item[0])
        edges_added = 0
        for distance, target in source_candidates:
            if edges_added >= 1:
                break
            if degrees[source] >= 2 or degrees[target] >= 3:
                continue

            left, right = sorted((int(source), int(target)))
            strength = round(max(0.18, 1.0 - distance / GRAPH_EDGE_MAX_DISTANCE), 4)
            key = (left, right)
            current = edges.get(key)
            if current is None or strength > current['strength']:
                edges[key] = {
                    'source': left,
                    'target': right,
                    'strength': strength,
                }
                degrees[source] += 1
                degrees[target] += 1
                edges_added += 1

    return {
        'nodes': nodes,
        'edges': list(edges.values()),
    }


def embed_documents(documents: list[str]) -> np.ndarray:
    model = get_model()
    matrix = np.asarray(list(model.embed(documents, batch_size=64)), dtype=np.float32)
    return normalize_matrix(matrix)


@lru_cache(maxsize=256)
def build_query_bundle(concepts: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    inputs = list(concepts)
    if len(concepts) > 1:
        inputs.append(' '.join(concepts))

    vectors = embed_documents(inputs)
    concept_vectors = vectors[: len(concepts)]
    if len(concepts) == 1:
        query_vector = concept_vectors[0]
    else:
        phrase_vector = vectors[-1]
        query_vector = normalize_vector(concept_vectors.mean(axis=0) * 0.58 + phrase_vector * 0.42)

    return concept_vectors, query_vector


def project_query_vectors(vectors: np.ndarray) -> np.ndarray:
    if len(vectors) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if len(vectors) == 1:
        return np.zeros((1, 2), dtype=np.float32)

    centered = vectors - vectors[0]

    try:
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        if vt.shape[0] >= 2:
            coords = centered @ vt[:2].T
        else:
            coords = centered[:, :1]
    except np.linalg.LinAlgError:
        coords = centered[:, :2]

    if coords.shape[1] == 1:
        coords = np.column_stack(
            (coords[:, 0], np.zeros(coords.shape[0], dtype=np.float32))
        )

    max_abs = np.max(np.abs(coords), axis=0)
    max_abs = np.where(max_abs < 1e-6, 1.0, max_abs)
    return coords / max_abs


def relax_constellation_layout(nodes: list[dict], iterations: int = 90) -> None:
    for _ in range(iterations):
        moved = False
        for left_index in range(len(nodes)):
            for right_index in range(left_index + 1, len(nodes)):
                left = nodes[left_index]
                right = nodes[right_index]

                delta_x = right['x'] - left['x']
                delta_y = right['y'] - left['y']
                distance = float(np.hypot(delta_x, delta_y))
                min_distance = left['collision_radius'] + right['collision_radius']

                if distance >= min_distance:
                    continue

                if distance < 1e-6:
                    delta_x = 0.0008 * (right_index + 1)
                    delta_y = 0.0005 * (left_index + 1)
                    distance = float(np.hypot(delta_x, delta_y))

                push = (min_distance - distance) / 2.0
                unit_x = delta_x / distance
                unit_y = delta_y / distance

                if left['locked'] and right['locked']:
                    continue
                if left['locked']:
                    right['x'] += unit_x * push * 2.0
                    right['y'] += unit_y * push * 2.0
                elif right['locked']:
                    left['x'] -= unit_x * push * 2.0
                    left['y'] -= unit_y * push * 2.0
                else:
                    left['x'] -= unit_x * push
                    left['y'] -= unit_y * push
                    right['x'] += unit_x * push
                    right['y'] += unit_y * push

                moved = True

        for node in nodes:
            node['x'] = clamp(float(node['x']), 0.08, 0.92)
            node['y'] = clamp(float(node['y']), 0.12, 0.88)

        if not moved:
            break


def add_query_edge(
    edge_map: dict[tuple[str, str, str], dict],
    source: str,
    target: str,
    strength: float,
    kind: str,
) -> None:
    if source == target:
        return

    left, right = sorted((source, target))
    key = (left, right, kind)
    current = edge_map.get(key)
    if current is None or strength > current['strength']:
        edge_map[key] = {
            'source': source,
            'target': target,
            'strength': round(float(strength), 4),
            'kind': kind,
        }


def segment_orientation(
    first: tuple[float, float],
    second: tuple[float, float],
    third: tuple[float, float],
) -> float:
    return (
        (second[0] - first[0]) * (third[1] - first[1])
        - (second[1] - first[1]) * (third[0] - first[0])
    )


def point_on_segment(
    start: tuple[float, float],
    end: tuple[float, float],
    point: tuple[float, float],
    epsilon: float = 1e-6,
) -> bool:
    return (
        min(start[0], end[0]) - epsilon <= point[0] <= max(start[0], end[0]) + epsilon
        and min(start[1], end[1]) - epsilon <= point[1] <= max(start[1], end[1]) + epsilon
    )


def segments_intersect(
    first_start: tuple[float, float],
    first_end: tuple[float, float],
    second_start: tuple[float, float],
    second_end: tuple[float, float],
    epsilon: float = 1e-6,
) -> bool:
    orientation_1 = segment_orientation(first_start, first_end, second_start)
    orientation_2 = segment_orientation(first_start, first_end, second_end)
    orientation_3 = segment_orientation(second_start, second_end, first_start)
    orientation_4 = segment_orientation(second_start, second_end, first_end)

    if (
        ((orientation_1 > epsilon and orientation_2 < -epsilon) or (orientation_1 < -epsilon and orientation_2 > epsilon))
        and ((orientation_3 > epsilon and orientation_4 < -epsilon) or (orientation_3 < -epsilon and orientation_4 > epsilon))
    ):
        return True

    if abs(orientation_1) <= epsilon and point_on_segment(first_start, first_end, second_start, epsilon):
        return True
    if abs(orientation_2) <= epsilon and point_on_segment(first_start, first_end, second_end, epsilon):
        return True
    if abs(orientation_3) <= epsilon and point_on_segment(second_start, second_end, first_start, epsilon):
        return True
    if abs(orientation_4) <= epsilon and point_on_segment(second_start, second_end, first_end, epsilon):
        return True

    return False


def edge_crosses_group(
    source_id: str,
    target_id: str,
    positions: dict[str, tuple[float, float]],
    selected_edges: list[tuple[str, str]],
) -> bool:
    source_point = positions[source_id]
    target_point = positions[target_id]

    for current_source, current_target in selected_edges:
        if source_id in (current_source, current_target) or target_id in (current_source, current_target):
            continue
        if segments_intersect(
            source_point,
            target_point,
            positions[current_source],
            positions[current_target],
        ):
            return True

    return False


def build_group_mesh_edges(
    group_nodes: list[dict],
    similarities: np.ndarray,
) -> list[tuple[str, str, float]]:
    if len(group_nodes) < 2:
        return []

    positions = {
        node['id']: (float(node['x']), float(node['y']))
        for node in group_nodes
    }
    node_ids = [str(node['id']) for node in group_nodes]
    node_count = len(group_nodes)
    degrees = {node_id: 0 for node_id in node_ids}
    selected_edges: list[tuple[str, str]] = []
    selected_keys: set[tuple[str, str]] = set()

    candidate_pairs: list[dict[str, float | int | str]] = []
    for left_index in range(node_count):
        for right_index in range(left_index + 1, node_count):
            left_id = node_ids[left_index]
            right_id = node_ids[right_index]
            left_point = positions[left_id]
            right_point = positions[right_id]
            layout_distance = float(np.hypot(
                left_point[0] - right_point[0],
                left_point[1] - right_point[1],
            ))
            similarity = float(similarities[left_index, right_index])
            candidate_pairs.append(
                {
                    'left_index': left_index,
                    'right_index': right_index,
                    'left_id': left_id,
                    'right_id': right_id,
                    'similarity': similarity,
                    'distance': layout_distance,
                }
            )

    parent = list(range(node_count))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left_index: int, right_index: int) -> None:
        left_root = find(left_index)
        right_root = find(right_index)
        if left_root != right_root:
            parent[right_root] = left_root

    tree_candidates = sorted(
        candidate_pairs,
        key=lambda item: (
            item['distance'],
            -float(item['similarity']),
        ),
    )

    for candidate in tree_candidates:
        left_index = int(candidate['left_index'])
        right_index = int(candidate['right_index'])
        left_id = str(candidate['left_id'])
        right_id = str(candidate['right_id'])
        edge_key = tuple(sorted((left_id, right_id)))

        if find(left_index) == find(right_index):
            continue
        if edge_crosses_group(left_id, right_id, positions, selected_edges):
            continue

        selected_edges.append((left_id, right_id))
        selected_keys.add(edge_key)
        degrees[left_id] += 1
        degrees[right_id] += 1
        union(left_index, right_index)

        if len(selected_edges) >= node_count - 1:
            break

    mesh_candidates = sorted(
        candidate_pairs,
        key=lambda item: (
            -(float(item['similarity']) * 0.72 + max(0.0, 0.22 - float(item['distance'])) * 1.45),
            item['distance'],
        ),
    )
    max_edges = min((node_count * 2) - 3, (node_count * (node_count - 1)) // 2)

    for candidate in mesh_candidates:
        if len(selected_edges) >= max_edges:
            break

        left_id = str(candidate['left_id'])
        right_id = str(candidate['right_id'])
        edge_key = tuple(sorted((left_id, right_id)))
        if edge_key in selected_keys:
            continue

        if degrees[left_id] >= 4 or degrees[right_id] >= 4:
            continue
        if float(candidate['distance']) > 0.24:
            continue
        if edge_crosses_group(left_id, right_id, positions, selected_edges):
            continue

        selected_edges.append((left_id, right_id))
        selected_keys.add(edge_key)
        degrees[left_id] += 1
        degrees[right_id] += 1

    edge_strengths = {
        tuple(sorted((str(candidate['left_id']), str(candidate['right_id'])))): clamp(
            0.18 + float(candidate['similarity']) * 0.56,
            0.18,
            0.82,
        )
        for candidate in candidate_pairs
    }

    return [
        (left_id, right_id, edge_strengths[tuple(sorted((left_id, right_id)))])
        for left_id, right_id in selected_edges
    ]


def serialize_query_nodes(nodes: list[dict]) -> list[dict]:
    return [
        {
            'id': node['id'],
            'word': node['word'],
            'role': node['role'],
            'group': node['group'],
            'x': round(float(node['x']), 6),
            'y': round(float(node['y']), 6),
            'size': round(float(node['size']), 4),
            'matched_concepts': list(node.get('matched_concepts', ())),
            'is_fallback': bool(node.get('is_fallback', False)),
            'count': int(node.get('count', 1)),
            'preview_words': list(node.get('preview_words', ())),
        }
        for node in nodes
    ]


def build_query_group_payload(
    group: str,
    hits: tuple[RankedWord, ...],
    center: np.ndarray,
) -> tuple[list[dict], list[tuple[str, str, float]]]:
    words = [hit.word for hit in hits]
    if not words:
        return [], []

    vectors = embed_documents(words)
    projected_words = project_query_vectors(vectors)
    spread = np.asarray(
        [0.16, 0.19] if group == 'consonant' else [0.14, 0.18],
        dtype=np.float32,
    )
    group_nodes: list[dict] = []

    for word_index, hit in enumerate(hits):
        if len(hits) == 1:
            position = center.copy()
        else:
            position = center + projected_words[word_index] * spread
        position = np.clip(position, [0.08, 0.14], [0.92, 0.88])
        group_nodes.append(
            {
                'id': f'result:{group}:{word_index}:{normalize_word(hit.word)}',
                'word': hit.word,
                'role': 'result',
                'group': group,
                'x': float(position[0]),
                'y': float(position[1]),
                'size': 0.96 + clamp(hit.score, 0.0, 0.34),
                'locked': False,
                'collision_radius': 0.056 if group == 'consonant' else 0.052,
                'matched_concepts': list(hit.matched_concepts),
                'is_fallback': hit.is_fallback,
            }
        )

    relax_constellation_layout(group_nodes, iterations=80)
    group_edges = build_group_mesh_edges(group_nodes, vectors @ vectors.T)
    return group_nodes, group_edges


def build_query_constellation(
    target_word: str,
    consonant_hits: tuple[RankedWord, ...],
    assonant_hits: tuple[RankedWord, ...],
) -> dict:
    normalized_target = normalize_word(target_word)
    group_centers = {
        'consonant': np.asarray([0.3, 0.53], dtype=np.float32),
        'assonant': np.asarray([0.72, 0.53], dtype=np.float32),
    }
    filtered_hits = {
        'consonant': tuple(
            hit for hit in consonant_hits if normalize_word(hit.word) != normalized_target
        ),
        'assonant': tuple(
            hit for hit in assonant_hits if normalize_word(hit.word) != normalized_target
        ),
    }

    query_nodes: list[dict] = []
    edge_map: dict[tuple[str, str, str], dict] = {}

    for group in ('consonant', 'assonant'):
        group_nodes, group_edges = build_query_group_payload(
            group,
            filtered_hits[group],
            group_centers[group],
        )
        query_nodes.extend(group_nodes)
        for left_id, right_id, strength in group_edges:
            add_query_edge(edge_map, left_id, right_id, strength, group)

    return {
        'nodes': serialize_query_nodes(query_nodes),
        'edges': list(edge_map.values()),
    }


def ensure_engine_assets() -> None:
    global _WORDS, _DISPLAY_WORDS, _EMBEDDINGS, _FREQUENCIES, _CONSONANT_INDEX, _ASSONANT_INDEX, _GRAPH_PAYLOAD, _DISPLAY_LOOKUP

    if (
        _WORDS is not None
        and _DISPLAY_WORDS is not None
        and _EMBEDDINGS is not None
        and _FREQUENCIES is not None
        and _CONSONANT_INDEX is not None
        and _ASSONANT_INDEX is not None
        and _GRAPH_PAYLOAD is not None
        and _DISPLAY_LOOKUP is not None
    ):
        return

    CACHE_DIR.mkdir(exist_ok=True)
    FASTEMBED_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    legacy_index_file = next((path for path in LEGACY_INDEX_CACHE_FILES if path.exists()), None)

    if INDEX_CACHE_FILE.exists():
        payload = np.load(INDEX_CACHE_FILE, allow_pickle=False)
        _WORDS = payload['words']
        _DISPLAY_WORDS = payload['display_words'] if 'display_words' in payload.files else restore_display_words(_WORDS)
        _EMBEDDINGS = payload['embeddings']
        _FREQUENCIES = payload['frequencies']
        _CONSONANT_INDEX = build_lookup_index(payload['consonant_codes'])
        _ASSONANT_INDEX = build_lookup_index(payload['assonant_codes'])
        _DISPLAY_LOOKUP = {
            str(word): str(display_word)
            for word, display_word in zip(_WORDS.tolist(), _DISPLAY_WORDS.tolist())
        }
        if GRAPH_CACHE_FILE.exists():
            _GRAPH_PAYLOAD = json.loads(GRAPH_CACHE_FILE.read_text(encoding='utf-8'))
        else:
            _GRAPH_PAYLOAD = build_constellation_payload(_WORDS, _DISPLAY_WORDS, _EMBEDDINGS, _FREQUENCIES)
            GRAPH_CACHE_FILE.write_text(
                json.dumps(_GRAPH_PAYLOAD, ensure_ascii=False),
                encoding='utf-8',
            )
        return

    if legacy_index_file is not None:
        payload = np.load(legacy_index_file, allow_pickle=False)
        words_array = payload['words']
        display_words_array = payload['display_words'] if 'display_words' in payload.files else restore_display_words(words_array)
        embeddings = payload['embeddings']
        frequencies = payload['frequencies']

        consonant_codes = []
        assonant_codes = []
        for display_word in display_words_array.tolist():
            consonant_code, assonant_code = rhyme_signature(str(display_word))
            consonant_codes.append(consonant_code)
            assonant_codes.append(assonant_code)

        consonant_codes_array = np.asarray(consonant_codes)
        assonant_codes_array = np.asarray(assonant_codes)
        np.savez_compressed(
            INDEX_CACHE_FILE,
            words=words_array,
            display_words=display_words_array,
            embeddings=embeddings,
            frequencies=frequencies,
            consonant_codes=consonant_codes_array,
            assonant_codes=assonant_codes_array,
        )

        _WORDS = words_array
        _DISPLAY_WORDS = display_words_array
        _EMBEDDINGS = embeddings
        _FREQUENCIES = frequencies
        _CONSONANT_INDEX = build_lookup_index(consonant_codes_array)
        _ASSONANT_INDEX = build_lookup_index(assonant_codes_array)
        _DISPLAY_LOOKUP = {
            str(word): str(display_word)
            for word, display_word in zip(words_array.tolist(), display_words_array.tolist())
        }
        if GRAPH_CACHE_FILE.exists():
            _GRAPH_PAYLOAD = json.loads(GRAPH_CACHE_FILE.read_text(encoding='utf-8'))
        else:
            _GRAPH_PAYLOAD = build_constellation_payload(_WORDS, _DISPLAY_WORDS, _EMBEDDINGS, _FREQUENCIES)
            GRAPH_CACHE_FILE.write_text(
                json.dumps(_GRAPH_PAYLOAD, ensure_ascii=False),
                encoding='utf-8',
            )
        return

    words, display_words, frequencies = select_lexicon()
    embeddings = embed_documents(words)

    consonant_codes = []
    assonant_codes = []
    for display_word in display_words:
        consonant_code, assonant_code = rhyme_signature(display_word)
        consonant_codes.append(consonant_code)
        assonant_codes.append(assonant_code)

    words_array = np.asarray(words)
    display_words_array = np.asarray(display_words)
    consonant_codes_array = np.asarray(consonant_codes)
    assonant_codes_array = np.asarray(assonant_codes)
    graph_payload = build_constellation_payload(words_array, display_words_array, embeddings, frequencies)

    np.savez_compressed(
        INDEX_CACHE_FILE,
        words=words_array,
        display_words=display_words_array,
        embeddings=embeddings,
        frequencies=frequencies,
        consonant_codes=consonant_codes_array,
        assonant_codes=assonant_codes_array,
    )
    GRAPH_CACHE_FILE.write_text(
        json.dumps(graph_payload, ensure_ascii=False),
        encoding='utf-8',
    )

    _WORDS = words_array
    _DISPLAY_WORDS = display_words_array
    _EMBEDDINGS = embeddings
    _FREQUENCIES = frequencies
    _CONSONANT_INDEX = build_lookup_index(consonant_codes_array)
    _ASSONANT_INDEX = build_lookup_index(assonant_codes_array)
    _GRAPH_PAYLOAD = graph_payload
    _DISPLAY_LOOKUP = {
        str(word): str(display_word)
        for word, display_word in zip(words_array.tolist(), display_words_array.tolist())
    }


def get_constellation_payload() -> dict:
    ensure_engine_assets()
    return _GRAPH_PAYLOAD or {'nodes': [], 'edges': []}


def get_embedding_metadata() -> dict[str, str | int]:
    ensure_engine_assets()
    return {
        'model_name': MODEL_NAME,
        'word_count': int(len(_WORDS) if _WORDS is not None else 0),
    }


def rank_candidates(
    indices: np.ndarray,
    target_word: str,
    concepts: tuple[str, ...],
    concept_vectors: np.ndarray,
    query_vector: np.ndarray,
) -> tuple[RankedWord, ...]:
    if indices.size == 0:
        return ()

    target_word = normalize_word(target_word)
    candidate_words = _WORDS[indices]
    candidate_display_words = _DISPLAY_WORDS[indices]
    candidate_vectors = _EMBEDDINGS[indices]
    candidate_frequencies = _FREQUENCIES[indices]

    similarity_to_query = candidate_vectors @ query_vector
    if len(concepts):
        similarity_to_concepts = candidate_vectors @ concept_vectors.T
        positive_similarity = np.maximum(similarity_to_concepts, 0.0)
    else:
        similarity_to_concepts = np.zeros((len(indices), 0), dtype=np.float32)
        positive_similarity = similarity_to_concepts

    frequency_bonus = (candidate_frequencies / 8.0) * 0.03
    if len(concepts):
        score = (
            similarity_to_query * 0.8
            + positive_similarity.mean(axis=1) * 0.18
            + positive_similarity.max(axis=1) * 0.08
            + frequency_bonus
        )
    else:
        score = similarity_to_query + frequency_bonus

    ranked_hits: list[RankedWord] = []
    fallback_hits: list[RankedWord] = []
    reserve_hits: list[RankedWord] = []
    order = np.argsort(-score)

    for position in order.tolist():
        word = str(candidate_words[position])
        display_word = str(candidate_display_words[position])
        normalized_word = normalize_word(word)
        if normalized_word == target_word:
            continue

        matched_concepts = tuple(
            concepts[index]
            for index, similarity in enumerate(positive_similarity[position].tolist())
            if similarity >= MATCH_THRESHOLD
        )
        max_concept_similarity = (
            float(positive_similarity[position].max()) if len(concepts) else 0.0
        )
        semantic_query_match = (
            float(similarity_to_query[position]) >= DIRECT_QUERY_MATCH_THRESHOLD
            or max_concept_similarity >= MATCH_THRESHOLD - 0.02
        )
        hit = RankedWord(
            word=display_word,
            score=float(score[position]),
            matched_concepts=matched_concepts,
            is_fallback=not matched_concepts and not semantic_query_match,
        )

        if matched_concepts or semantic_query_match:
            ranked_hits.append(hit)
        elif float(similarity_to_query[position]) >= SEMANTIC_FALLBACK_THRESHOLD:
            fallback_hits.append(hit)
        else:
            reserve_hits.append(hit)

    ordered_hits = ranked_hits + fallback_hits + reserve_hits

    if not ordered_hits:
        return ()

    return tuple(ordered_hits)


def search_rhymes(
    target_word: str,
    concepts: list[str],
    progress_callback: Callable[[float, str], None] | None = None,
    result_limit: int = DEFAULT_RESULTS_PER_GROUP,
    result_page: int = 0,
) -> SearchOutcome:
    result_limit = max(MIN_RESULTS_PER_GROUP, min(MAX_RESULTS_PER_GROUP, int(result_limit)))
    result_page = max(0, int(result_page))
    report_progress(progress_callback, 0.08, 'Inicializando embeddings')
    ensure_engine_assets()

    report_progress(progress_callback, 0.18, 'Normalizando consulta')
    normalized_target = normalize_word(target_word)
    rhyme_target = resolve_rhyme_surface(target_word)
    normalized_concepts = tuple(
        concept for concept in (normalize_word(item) for item in concepts) if concept
    )

    report_progress(progress_callback, 0.34, 'Construyendo vector semantico')
    concept_vectors, query_vector = build_query_bundle(normalized_concepts)
    consonant_code, assonant_code = rhyme_signature(rhyme_target)

    consonant_indices = _CONSONANT_INDEX.get(consonant_code, np.asarray([], dtype=np.int32))
    assonant_indices = _ASSONANT_INDEX.get(assonant_code, np.asarray([], dtype=np.int32))

    report_progress(progress_callback, 0.52, 'Buscando rima consonante')
    consonant_hits = rank_candidates(
        consonant_indices,
        normalized_target,
        normalized_concepts,
        concept_vectors,
        query_vector,
    )
    report_progress(progress_callback, 0.72, 'Buscando rima asonante')
    assonant_hits = rank_candidates(
        assonant_indices,
        normalized_target,
        normalized_concepts,
        concept_vectors,
        query_vector,
    )

    consonant_total = len(consonant_hits)
    assonant_total = len(assonant_hits)
    page_count = max(
        1,
        (max(consonant_total, assonant_total) + result_limit - 1) // result_limit,
    )
    result_page = min(result_page, page_count - 1)
    page_start = result_page * result_limit
    page_end = page_start + result_limit
    visible_consonant_hits = consonant_hits[page_start:page_end]
    visible_assonant_hits = assonant_hits[page_start:page_end]

    warning_message = None
    if any(hit.is_fallback for hit in visible_consonant_hits + visible_assonant_hits):
        warning_message = (
            'No todas las sugerencias tienen una cercania semantica fuerte. '
            'Cuando el embedding no encuentra una asociacion clara, dejo algunas rimas por sonido como apoyo.'
        )

    report_progress(progress_callback, 0.9, 'Trazando constelacion')
    query_constellation = build_query_constellation(
        normalized_target,
        visible_consonant_hits,
        visible_assonant_hits,
    )

    report_progress(progress_callback, 1.0, 'Listo')

    return SearchOutcome(
        consonant=visible_consonant_hits,
        assonant=visible_assonant_hits,
        consonant_total=consonant_total,
        assonant_total=assonant_total,
        result_page=result_page,
        page_count=page_count,
        warning_message=warning_message,
        query_constellation=query_constellation,
    )
