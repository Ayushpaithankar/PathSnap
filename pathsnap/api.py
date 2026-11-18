import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from threadpoolctl import threadpool_limits

from .candidates import predict_trip
from .core import list_trip_ids, load_reference, load_trip
from .pipeline import metrics, model_trip, reconstruct

app = FastAPI(title='PathSnap')
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['GET', 'POST'],
                   allow_headers=['Content-Type'])
PROJECT = Path(__file__).resolve().parents[1]
ROOT = Path(os.getenv('PATHSNAP_DATA_ROOT', str(PROJECT / 'inputs/Student folder.zip')))
EVALUATOR_ROOT = os.getenv('PATHSNAP_EVALUATOR_ROOT')
ORDERED_MODEL = PROJECT / 'results/ordered/ordered_model.joblib'
REVIEW_MODEL = PROJECT / 'results/review/deployed.joblib'
RANKER_MODEL = PROJECT / 'results/review/tuned_ranker.joblib'
NEAREST_MODEL = PROJECT / 'results/review/nearest_rule.joblib'
MODEL = Path(os.getenv('PATHSNAP_MODEL', str(next((path for path in (
    REVIEW_MODEL, ORDERED_MODEL, PROJECT / 'results/best_classifier.joblib'
) if path.is_file()), PROJECT / 'results/best_classifier.joblib'))))


@lru_cache(maxsize=2)
def artifact(method='default'):
    source = {'ranker': RANKER_MODEL, 'nearest': NEAREST_MODEL}.get(method, MODEL)
    return joblib.load(source)


class Request(BaseModel):
    partition: str = 'evaluator_test'
    trip_id: str
    method: Literal['default', 'ranker', 'nearest'] = 'default'


def data_root(partition: str) -> Path:
    """Allow the supplied evaluator_test folder to stay outside the project data ZIP."""
    if partition == 'evaluator_test' and EVALUATOR_ROOT:
        path = Path(EVALUATOR_ROOT).expanduser().resolve()
        return path.parent if path.name == 'evaluator_test' else path
    return ROOT


@app.get('/api/models')
def models():
    if not MODEL.is_file():
        return {'models': []}
    items = [{'id': 'default', 'name': artifact().get('name', 'Default model')}]
    if RANKER_MODEL.is_file() and items[0]['name'] != 'LightGBM tuned':
        items.append({'id': 'ranker', 'name': 'LightGBM tuned'})
    if NEAREST_MODEL.is_file() and items[0]['name'] != 'Nearest road':
        items.append({'id': 'nearest', 'name': 'Nearest road baseline'})
    return {'models': items}


@app.get('/api/trips')
def trips(partition: str = 'evaluator_test'):
    try:
        ids = list_trip_ids(data_root(partition), partition)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {'partition': partition, 'trip_ids': ids}


@app.post('/api/predict')
def predict(request: Request):
    source = data_root(request.partition)
    try:
        track, nodes, arcs = load_trip(source, request.partition, request.trip_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(404, 'Trip files unavailable') from exc
    try:
        fitted = artifact(request.method)
    except FileNotFoundError as exc:
        raise HTTPException(503, 'Model artifact unavailable') from exc
    trip = model_trip(request.trip_id, track, nodes, arcs)
    with threadpool_limits(limits=1):
        _, scores = predict_trip(trip, fitted)
    rows = scores.to_dict('records')
    result = reconstruct(request.trip_id, track, nodes, arcs, rows, fitted.get('decoder'))
    # Reference is read only after inference. Only aggregate metrics leave the server.
    reference = load_reference(source, request.partition, request.trip_id)
    ordered = fitted.get('version') == 'ordered-v2'
    result.update(partition=request.partition, model=fitted.get('name', 'Classifier'),
                  label_method='ordered alignment' if ordered else 'geometric projection',
                  metrics=metrics(result, trip, rows, reference, ordered=ordered))
    return result


@app.get('/', response_class=HTMLResponse)
def ui():
    return Path(__file__).with_name('ui.html').read_text()
