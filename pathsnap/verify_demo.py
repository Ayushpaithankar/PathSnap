"""Exercise the served UI and run live inference over the student test split."""
import argparse
import csv
import json
import math
from itertools import product
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--api', default='http://127.0.0.1:8000')
    parser.add_argument('--partition', choices=['student_test', 'evaluator_test'], default='student_test')
    parser.add_argument('--expected', type=Path, help='Ordered experiment per-trip comparison CSV')
    parser.add_argument('--review', type=Path, help='Review experiment per-trip comparison CSV')
    parser.add_argument('--all-methods', action='store_true', help='Exercise every method exposed by the API')
    args = parser.parse_args()
    api = args.api.rstrip('/')
    expected = {}
    if args.expected:
        with args.expected.open(newline='') as stream:
            expected = {(r['model'], r['trip_id']): r for r in csv.DictReader(stream)}
    review = {}
    if args.review:
        with args.review.open(newline='') as stream:
            review = {(r['method'], r['trip_id']): r for r in csv.DictReader(stream)}

    def request(path, body=None):
        payload = json.dumps(body).encode() if body is not None else None
        return urlopen(Request(api + path, data=payload,
                               headers={'Content-Type': 'application/json'}), timeout=90)

    with request('/') as response:
        assert b'<title>PathSnap' in response.read()
    with request('/api/trips?' + urlencode({'partition': args.partition})) as response:
        ids = json.load(response)['trip_ids']
    assert len(ids) == len(set(ids)) == 10
    methods = ['default']
    if review or args.all_methods:
        with request('/api/models') as response:
            methods = [entry['id'] for entry in json.load(response)['models']]
        assert methods and methods[0] == 'default'
    for trip_id, method in product(ids, methods):
        with request('/api/predict', {'trip_id': trip_id, 'partition': args.partition,
                                      'method': method}) as response:
            result = json.load(response)
            decoded = result['decoded']
            assert result['trip_id'] == trip_id
            assert [row['index'] for row in result['observations']] == list(range(len(decoded['arc_ids'])))
            assert len(decoded['connecting_arc_ids']) == len(decoded['arc_ids'])
            assert 'reference_arc_id' not in json.dumps(result)
            assert result['metrics']['observations'] == len(decoded['arc_ids'])
            assert result['decoded']['segments']['type'] == 'FeatureCollection'
            if expected:
                m = result['metrics']
                row = expected[result['model'], trip_id]
                assert m['labeled'] == m['observations']
                for actual, recorded in (
                    (m['baseline']['point_accuracy'], row['classifier_top1']),
                    (m['decoded']['point_accuracy'], row['decoded_top1']),
                    (m['decoded']['sequence_f1'], row['sequence_f1']),
                    (m['rules']['nearest_road'], expected['Nearest road', trip_id]['classifier_top1']),
                    (m['rules']['distance_heading'], expected['Distance + heading', trip_id]['classifier_top1']),
                ):
                    assert math.isclose(actual, float(recorded), abs_tol=1e-12), (trip_id, actual, recorded)
            if review:
                m = result['metrics']
                row = review[result['model'], trip_id]
                n = int(row['observations'])
                assert m['labeled'] == m['observations'] == n
                for actual, value in [
                    (m['baseline']['point_accuracy'], int(row['point_correct'])/n),
                    (m['decoded']['point_accuracy'], int(row['decoded_correct'])/n),
                    (m['decoded']['sequence_f1'], 2*int(row['route_matches'])/int(row['sequence_total'])),
                    (m['rules']['nearest_road'], int(review['Nearest road', trip_id]['point_correct'])/n),
                    (m['rules']['distance_heading'], int(review['Distance + heading', trip_id]['point_correct'])/n),
                ]:
                    assert math.isclose(actual, value, abs_tol=1e-12), (trip_id, method, actual, value)
                print(method, end=' ')
            print(trip_id, response.status, result['metrics']['observations'],
                  result['metrics']['baseline']['point_accuracy'],
                  result['metrics']['decoded']['point_accuracy'], flush=True)
    try:
        request('/api/predict', {'trip_id': '99999999', 'partition': args.partition})
    except HTTPError as error:
        assert error.code == 404
        print('Missing trip returned HTTP 404')
    else:
        raise AssertionError('Expected HTTP 404 for missing trip')


if __name__ == '__main__':
    main()
