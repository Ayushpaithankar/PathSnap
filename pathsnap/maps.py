"""Export development Folium maps from the fitted classifier and decoder."""
import argparse
from pathlib import Path

import folium
import joblib
from threadpoolctl import threadpool_limits

from .candidates import predict_trip
from .core import load_reference, load_trip
from .pipeline import model_trip, reconstruct


def render(result, output, reference=None, nodes=None, arcs=None):
    observations = result['observations']
    bounds = [[o['lat'], o['lon']] for o in observations]
    road_map = folium.Map(location=bounds[0], zoom_start=14, control_scale=True)
    gps = folium.FeatureGroup(name='GPS observations')
    for lat, lon in bounds:
        folium.CircleMarker((lat, lon), radius=2, color='#1764bb', fill=True).add_to(gps)
    gps.add_to(road_map)
    folium.GeoJson(result['road_network'], name='Candidate road network',
                   style_function=lambda _: {'color': '#777', 'weight': 2}).add_to(road_map)
    folium.GeoJson(result['baseline']['segments'], name='Classifier-only route', show=False,
                   style_function=lambda _: {'color': '#a83279', 'weight': 3}).add_to(road_map)
    if reference:
        layer = folium.FeatureGroup(name='Reference route (development)', show=False)
        for arc in reference:
            u, v = arcs[arc]
            folium.PolyLine([nodes[u][::-1], nodes[v][::-1]],
                            color='#d26b17', weight=4).add_to(layer)
        layer.add_to(road_map)
    folium.GeoJson(result['decoded']['segments'], name='Decoded route',
                   style_function=lambda feature: {'color': '#138a36', 'weight': 4,
                       'dashArray': '5 4' if feature['properties']['connector'] else None}).add_to(road_map)
    legend = '''<div style="position:fixed;bottom:18px;left:18px;z-index:1000;background:white;
    padding:9px 12px;border:1px solid #999;border-radius:6px;font:13px Arial">
    <b>PathSnap layers</b><br><span style="color:#1764bb">&#9679;</span> GPS points<br>
    <span style="color:#777">&#9473;</span> Candidate roads<br>
    <span style="color:#a83279">&#9473;</span> Classifier-only route<br>
    <span style="color:#138a36">&#9473;</span> Decoded route (dashed connectors)'''
    if reference:
        legend += '<br><span style="color:#d26b17">&#9473;</span> Reference route'
    road_map.get_root().html.add_child(folium.Element(legend + '</div>'))
    folium.LayerControl().add_to(road_map)
    road_map.fit_bounds(bounds)
    road_map.save(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_root', type=Path)
    parser.add_argument('partition', choices=['train', 'student_test', 'evaluator_test'])
    parser.add_argument('trip_id')
    parser.add_argument('output', type=Path)
    parser.add_argument('--model', type=Path, default=Path('results/review/deployed.joblib'))
    parser.add_argument('--reference', action='store_true')
    args = parser.parse_args()
    if args.partition == 'evaluator_test' and args.reference:
        parser.error('Evaluator reference routes must remain outside development maps')
    track, nodes, arcs = load_trip(args.data_root, args.partition, args.trip_id)
    trip = model_trip(args.trip_id, track, nodes, arcs)
    artifact = joblib.load(args.model)
    with threadpool_limits(limits=1):
        _, scores = predict_trip(trip, artifact)
    result = reconstruct(args.trip_id, track, nodes, arcs, scores.to_dict('records'), artifact.get('decoder'))
    reference = load_reference(args.data_root, args.partition, args.trip_id) if args.reference else None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    render(result, args.output, reference, nodes, arcs)


if __name__ == '__main__':
    main()
