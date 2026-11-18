from pathsnap.core import RoadGraph,validate_candidates,decode,baseline

def test_missing_candidate_splits_route():
    nodes=[(8.0,49.0),(8.001,49.0),(8.002,49.0)]
    arcs=[(0,1),(1,2)]
    track=[{'index':i,'lon':8+i*.001,'lat':49.,'timestamp':float(i+1)} for i in range(3)]
    rows=[{'trip_id':'00000001','observation_index':i,'timestamp':float(i+1),'arc_id':a,'snapped_lon':8+i*.001,'snapped_lat':49.,'distance_m':0.,'probability':p} for i,a,p in [(0,0,.8),(0,1,.2),(2,1,.9)]]
    group=validate_candidates(track,arcs,rows)
    assert baseline(track,group)==[0,None,1]
    predicted,connectors,warnings=decode(track,RoadGraph(nodes,arcs),group)
    assert predicted==[0,None,1]
    assert connectors==[[],[],[]]
    assert 'no candidates' in warnings[0]


def test_backward_move_needs_a_directed_return_path():
    nodes = [(0., 0.), (.0005, 0.)]
    track = [dict(lon=lon, lat=0., timestamp=float(i)) for i, lon in enumerate((.0004, .0001))]
    rows = {i: [dict(arc_id=0, snapped_lon=o['lon'], snapped_lat=0., probability=1.)]
            for i, o in enumerate(track)}
    ids, links, warnings = decode(track, RoadGraph(nodes, [(0, 1)]), rows)
    assert ids == [0, 0] and links == [[], []] and 'disconnected' in warnings[0]
    ids, links, warnings = decode(track, RoadGraph(nodes, [(0, 1), (1, 0)]), rows)
    assert ids == [0, 0] and links == [[], [1]] and not warnings


def test_cached_path_respects_the_exact_limit():
    graph = RoadGraph([(0., 0.), (.0005, 0.)], [(0, 1)])
    distance = graph.lengths[0]
    assert graph.path(0, 1, distance + .01) == (distance, [0])
    assert graph.path(0, 1, distance - .01) == (float('inf'), [])


def test_decoder_can_restart_instead_of_staying_on_an_unlikely_arc():
    graph = RoadGraph([(0., 0.), (.0005, 0.), (0., .0001), (.0005, .0001)],
                      [(0, 1), (2, 3)])
    track = [dict(lon=0., lat=0., timestamp=0.),
             dict(lon=0., lat=.0001, timestamp=60.)]
    grouped = {0: [dict(arc_id=0, probability=1., snapped_lon=0., snapped_lat=0.)],
               1: [dict(arc_id=0, probability=.001, snapped_lon=0., snapped_lat=0.),
                   dict(arc_id=1, probability=.999, snapped_lon=0., snapped_lat=.0001)]}

    assert decode(track, graph, grouped, bridge_gaps=True)[0] == [0, 0]
    ids, links, warnings = decode(track, graph, grouped, bridge_gaps=True, restart_penalty=2.)
    assert ids == [0, 1]
    assert links == [[], []]
    assert warnings == ['observation 1: evidence favors a new segment']
