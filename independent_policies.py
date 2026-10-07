"""A declared observation-only frontier/rendezvous baseline; no learned weights."""
import heapq
import numpy as np


def nearest_goal_hop(features, edge_mask, current, legal_first, candidates):
    """Shortest reachable goal using observed edges and observed coordinates."""
    targets=set(int(i) for i in candidates if i != current)
    if not targets:
        return None
    distances={current:0.0}; first={}; heap=[(0.0,current)]
    while heap:
        distance,node=heapq.heappop(heap)
        if distance != distances[node]:
            continue
        if node in targets:
            return first[node]
        neighbors=legal_first if node==current else np.flatnonzero(edge_mask[node]==0)
        for neighbor in neighbors:
            neighbor=int(neighbor)
            if neighbor==node:
                continue
            cost=distance+float(np.linalg.norm(features[node,:2]-features[neighbor,:2]))
            if cost<distances.get(neighbor,float('inf')):
                distances[neighbor]=cost
                first[neighbor]=neighbor if node==current else first[node]
                heapq.heappush(heap,(cost,neighbor))
    return None


def frontier_rendezvous(features, edge_mask, current, legal_first, visits, last_visit):
    hop=nearest_goal_hop(features,edge_mask,current,legal_first,np.flatnonzero(features[:,2]>0))
    reason='frontier'
    if hop is None:
        hop=nearest_goal_hop(features,edge_mask,current,legal_first,np.flatnonzero(features[:,5]>0))
        reason='known_peer'
    if hop is None:
        movable=[int(i) for i in legal_first if i!=current]
        options=movable or [int(i) for i in legal_first]
        def rank(i):
            key=tuple(float(v) for v in features[i,:2])
            return (visits.get(key,0),last_visit.get(key,-1),float(np.linalg.norm(features[i,:2]-features[current,:2])),i)
        hop=min(options,key=rank)
        reason='least_visited_neighbor'
    return hop,reason
