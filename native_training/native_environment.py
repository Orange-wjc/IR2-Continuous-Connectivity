"""Original hierarchical IR2 graph behind the paid finite-contact protocol.

Only own trajectory and locally known occupancy enter graph construction.
No upstream instantaneous team-map merger is executed.
"""
import numpy as np
from finite_contact.maps import configured_map
from finite_contact.communication import Radio
from graph_generator import Graph_generator


def segment_clear(belief, a, b):
    """Positive-length intersection with an unknown/occupied pixel interior."""
    a,b=np.asarray(a,float),np.asarray(b,float)
    lo=np.maximum(0,np.floor(np.minimum(a,b)-.5).astype(int))
    hi=np.minimum(np.array(belief.shape[::-1])-1,np.ceil(np.maximum(a,b)+.5).astype(int))
    ys,xs=np.where(belief[lo[1]:hi[1]+1,lo[0]:hi[0]+1]!=255)
    if not len(xs): return True
    centers=np.column_stack([xs+lo[0],ys+lo[1]])
    entry=np.zeros(len(xs));leave=np.ones(len(xs));delta=b-a
    for axis in (0,1):
        if abs(delta[axis])<1e-12:
            inside=np.abs(centers[:,axis]-a[axis]) < .5-1e-10
            entry[~inside]=2
        else:
            t1=(centers[:,axis]-.5-a[axis])/delta[axis]
            t2=(centers[:,axis]+.5-a[axis])/delta[axis]
            entry=np.maximum(entry,np.minimum(t1,t2))
            leave=np.minimum(leave,np.maximum(t1,t2))
    return not np.any(leave-entry>1e-10)


def initialize_native(env,config,split,index,scorer):
    import graph_generator as native_graph
    native_graph.GLOBAL_GRAPH_NODE_COORDS_THRESH=340 if config.map_family=='dungeon_difficult' else 200
    env.config=config
    env.truth,env.positions,env.map_info=configured_map(config,split,index)
    env.beliefs=[np.full(env.truth.shape,127,np.uint8) for _ in range(2)]
    env.resolution=4;env.time=0.;env.native_steps=0
    for name in ('wait','distance','wait_seconds','wait_disconnected_seconds',
                 'wait_without_new_map_seconds','sync_idle_seconds'):
        setattr(env,name,np.zeros(2))
    env.coverage_deficit=0.;env.connected_seconds=0.;env.union_completion_time=None
    env.reward_components={name:np.zeros(2) for name in
        ('exploration','delivery','time','distance','completion','frontier_shaping')}
    env.radio=Radio(env.truth.shape,config);env.scorer=scorer
    env.feature_provider=None;env._rank_cache={};env.graphs=[];env.graph_data=[None,None]
    env.native_previous_frontiers=[None,None];env.native_old_lengths=[[[0,0],[0,0]] for _ in range(2)]
    for robot in range(2):
        graph=Graph_generator(robot,env.truth.shape,config.neighbors,config.sensor_range,env.map_info['source_file'],plot=False)
        graph.route_node=[env.positions[robot].astype(int).copy()]
        env.graphs.append(graph);env.sense(robot)
        frontier=env.frontiers(robot)
        graph.generate_graph(env.positions[robot].astype(int),env.beliefs[robot],frontier)
        env.native_previous_frontiers[robot]=frontier
    # Same deterministic distribution rule as upstream Worker: local graph node i.
    for robot in range(2):
        env.positions[robot]=env.graphs[robot].node_coords[min(robot,len(env.graphs[robot].node_coords)-1)]
        env.graphs[robot].route_node=[env.positions[robot].astype(int).copy()]
        env.sense(robot)
    env.map_info['starts']=env.positions.tolist()
    for robot in range(2):
        env.radio.peer_pose[robot]=np.r_[env.positions[1-robot],[0.,0.]]
        env.radio.peer_pose_time[robot]=0.
        env.radio.enqueue_updates(robot,env.beliefs[robot],0.)
    env.trajectory=[dict(time=0.,positions=env.positions.tolist())]
    env.refresh_graphs()
    local,union=env.coverage()
    env.trajectory[0].update(coverage=local.tolist(),union_coverage=union)


def refresh_native_graphs(env):
    for robot,graph in enumerate(env.graphs):
        current=np.rint(env.positions[robot]).astype(int)
        peer=np.rint(env.radio.peer_pose[robot][:2]).astype(int)
        known=env.beliefs[robot]
        peer_valid=(0<=peer[0]<known.shape[1] and 0<=peer[1]<known.shape[0] and known[peer[1],peer[0]]==255)
        positions=[None,None];positions[robot]=current
        positions[1-robot]=peer if peer_valid else None
        # Only the own trajectory is available as a global-graph history.
        history=[[[],[]],[[],[]]]
        history[robot][0]=[np.asarray(p,dtype=int) for p in graph.route_node]
        frontier=env.frontiers(robot)
        result=graph.update_graph(known,frontier,env.native_previous_frontiers[robot],positions,
            history,env.native_old_lengths[robot],extend_global_graph_towards_fronters=True,
            eps=env.map_info['index'],step=env.native_steps)
        success,coords,edges,utility,visited=result
        if not success:raise RuntimeError('Native hierarchical graph update failed')
        env.native_old_lengths[robot]=[[len(v) for v in row] for row in history]
        env.native_previous_frontiers[robot]=frontier
        own=int(np.argmin(np.linalg.norm(coords-current,axis=1)))
        assert np.array_equal(coords[own],current)
        order=np.r_[own,np.delete(np.arange(len(coords)),own)]
        env.graph_data[robot]=(coords[order],edges,utility[order],visited[order])
        if len(coords)>env.config.nodes:
            raise RuntimeError('Native graph exceeds explicit resource guard; no map is silently skipped')
