"""Minimal in-memory counterfactual Env; upstream source is untouched.
Copy on write preserves initial observations and immutable shared map snapshots.
For hypothetical stale links only, use ego's map (unknown counted as obstacles
by the existing signal model). Physical links still use world ground truth.
"""
import difflib
import inspect
import textwrap
from types import FunctionType
import env


def source_for(mode='both'):
    if mode not in ('map_only','both'):
        raise ValueError(mode)
    original=textwrap.dedent(inspect.getsource(env.Env.single_robot_step))
    source=original
    old='self.update_robot_belief(robot_position, self.sensor_range, self.all_robot_belief[robot_id][robot_id], self.ground_truth)'
    new='self.update_robot_belief(robot_position, self.sensor_range, self.all_robot_belief[robot_id][robot_id].copy(), self.ground_truth)'
    assert source.count(old)==1
    source=source.replace(old,new)
    if mode=='both':
        old='belief_in_comms_range = self.ss_realistic_model.is_within_signal_strength(self.ground_truth, '
        new='belief_in_comms_range = self.ss_realistic_model.is_within_signal_strength(self.all_robot_belief[robot_id][robot_id], '
        assert source.count(old)==1
        source=source.replace(old,new)
    return original,source


def build_env(mode='both'):
    original,source=source_for(mode)
    namespace={}
    exec(compile(source,__file__+':'+mode,'exec'),namespace)
    method=FunctionType(namespace['single_robot_step'].__code__,env.__dict__)
    return type('ControlledEnv',(env.Env,),{'single_robot_step':method,'control_mode':mode})


def patch_text(mode='both'):
    before,after=source_for(mode)
    return ''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='upstream/Env.single_robot_step',tofile=mode+'/Env.single_robot_step'))
