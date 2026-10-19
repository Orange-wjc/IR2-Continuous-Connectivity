# Candidate v7: explicit local communication state

Status: full engineering gate passed; formal launch recorded separately in launcher.json. Worktree `/home/wyz/wjc_ws/IR2-contact-state-v7`,
branch `research/ir2-contact-state-v7`, base `1cfa3779`. Do not edit the live
source in `/home/wyz/wjc_ws/IR2-Continuous-Connectivity` or run another formal
training instance. The old sole formal v6 PID1316835 and its workers have exited after the600
review. Both300 windows achieved183/300; all600 valid, three maps never
successful. Exact model/source are archived; native SAC12608/head7436
are preserved in the prepared v7 checkpoint. Goal remains raw300>=297, all10 original maps,
3 robots, no invalid/reward anomalies or privileged actor information.

## Evidence and limits

Frozen300 v6 traces on9995/9999 found no positive local/team-return movement
cycles or WAIT without sender gains. Both retained reachable frontier goals
and advancing candidates while unfinished robots increasingly chose WAIT.
The six graph features expose geometry, frontier/RV utility, visits and
occupancy, but no explicit recent paid-receive/novelty/queue state. The message
head's12 features do not directly reach the movement actor/critics. RV at
the current node is deliberately zero, and many candidate RV values are zero.
Recent receive evidence can therefore differ without being explicitly
represented in the movement input. This is an observability hypothesis,
not a claim that all remaining learning problems have been proved.

A conservative WAIT mask was tested with this local evidence. The first
probe was invalid: PolicyNet force-opened WAIT despite291 requested masks.
The corrected private policy respected masks (zero probability asserted),
with ordinary control36 actions/rewards/wire metrics exact. Results were
mixed:9995 WAIT105->20 but union100%->79.69%;9999 WAIT196->108, minimum
coverage88.63%->93.92%, still failed. Neither completed. Do NOT promote this
action restriction, alter rewards speculatively, or count these probes as
formal training. Proofs are in v6/reviews/episode_000300/frozen_diagnosis/.

## Controlled change

Keep all original6 graph features and128-dimensional IR2 encoders, all
movement choices including WAIT, original SAC equations/hyperparameters,
v6 decision-interval replay and finite protocol/time/byte constraints.
Add six receiver/sender-local communication fields, placed only on the
current node (other/padded nodes zero):

1. Existing expected outgoing delivery sum / gain_scale, clipped to4.
   Reuse the already computed finite RV delta; do not consult receiver truth.
2. Own remaining queued map/control wire bytes, normalized with
   log1p(bytes)/log1p(bytes_per_second*contact_seconds), clipped to4.
3. Freshness of the last fully received paid packet: exp(-age/pose_period),
   zero when none received. ACK, pose and map reception all qualify.
4. Newly received free cells in the last novel-map service, /gain_scale,
   multiplied by that service's age freshness, clipped to4.
5. Newly received known cells (free or occupied) in that same service,
   with the same normalization/freshness. Distinguishes wall-only traffic.
6. Freshest paid peer pose's creation age /60, clipped to4. The common
   deployment prior is the initial reference when no paid pose exists.

Radio records completed-receive times and own novel/free payload counts
after paying for complete packets. These counters must not change ordering,
queues, bytes, ACKs, payloads or same-service forwarding. The actor accessor
reads only receiver[robot] and sender queues, never peer maps/link truth/GT.

Use a separate bias-free6->128 projection initialized to exact zero, appended
after existing modules so old parameter order is retained. Add its output to
the current node's base embedding before the existing encoder. Construct it
inside torch.random.fork_rng to preserve all existing initialization RNG.
Initial extended-input actor/Q outputs must exactly match v6. Original mode
has no projection and still consumes6 fields; finite mode consumes12 total.
The original initial_embedding remains6-wide; report both graph6 and actual
node input12 in setup. This updates the earlier six-field-only engineering
choice because the communication state is now an explicit hypothesis.

The frozen content context continues to encode only the original6 fields;
its original weights/context values and12-feature head remain unchanged.
Graph-only encode_graph may accept6 fields; actual finite actor/Q forward
must validate12, so missing communication observations cannot pass silently.
Keep the existing15 replay columns: node tensors0/9 carry the extra fields,
including true next-decision state from v6. No new masks or handcoded moves.

## Checkpoint and checks

Schema7 must reject ordinary cross-schema resume. An explicit v6->v7 upgrade
preserves every old actor/Q/temperature tensor and old Adam moments/schedulers,
frozen context, head, auxiliary replay/RNG. Append exactly one zero projection
weight to each actor/Q state and a fresh parameter ID (no fabricated moments)
to each movement optimizer; assert old parameter ordering and ID mapping.
Auxiliary encoder state may gain a zero projection for structural compatibility
but graph-only context must stay bit exact. Reset only new-condition
episode/visits/acceptance, retain all v6 evidence and source exposure counts;
movement replay rebuilds as in original driver. Head replay can remain because
its features and frozen context have not changed. Validate legacy fresh
original2/prepared-head imports explicitly; do not silently fill missing
adapter weights in a purported schema7 checkpoint.

Before deployment: local-info/receipt fixtures, exact wire-state comparison,
original-disabled equivalence, zero-init legacy actor/Q/RNG equality, adapter
gradients, observation width/mask/replay continuity, checkpoint old-moment
identity and actual driver save/resume. CPU preparation is allowed here while
v6 runs. Do not run a second Ray/GPU training check while v6 is alive. After
its600 review/confirmed stop, archive exact final model/source, complete
8-worker full10x196/GPU batch128x8/RAM checks on this candidate, then commit,
fast-forward the original branch only while stopped, push and automatically
launch one v7. Original10 maps/3 robots/196 steps/8 samplers remain. Every300
review remains required; this state addition is NOT yet performance evidence.

## CPU preparation checkpoint (historical)

state.py, receiver-only Radio metadata, zero additive Policy/Q projection,12-field recording/runner/driver setup and strict schema7 checks are implemented here. Real36-action paired trace on frozen v6/9995 is exact in physical actions/rewards/wire/metrics/frozen context and original replay data; extra current/next states are correctly stored and continuous. Config reports graph6+communication6=node12. Native protocol60 ticks, own-info fixtures, RNG/zero-init outputs/gradients, original-disabled behavior, useful/empty WAIT and utility-cache regressions passed. CPU copy of actual300 checkpoint passes old tensor/moment identity, fresh projection Adam ID/one-step gradient and missing-state rejection. SAC math block is byte identical to v6. No candidate formal training, GPU/Ray full test or actual driver resume performed yet; no candidate commit/merge/push yet. See baseline_checks/communication_pre_gpu_review.json.

## Full engineering gate passed

v6 stopped automatically at600, all logged workers exited, source/model archived. Native v6 final checkpoint SHA3659542a... was expanded to prepared v7 SHA6c58d5b2..., inheriting SAC12608/head7436, all previous tensors/moments preserved. Actual temporary driver resume twice passed (head7444->7452), prepared binary unchanged. Full8worker/10x196 check returned3300 transitions, exact next-decision states and actual-delivery conservation; original GPU batch128x8 updated new policy/Q1 projections to norms0.001733/0.002121. Peak CUDA17.66GiB and minimum host available15.75GiB. See communication_final_review.json. Freeze this source, merge only while stopped, push and launch one formal v7. No99% result claimed.
