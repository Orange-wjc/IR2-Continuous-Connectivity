"""Exercise the actual driver pause predicate against real performance flags.

No Ray instance, policy training, or synthetic copy of the predicate.
"""
import ast
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
source = (root/'driver.py').read_text()
tree = ast.parse(source)
matches = [node for node in ast.walk(tree) if isinstance(node, ast.If)
           and any(isinstance(x, ast.Constant) and x.value == 'positive_reward_idle_wait' for x in ast.walk(node.test))]
assert len(matches) == 1
condition = compile(ast.Expression(body=matches[0].test), str(root/'driver.py'), 'eval')


def pauses(flags):
    return eval(condition, {'report': {'flags': flags}})


review = json.loads((root/'experiments/contact_easy10_v7_20261019/reviews/episode_001200/review.json').read_text())
assert review['trend'] == 'falling' and review['current_block']['success_percent'] == 59
assert not pauses(review['flags'])
assert not pauses(['never_successful:img_9995.png'])
assert not pauses([])
for flag in ('positive_reward_idle_wait', 'invalid_rollouts', 'non_finite_metrics',
             'review_failed', 'progress_history_requires_audit'):
    assert pauses([flag])
    assert pauses(review['flags']+[flag])
assert 'if auto_review and progress.acceptance is not None:' in source
assert 'if auto_review and consecutive_invalid >= 3:' in source
print(json.dumps(dict(passed=True, real1200_performance_flags_continue=True,
                     all_five_correctness_flags_still_pause=True,
                     target_and_three_invalid_guards_retained=True,
                     no_training_or_evaluation=True), indent=2))
