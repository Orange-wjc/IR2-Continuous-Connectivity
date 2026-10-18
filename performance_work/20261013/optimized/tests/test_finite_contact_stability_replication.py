import copy
import unittest
import json
from pathlib import Path
import tempfile

from finite_contact_multimap_joint import GROUPS
from finite_contact_stability import CONDITIONS
from finite_contact_stability_replication import aggregate_records, resume_plan, ROOT, IDENTITY_FILES
from finite_contact_multimap_joint import sha256


class StabilityReplicationChecks(unittest.TestCase):
    def test_resource_resume_preserves_original_and_refuses_training_source_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);(root/'checks').mkdir();(root/'seed_1').mkdir()
            original={name:sha256(ROOT/name) for name in IDENTITY_FILES}
            original['finite_contact_stability_replication.py']='previous orchestration version'
            (root/'checks/source_identity_before.json').write_text(json.dumps(original))
            plan={'concurrent_workers':3,'runs':[{'seed':1,'directory':str(root/'seed_1')}],
                  'status':'interrupted_or_failed'}
            (root/'replication.json').write_text(json.dumps(plan))
            (root/'seed_1/plan.json').write_text(json.dumps({'workers':3,'training_maps':100,'updates_per_map':20}))
            restored=resume_plan(root,2)
            self.assertEqual(restored['concurrent_workers'],2)
            self.assertEqual(len(restored['resource_changes']),1)
            self.assertEqual(json.loads((root/'checks/source_identity_before.json').read_text()),original)
            child=json.loads((root/'seed_1/plan.json').read_text())
            self.assertEqual((child['workers'],child['training_maps'],child['updates_per_map']),(2,100,20))
            self.assertEqual(len(resume_plan(root,2)['resource_changes']),1)
            active=json.loads((root/'replication.json').read_text())['source_identity_file']
            identity=json.loads((root/active).read_text());identity['finite_contact/learning.py']='different algorithm'
            (root/active).write_text(json.dumps(identity))
            before=(root/'replication.json').read_bytes()
            with self.assertRaises(ValueError):resume_plan(root,1)
            self.assertEqual((root/'replication.json').read_bytes(),before)

    def records(self):
        records=[]
        for seed in (3,1,2):
            groups={}
            for group in GROUPS:
                groups[group]={}
                for condition in CONDITIONS:
                    tail=condition=='map_tail'
                    groups[group][condition]={'new_sac_updates':2000,'cumulative_sac_updates':12895,
                        'validation_final':{'maps':20,'successes':16+seed+int(tail),
                            'mean_capped_time':100+10*seed-5*int(tail),
                            'mean_minimum_coverage':.99,'mean_coverage_deficit':.1}}
            records.append({'seed':seed,'groups':groups})
        return records

    def test_both_sampling_conditions_use_seed_units_and_paired_changes(self):
        result=aggregate_records(self.records())
        for group in GROUPS:
            a=result['conditions']['pooled']['groups'][group]
            b=result['conditions']['map_tail']['groups'][group]
            self.assertEqual(a['successes']['per_seed'],[17,18,19])
            self.assertEqual(b['successes']['per_seed'],[18,19,20])
            self.assertEqual(a['successes']['sample_std'],1.)
            self.assertEqual(a['mean_capped_time']['mean'],120.)
            self.assertEqual(a['mean_capped_time']['sample_std'],10.)
            changes=result['paired_sampling_changes'][group]
            self.assertEqual([r['seed'] for r in changes],[1,2,3])
            self.assertTrue(all(r['success_gain']==1 and r['time_saved_seconds']==5 for r in changes))

    def test_partial_budget_missing_seed_and_incomplete_evaluation_are_rejected(self):
        original=self.records()
        with self.assertRaises(ValueError):aggregate_records(original[:2])
        duplicate=copy.deepcopy(original);duplicate[0]['seed']=1
        with self.assertRaises(ValueError):aggregate_records(duplicate)
        for field,value in [('new_sac_updates',1999),('cumulative_sac_updates',12894)]:
            records=copy.deepcopy(original)
            records[0]['groups']['AB_learned']['map_tail'][field]=value
            with self.assertRaises(ValueError):aggregate_records(records)
        for field,value in [('maps',19),('mean_capped_time',float('nan')),('successes',21)]:
            records=copy.deepcopy(original)
            records[0]['groups']['A_rule']['pooled']['validation_final'][field]=value
            with self.assertRaises(ValueError):aggregate_records(records)


if __name__=='__main__':unittest.main()
