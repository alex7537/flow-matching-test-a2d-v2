import unittest,tempfile,json,tarfile,io,hashlib
from pathlib import Path
import loop

def plan(n=3):
 return dict(episodes_per_configuration=n,modes=['step_gt'],max_steps=300,horizon=16,policy_seed=42,scene_seeds=list(range(42,42+n)),metric='ever lift>=0.05m for 5 action observations',models={'cfm':'x'},expected_hashes={'cfm':'sha'})
def row(ep=0,outcome='completed',success=False):
 return dict(model='cfm',mode='step_gt',episode=ep,scene_seed=42+ep,policy_seed=42,outcome=outcome,success=success,max_streak=5 if success else 0,executed=300 if outcome=='completed' else 16,contacts=1 if success else 0)
class Checks(unittest.TestCase):
 def test_denominators(self):
  g=loop.aggregate([row(0,success=True),row(1,'policy_rejected'),row(2,'reset_invalid')],plan())['cfm']
  self.assertEqual((g['attempts'],g['valid'],g['invalid'],g['successes']),(3,2,1,1));self.assertTrue(g['coverage_complete'])
 def test_rejected_after_success(self):
  self.assertEqual(loop.aggregate([row(0,'policy_rejected',True)],plan())['cfm']['successes'],1)
 def test_duplicate(self):
  with self.assertRaisesRegex(ValueError,'Duplicate'):loop.aggregate([row(),row()],plan())
 def test_bad_identity(self):
  r=row();r['scene_seed']=99
  with self.assertRaisesRegex(ValueError,'Seed'):loop.aggregate([r],plan())
 def test_metric_contradiction(self):
  r=row();r['success']=True
  with self.assertRaisesRegex(ValueError,'contradiction'):loop.aggregate([r],plan())
 def test_false_completed(self):
  r=row();r['executed']=16
  with self.assertRaisesRegex(ValueError,'missing actions'):loop.aggregate([r],plan())
 def test_partial_snapshot(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'x';p.write_bytes(b'{"a":1}\n{"a":');_,rs,n=loop.snapshot_rows(p)
   self.assertEqual(rs,[{'a':1}]);self.assertGreater(n,0)
 def test_zero_valid(self):
  g=loop.aggregate([row(0,'reset_invalid')],plan())['cfm'];self.assertIsNone(g['wilson95']);self.assertFalse(g['coverage_complete'])
 def test_archive_integrity(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'b.tgz';items={x:b'abc' for x in ['ckpt.pt','config.yaml','norm_stats.json']}
   manifest={'files':{k:{'bytes':len(v),'sha256':hashlib.sha256(v).hexdigest()} for k,v in items.items()}}
   items['manifest.json']=json.dumps(manifest).encode()
   with tarfile.open(p,'w:gz') as t:
    for name,data in items.items():
     info=tarfile.TarInfo('bundle/'+name);info.size=len(data);t.addfile(info,io.BytesIO(data))
   self.assertTrue(loop.bundle_receipt(p,loop.digest(p))['internal_hashes_verified'])
   with self.assertRaisesRegex(ValueError,'hash mismatch'):loop.bundle_receipt(p,'0'*64)
if __name__=='__main__':unittest.main()
