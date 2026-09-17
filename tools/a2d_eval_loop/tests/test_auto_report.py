import importlib.util,json,tempfile,unittest
from pathlib import Path
spec=importlib.util.spec_from_file_location('auto_report',Path(__file__).resolve().parents[1]/'watch_eval_to_obsidian.py')
reporter=importlib.util.module_from_spec(spec);spec.loader.exec_module(reporter)
class ReportTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
  self.batch=Path(self.tmp.name)/'batch';self.vault=Path(self.tmp.name)/'vault'
  (self.batch/'box').mkdir(parents=True);self.vault.mkdir()
  (self.batch/'box/plan.json').write_text(json.dumps(dict(models={'dp':'bundle'},episodes_per_configuration=2,modes=['step_gt'])))
  self.row=dict(model='dp',mode='step_gt',episode=0,outcome='completed',success=True,contacts=8,max_lift=.1)
 def ledger(self,rows):
  (self.batch/'box/episodes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
 def test_running_and_interrupted(self):
  self.ledger([self.row]);x=reporter.report(self.batch,self.vault,True)
  self.assertEqual(x['state'],'进行中');self.assertFalse(x['complete'])
  x=reporter.report(self.batch,self.vault,False);self.assertEqual(x['state'],'异常停止／未完成')
  self.assertEqual((self.batch/'AUTO_REPORT.md').read_bytes(),Path(x['obsidian_note']).read_bytes())
 def test_complete_preserves_invalid_denominator(self):
  self.ledger([self.row,dict(self.row,episode=1,outcome='infrastructure_error',success=False)])
  x=reporter.report(self.batch,self.vault,False);self.assertTrue(x['complete']);self.assertEqual(x['rows'][0]['valid'],1);self.assertEqual(x['rows'][0]['invalid'],1)
 def test_duplicate_and_truncated_lines(self):
  self.ledger([self.row,self.row])
  with (self.batch/'box/episodes.jsonl').open('a') as f:f.write('{"partial":')
  x=reporter.report(self.batch,self.vault,False);self.assertEqual(x['recorded'],1);self.assertEqual(len(x['warnings']),2);self.assertFalse(x['complete'])
 def test_pause_and_gallery_review(self):
  self.ledger([self.row])
  for q,expected in [('paused','暂停'),('awaiting_human_review','采集结束／等待确认')]:
   (self.batch/'queue_status.json').write_text(json.dumps({'state':q}))
   self.assertEqual(reporter.report(self.batch,self.vault,False)['state'],expected)
if __name__=='__main__':unittest.main()
