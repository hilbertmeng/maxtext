import json,unittest
import numpy as np
from diagnostic_json import dumps
class DiagnosticJsonTest(unittest.TestCase):
 def test_real_fidelity_metadata_roundtrip(self):
  record=dict(fidelity=[dict(weak_power_relative=np.float32(0.0003),head_relative=0.002)],step=np.int64(33000),accepted=np.bool_(True))
  q=json.loads(dumps(record));self.assertEqual(q['step'],33000);self.assertIs(q['accepted'],True);self.assertAlmostEqual(q['fidelity'][0]['weak_power_relative'],0.0003,places=8)
 def test_nonfinite_and_unexpected_arrays_fail(self):
  with self.assertRaises(ValueError):dumps(dict(value=np.float32(np.nan)))
  with self.assertRaises(TypeError):dumps(dict(value=np.zeros(3)))
if __name__=='__main__':unittest.main()
