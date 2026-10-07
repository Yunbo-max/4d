import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from research_math.unit_resources import HostSamples
from research_math.complete_unit_plan import validate_inventory

class UnitResourcesPlanTests(unittest.TestCase):
    def test_resource_capture_records_process_ram_and_output_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'payload').write_bytes(b'x'*4096)
            sample=HostSamples(root,root/'host.jsonl')
            sample.start();sample.close()
            summary=sample.summary()
            self.assertGreater(summary['observed_peak_rss_bytes'],0)
            self.assertGreaterEqual(summary['observed_peak_output_bytes'],4096)
            self.assertGreater(summary['minimum_free_disk_bytes'],0)
            self.assertFalse(summary['exact_peak'])
            self.assertGreaterEqual(len((root/'host.jsonl').read_text().splitlines()),2)

    def test_unexecuted_admissions_do_not_allow_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            paths={}
            for name in ('snapshot_admission','dataset_semantics','unit_manifest'):
                paths[name]=root/(name+'.json');paths[name].write_text(json.dumps({'status':'generated_unexecuted'}))
            with self.assertRaises(ValueError):validate_inventory(SimpleNamespace(**paths))
