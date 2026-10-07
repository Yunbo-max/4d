"""Observed process-tree RSS and output-directory disk use, sampled each second."""
import json
import os
from pathlib import Path
import shutil
import threading
import time

class HostSamples:
    def __init__(self, root: Path, output: Path):
        self.root,self.output=Path(root),Path(output)
        self.stop=threading.Event();self.thread=None;self.rows=[];self.errors=[]

    def capture(self):
        import psutil
        try:
            parent=psutil.Process(os.getpid())
            processes=[parent]+parent.children(recursive=True)
            rss=0
            for process in processes:
                try:rss+=process.memory_info().rss
                except psutil.NoSuchProcess:pass
            size=0
            for path in self.root.rglob('*'):
                try:
                    if path.is_file() and not path.is_symlink():size+=path.stat().st_size
                except FileNotFoundError:pass
            row={'epoch':time.time(),'process_tree_rss_bytes':rss,'output_bytes':size,
                 'free_disk_bytes':shutil.disk_usage(self.root).free}
            self.rows.append(row)
        except Exception as error:
            row={'epoch':time.time(),'error':type(error).__name__+': '+str(error)}
            self.errors.append(row['error'])
        with self.output.open('a') as stream:stream.write(json.dumps(row)+'\n')

    def poll(self):
        while not self.stop.wait(1.):self.capture()

    def start(self):
        self.capture()
        if self.errors:raise RuntimeError('Host resource telemetry unavailable')
        self.thread=threading.Thread(target=self.poll,daemon=True);self.thread.start()

    def close(self):
        self.stop.set()
        if self.thread is not None:self.thread.join()
        self.capture()

    def summary(self):
        return {'sample_interval_seconds':1,'samples':len(self.rows),'exact_peak':False,
            'observed_peak_rss_bytes':max((r['process_tree_rss_bytes'] for r in self.rows),default=None),
            'observed_peak_output_bytes':max((r['output_bytes'] for r in self.rows),default=None),
            'minimum_free_disk_bytes':min((r['free_disk_bytes'] for r in self.rows),default=None),
            'errors':self.errors}
