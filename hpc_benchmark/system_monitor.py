import csv
import shutil
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

import psutil


class SystemMonitor:
    def __init__(self, process, output_file, interval=1.0):
        self.process = process
        self.output_file = Path(output_file)
        self.interval = interval
        self.samples = []
        self.running = False
        self.thread = None
        self.start_time = None

    def start(self):
        self.running = True
        self.start_time = time.perf_counter()
        psutil.cpu_percent(interval=None)
        self.thread = threading.Thread(target=self._sample_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()

        self._write_csv()

    def _gpu_metrics(self):
        executable = shutil.which("nvidia-smi")
        if not executable:
            return None, None, None

        try:
            result = subprocess.run(
                [
                    executable,
                    "--query-gpu=utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=3,
                check=True,
            )

            values = result.stdout.strip().splitlines()[0].split(",")
            return (
                float(values[0].strip()),
                float(values[1].strip()),
                float(values[2].strip()),
            )
        except (subprocess.SubprocessError, ValueError, IndexError, OSError):
            return None, None, None

    def _process_metrics(self):
        cpu = 0.0
        memory = 0

        try:
            root = psutil.Process(self.process.pid)
            processes = [root] + root.children(recursive=True)

            for proc in processes:
                try:
                    cpu += proc.cpu_percent(interval=None)
                    memory += proc.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

        return cpu, memory / (1024 ** 2)

    def _sample_loop(self):
        while self.running:
            elapsed = time.perf_counter() - self.start_time
            gpu_util, gpu_used, gpu_total = self._gpu_metrics()
            process_cpu, process_memory = self._process_metrics()
            system_memory = psutil.virtual_memory()

            self.samples.append({
                "timestamp": datetime.now().astimezone().isoformat(),
                "elapsed_seconds": round(elapsed, 3),
                "system_cpu_percent": psutil.cpu_percent(interval=None),
                "system_ram_used_mb": round(system_memory.used / (1024 ** 2), 2),
                "system_ram_percent": system_memory.percent,
                "graphrag_process_tree_cpu_percent": round(process_cpu, 2),
                "graphrag_process_tree_ram_mb": round(process_memory, 2),
                "gpu_utilization_percent": gpu_util,
                "gpu_memory_used_mb": gpu_used,
                "gpu_memory_total_mb": gpu_total,
            })

            time.sleep(self.interval)

    def _write_csv(self):
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

        fields = [
            "timestamp", "elapsed_seconds",
            "system_cpu_percent", "system_ram_used_mb", "system_ram_percent",
            "graphrag_process_tree_cpu_percent",
            "graphrag_process_tree_ram_mb",
            "gpu_utilization_percent", "gpu_memory_used_mb",
            "gpu_memory_total_mb",
        ]

        with self.output_file.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.samples)

        print(f"Resource CSV: {self.output_file}")
        print(f"Resource samples collected: {len(self.samples)}")
