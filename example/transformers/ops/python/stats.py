#
# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
#

import multiprocessing as mp
import os
import time

import RyzenAI


class CPUStats:
    def __init__(self, name, PID, interval=1, collect_data=True):
        self.name = name
        self.pid = PID
        self.begin_mem = 0
        self.end_mem = 0
        self.peak_mem = 0
        self.interval = interval
        self.collect_data = collect_data
        self.cpu_stats = RyzenAI.CPULoad(PID)
        self.cpu_load = self.cpu_stats.get_cpu_load()
        self.stop_signal = mp.Value("i", 0)
        self.proc = mp.Process(target=self.process, args=())

    def __enter__(self):
        if self.collect_data == True:
            self.begin_mem = RyzenAI.get_sys_commit_mem().commit_memory
            self.end_mem = self.begin_mem
            self.peak_mem = self.begin_mem
            self.cpu_load = self.cpu_stats.get_cpu_load()
            self.proc.start()
        return self

    def process(self):
        while True:
            time.sleep(self.interval)
            with self.stop_signal.get_lock():
                stop_value = self.stop_signal.value
            if stop_value == 1:
                break
            else:
                self.end_mem = RyzenAI.get_sys_commit_mem().commit_memory
                self.peak_mem = max(
                    self.peak_mem, RyzenAI.get_sys_commit_mem().commit_memory
                )

    def stop(self):
        if self.collect_data == True:
            with self.stop_signal.get_lock():
                self.stop_signal.value = 1
            self.proc.join()

    def __exit__(self, a, b, c):
        if self.collect_data == True:
            self.stop()
            self.end_mem = RyzenAI.get_sys_commit_mem().commit_memory
            self.peak_mem = max(self.peak_mem, self.end_mem)
            self.cpu_load = self.cpu_stats.get_cpu_load()

    def get_data(self):
        result = {}
        result["sys_commit_mem_delta (B)"] = self.end_mem - self.begin_mem
        result["peak_sys_commit_mem_delta (B)"] = self.peak_mem - self.begin_mem
        result["proc_cpu_load"] = self.cpu_load
        return result

    def print_data(self):
        res = self.get_data()
        print("CPUStats :", self.name)
        for key, val in res.items():
            print("  ", key, "=", val)
