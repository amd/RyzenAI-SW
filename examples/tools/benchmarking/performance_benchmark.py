import os
import onnxruntime as rt
import numpy as np
import time
from pathlib import Path
import queue
import concurrent.futures
import threading
import math
import subprocess
from typing import Optional, Protocol, Any, Dict, List, Mapping, Sequence
from itertools import repeat
try:
    import keyboard  # type: ignore
    _HAS_KEYBOARD = True
except Exception:
    keyboard = None  # type: ignore
    _HAS_KEYBOARD = False
from concurrent.futures import ThreadPoolExecutor

def _find_vaip_config_from_env() -> Optional[str]:
    """Locate vaip_config.json under the installation path, if present."""
    install_dir = os.environ.get("RYZEN_AI_INSTALLATION_PATH")
    if not install_dir:
        return None
    install_dir = install_dir.strip().strip('"')
    if not os.path.isdir(install_dir):
        return None
    for root, _, files in os.walk(install_dir):
        if "vaip_config.json" in files:
            return os.path.join(root, "vaip_config.json")
    return None

from utilities import *

class Benchmark:
    def __init__(
        self,
        args,
        apu_type,
        power_meter: Optional[PowerMeter] | None = None,
        defaults: Optional[Dict[str, Any]] = None,
    ):
        self.release = 22
        self.apu_type = apu_type
        self.args = args
        self.power_meter = power_meter
        self.defaults = defaults

        self.model_input = {}     
        self.session: SessionLike = DummySession()
        self.inferences = 0
        self.nctlist = []
        self.final_total_time = []
        self.latencies_sum = 0
        self.lat_results = 0
        self.thr_results = 0
        self.requested_images = 0
        self.scheduled_images = 0
        self.processed_images = 0

        self.measurement = {}
        self.config_file = None
        self.TABLEW = 51

    def run(self):
        # set environment variables
        set_environment_variable(self.apu_type)

        self.config_file = _find_vaip_config_from_env()
        if not self.config_file:
            install_dir = os.environ.get("RYZEN_AI_INSTALLATION_PATH", "")
            fallback = os.path.join(install_dir, "voe-4.0-win_amd64", "vaip_config.json") if install_dir else ""
            if fallback and os.path.isfile(fallback):
                self.config_file = fallback
            else:
                ggprint("[WARN] vaip_config.json not found under installation path.")

        # environment control
        check_env(self.release, self.args)
        # input parameters control
        if self.defaults is None:
            raise RuntimeError("Benchmark defaults not provided; pass defaults to Benchmark.__init__")
        check_args(self.args, self.defaults)
        # delete old measurement
        del_file("report_benchmark.json")

        try:
            resolved_model = ensure_model_exists(
                model_selector=getattr(self.args, "model", ""),
                model_path=getattr(self.args, "model_path", ""),
                models_root="models",
                overwrite=(getattr(self.args, "renew", "1") == "1"),
            )
        except Exception as exc:
            ggprint(f"[ERROR] Unable to locate model: {exc}")
            raise

        self.args.model_path = str(resolved_model)
        self.args.model = str(resolved_model)
        ggprint(f"[MODEL] Using: {self.args.model_path}")

        if getattr(self.args, "force_batch", 1) == 1:
            try:
                forced_path = fix_model_batch1(self.args.model_path)
                self.args.model_path = forced_path
                self.args.model = forced_path
                ggprint(f"[MODEL] Forced batch=1 -> {self.args.model_path}")
            except DynamicInputError as exc:
                ggprint(f"[ERROR] {exc}")
                raise
            except Exception as exc:
                ggprint(f"[WARN] Unable to normalize batch dimension: {exc}")

        if self.args.execution_provider == "VitisAIEP":
            os.environ['XLNX_ONNX_EP_VERBOSE'] = '0'
            os.environ['XLNX_ENABLE_STAT_LOG'] = '0'
            quantized_path = ggquantize(self.args)
            self.args.model_path = quantized_path
            self.args.model = quantized_path
            if self.args.renew == "1":
                cache_dir = os.path.join(Path(__file__).parent.resolve(), "cache", os.path.basename(self.args.model_path))
                cancelcache(cache_dir)

        if self.args.execution_provider == "ZenDNN":
            set_ZEN_env()

        # Warmup. Skipping in case of infinite loop
        if getattr(self.args, "infinite", "1") == "1":
            ggprint("Infinite Loop: skipping warmup")
        else:
            ggprint("Single run: warmup")

        self.profile()
    def timed_inference(self, fn, output_names, input_dict_B):
        timed_inference_start = time.perf_counter()
        output = fn(output_names, input_dict_B)
        end = time.perf_counter()
        self.latencies_sum += end - timed_inference_start
        self.inferences += 1
        if self.args.verbose == "2":
            ggprint(f"end - timed_inference_start = {end - timed_inference_start}")
            ggprint(f"num_completed_tasks = {self.inferences}")
        return 1


    def profile(self):
        so = rt.SessionOptions()
        # Enable profiling by providing the profiling options
        # so.enable_profiling = True

        EP_List = []

        if self.args.execution_provider == "iGPU":
            ggprint(f"Provider = {self.args.execution_provider}")
            providers = [("DmlExecutionProvider", {"device_id": 0})]
            # override:
            self.args.threads = 1  # iGPU does not take advantage of async multi-threading
            ggprint("Check with Task Manager which GPU is used!")
            self.session = rt.InferenceSession(self.args.model_path, so, providers=providers)
        elif self.args.execution_provider == "dGPU":
            ggprint(f"Provider = {self.args.execution_provider}")
            # providers = ['DmlExecutionProvider', {'device_id':1}]
            providers = [("DmlExecutionProvider", {"device_id": 1})]
            ggprint("Check with Task Manager which GPU is used!")
            self.session = rt.InferenceSession(self.args.model_path, providers=providers)
        elif self.args.execution_provider == "ZenDNN":
            ggprint(f"Provider = {self.args.execution_provider}")
            providers = ["ZendnnExecutionProvider"]
            self.session = rt.InferenceSession(self.args.model_path, providers=providers)
        elif self.args.execution_provider == "CPU":
            ggprint(f"Provider = {self.args.execution_provider}")
            providers = ["CPUExecutionProvider"]
            # ref: https://onnxruntime.ai/docs/performance/tune-performance/threading.html
            so.intra_op_num_threads = self.args.intra_op_num_threads
            self.session = rt.InferenceSession(self.args.model_path, so, providers=providers)
        elif self.args.execution_provider == "ZenDPU":
            ggprint(f"Provider = {self.args.execution_provider}")
            providers = ["ZendpuExecutionProvider"]
            self.session = rt.InferenceSession(self.args.model_path, providers=providers)
        elif self.args.execution_provider == "VitisAIEP":
            ggprint(f"Provider = {self.args.execution_provider}")
            ggprint(f"config path = {self.args.config}")
            EP_List.append("VitisAIExecutionProvider")

            cache_dir = os.path.join(Path(__file__).parent.resolve(), "cache", os.path.basename(self.args.model_path))

            so.intra_op_num_threads = self.args.intra_op_num_threads


            if self.apu_type == 'PHX/HPT':
                ggprint("Setting environment for PHX/HPT")
                provider_options = [{
                        'cache_dir': str(cache_dir),
                        'cache_key': 'modelcachekey',
                        'target': 'X1',
                        'config_file': self.args.config,
                        'enable_cache_file_io_in_mem': 0,
                        'xclbin': os.environ['XLNX_VART_FIRMWARE'],
                    }]
            else:
                provider_options = [{
                        'cache_dir': str(cache_dir),
                        'cache_key': 'modelcachekey',
                        'config_file': self.args.config,
                        'enable_cache_file_io_in_mem': 0,
                    }]


            self.session = rt.InferenceSession(
                self.args.model_path,
                so,
                providers=EP_List,
                provider_options=provider_options
            )

        output_nodes = self.session.get_outputs()
        output_names = [n.name for n in output_nodes]

        #feed = generate_dummy_inputs(self.session, orig_im_size=(480, 640))
        self.model_input = generate_dummy_inputs(self.session, orig_im_size=(480, 640))
        feed = self.model_input

        target_delay = self.args.min_interval

        requested_images = max(0, self.args.num)
        total_steps = math.ceil(requested_images / self.args.batchsize) if requested_images else 0
        actual_images = total_steps * self.args.batchsize
        self.requested_images = requested_images
        self.scheduled_images = actual_images
        r = range(total_steps)
        ggprint(f"Executing {total_steps} steps, with each step involving the inference of {self.args.batchsize} images")
        if actual_images != requested_images:
            ggprint(f"[INFO] Preloading {actual_images} images (requested {requested_images}) to align with batch size {self.args.batchsize}.")


        if getattr(self.args, "infinite", "1") == "1":
            if _HAS_KEYBOARD:
                ggprint("Infinite loop: Press q to Stop.")
            else:
                ggprint("Infinite loop: press Ctrl+C to stop (keyboard module not available).")

        profile_start = time.perf_counter()
        printonce = True

        try:
            while True:
                if self.args.no_inference == "1":
                    if printonce:
                        printonce = False
                        ggprint("Running without inference for power baseline")
                else:
                    tasks = repeat(feed, len(r))

                    start = time.perf_counter()
                    if target_delay > 0:
                        elapsed = 0.0
                        while elapsed < target_delay:
                            time.sleep(max(0.0, target_delay - elapsed))
                            elapsed = time.perf_counter() - start

                    with ThreadPoolExecutor(max_workers=self.args.threads) as pool:
                        futures = [pool.submit(self.timed_inference, self.session.run, output_names, task)
                                   for task in tasks]
                        _ = [f.result() for f in futures]

                    self.nctlist.append(self.inferences)
                    total_run_time = time.perf_counter() - start
                    self.final_total_time.append(total_run_time)

                    if self.args.verbose in ("1", "2"):
                        ggprint(f"{len(self.nctlist)} - test time = {(time.perf_counter() - profile_start):.2f} / {self.args.timelimit:.2f} sec")

                    if getattr(self.args, "infinite", "1") != "1":
                        break
                    if _HAS_KEYBOARD and keyboard and keyboard.is_pressed("q"):
                        ggprint("Key pressed. Stopping the loop.")
                        break
                    if (time.perf_counter() - profile_start) >= self.args.timelimit:
                        ggprint(f"\nLimit time {self.args.timelimit} s reached")
                        break
        except KeyboardInterrupt:
            ggprint("Interrupted by user (Ctrl+C).")

        if self.args.no_inference == "0":
            self.processed_images = self.inferences * self.args.batchsize
            if self.inferences:
                self.lat_results = 1000 * self.latencies_sum / self.inferences
            else:
                self.lat_results = 0
            total_time = sum(self.final_total_time)
            self.thr_results = (self.processed_images / total_time) if total_time else 0
        else:
            self.lat_results = 0
            self.thr_results = 0
            self.processed_images = 0

        scheduled_images = self.scheduled_images
        processed_images = self.processed_images

        if self.args.verbose in ("1", "2"):
            print("\n")
            print(f'requested images = {self.requested_images}')
            print(f'preloaded images = {scheduled_images}')
            print(f'scheduled batches = {total_steps}')
            print(f'completed batches = {self.inferences}')
            print(f'processed images = {processed_images}')
            print(f'cumulative time spent by all threads ={self.latencies_sum}')
            print(f'num_completed_tasks ={self.inferences}')
            if self.inferences:
                print(f'average time spent on each task ms={1000 * self.latencies_sum / self.inferences}')
                print(f'average latency= {self.lat_results:.2f} ms')
            else:
                print('average time spent on each task ms=0.0')
                print('average latency= 0.00 ms')
            print("\n")
            print(f'number of processed batches recorded ={len(self.final_total_time)}')
            if self.final_total_time:
                print(f'total inference wall time ={sum(self.final_total_time):.6f} s')
            print("\n")
            print(f'average throughput = {self.thr_results:.2f} fps')
            if processed_images != scheduled_images:
                print(f'note: processed images ({processed_images}) differ from scheduled ({scheduled_images}) due to early exit conditions')

        # session.end_profiling()

    def report(self):
        if self.args.no_inference == "0":
            self.measurement = meas_init(self.args, self.release, self.thr_results, self.lat_results, "", self.model_input)
            self.measurement["results"]["performance"]["images_requested"] = self.requested_images
            self.measurement["results"]["performance"]["images_preloaded"] = self.scheduled_images
            self.measurement["results"]["performance"]["images_processed"] = self.processed_images
       
            test_summary =  [["model", Path(self.args.model_path).stem],
                            ["threads", self.args.threads],
                            ["APU", self.apu_type]]       

            performance  =  [["throughput", f"{self.thr_results:.2f} [fps]"],
                            ["latency", f"{self.lat_results:.2f} [ms]"],
                            ["processed images", self.processed_images],
                            ["preloaded images", self.scheduled_images],
                            ["requested images", self.requested_images]]

            # disabled efficency report ...
            #comp_efficency = measurement['model']['Forward_FLOPs']*thr_results/(tops_peak(device)*1e12)
            #efficency    = [["model", measurement['model']['name']],
            #                ["model operations", f"{measurement['model']['Forward_FLOPs']/1e6:.2f} [MFLOPs]"],
            #                ["device", device],
            #                ["theoretical compute peak", f"{tops_peak(device)} [TOPs]"],
            #                ["compute efficency", f"{comp_efficency*100:.2f}%"]]
            #self.measurement['model']['device'] = self.apu_type
            #self.measurement['model']['core'] = self.args.core
            #measurement['model']['comp_peak'] = tops_peak(apu_type)
            #measurement['model']['comp_efficency'] = comp_efficency*100

            print(Colors.YELLOW)
            tableprint("TEST SUMMARY", test_summary, self.TABLEW)
            tableprint("PERFORMANCE", performance, self.TABLEW)
            #tableprint("EFFICENCY", efficency, self.TABLEW)

            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            if self.args.power in ["AGM", "BOTH"]:
                if not self.power_meter:
                    ggprint("[WARN] Power measurement requested (AGM) but PowerMeter instance is missing; skipping AGM analysis.")
                else:
                    recordfilename = self.power_meter.agm_log_file

                    medians = getmedians(recordfilename)

                    cpu_perf_pow = self.thr_results / medians["CPU GPU Power"]
                    npu_perf_pow = self.thr_results / medians["NPU Power"]
                    mem_perf_pow = self.thr_results / medians["MEM_PHY Power"]
                    apu_perf_pow = self.thr_results / medians["APU Power"]
                    apuenergy = 1000 / apu_perf_pow
                    cpuenergy = 1000 / cpu_perf_pow
                    npuenergy = 1000 / npu_perf_pow
                    memenergy = 1000 / mem_perf_pow

                    powertable = [
                        [f"CPU:", f"{cpu_perf_pow:.3f} [fps/W]"],
                        [f"NPU:", f"{npu_perf_pow:.3f} [fps/W]"],
                        [f"MEM:", f"{mem_perf_pow:.3f} [fps/W]"],
                        [f"APU:", f"{apu_perf_pow:.3f} [fps/W]"]]

                    energytable = [
                        [f"CPU:", f"{cpuenergy:.3f} [mJ/f]"],
                        [f"NPU:", f"{npuenergy:.3f} [mJ/f]"],
                        [f"MEM:", f"{memenergy:.3f} [mJ/f]"],
                        [f"APU:", f"{apuenergy:.3f} [mJ/f]"]]

                    tableprint("Energy [mJ] per frame [f]", energytable, self.TABLEW)

                    plotefficency(mem_perf_pow, npu_perf_pow, cpu_perf_pow, apu_perf_pow, recordfilename)
                    plotenergy(memenergy, npuenergy, cpuenergy, apuenergy, recordfilename)
                    time_violin(recordfilename, ["MPNPUCLK","NPUHCLK","FCLK","LCLK"], "Frequencies", "clocks", "MHz")
                    time_violin(recordfilename, ["APU Power", "CPU GPU Power", "NPU Power","MEM_PHY Power"], "Powers", "Power rails", "Watt")

                    self.measurement["results"]["efficency perf/W"]["apu_perf_pow"] = apu_perf_pow
                    self.measurement["results"]["energy mJ/frame"]["apu"] = apuenergy
                    self.measurement["results"]["energy mJ/frame"]["cpu"] = cpuenergy
                    self.measurement["results"]["energy mJ/frame"]["npu"] = npuenergy
                    self.measurement["results"]["energy mJ/frame"]["mem"] = memenergy

                    self.measurement["system"]["frequency"]["MPNPUCLK"] = medians["MPNPUCLK"]
                    self.measurement["system"]["frequency"]["NPUHCLK"] = medians["NPUHCLK"]
                    self.measurement["system"]["frequency"]["FCLK"] = medians["FCLK"]
                    self.measurement["system"]["frequency"]["LCLK"] = medians["LCLK"]

            if self.args.power in ["HWINFO", "BOTH"]:
                if not self.power_meter:
                    ggprint("[WARN] Power measurement requested (HWINFO) but PowerMeter instance is missing; skipping HWINFO analysis.")
                else:
                    recordfilename = self.power_meter.hwinfo_log_file

                    medians = getmedians(recordfilename)

                    cpu_perf_pow = self.thr_results / medians["CPU GPU Power"]
                    npu_perf_pow = self.thr_results / medians["NPU Power"]
                    apu_perf_pow = self.thr_results / medians["APU Power"]
                    apuenergy = 1000 / apu_perf_pow
                    cpuenergy = 1000 / cpu_perf_pow
                    npuenergy = 1000 / npu_perf_pow

                    powertable = [
                        [f"CPU:", f"{cpu_perf_pow:.3f} [fps/W]"],
                        [f"NPU:", f"{npu_perf_pow:.3f} [fps/W]"],
                        [f"APU:", f"{apu_perf_pow:.3f} [fps/W]"]]

                    energytable = [
                        [f"CPU:", f"{cpuenergy:.3f} [mJ/f]"],
                        [f"NPU:", f"{npuenergy:.3f} [mJ/f]"],
                        [f"APU:", f"{apuenergy:.3f} [mJ/f]"]]

                    tableprint("Energy [mJ] per frame [f]", energytable, self.TABLEW)

                    plotefficency_hwinfo(npu_perf_pow, cpu_perf_pow, apu_perf_pow, recordfilename)
                    plotenergy_hwinfo(npuenergy, cpuenergy, apuenergy, recordfilename)
                    time_violin(recordfilename, ["NPUHCLK"], "Frequencies", "clocks", "MHz")
                    time_violin(recordfilename, ["APU Power", "CPU GPU Power", "NPU Power"], "Powers", "Power rails", "Watt")

                    self.measurement["results"]["efficency perf/W"]["apu_perf_pow"] = apu_perf_pow
                    self.measurement["results"]["energy mJ/frame"]["apu"] = apuenergy
                    self.measurement["results"]["energy mJ/frame"]["cpu"] = cpuenergy
                    self.measurement["results"]["energy mJ/frame"]["npu"] = npuenergy
                    self.measurement["results"]["energy mJ/frame"]["mem"] = "N/A"

                    self.measurement["system"]["frequency"]["MPNPUCLK"] = "N/A"
                    self.measurement["system"]["frequency"]["NPUHCLK"] = medians["NPUHCLK"]
                    self.measurement["system"]["frequency"]["FCLK"] = "N/A"
                    self.measurement["system"]["frequency"]["LCLK"] = "N/A"

            #if args.execution_provider == "VitisAIEP" and args.cpp!="1":
            if self.args.execution_provider == "VitisAIEP":
                cache_dir = os.path.join(Path(__file__).parent.resolve(), "cache", os.path.basename(self.args.model_path))
                try:
                    with open(os.path.join(cache_dir, r"modelcachekey\vitisai_ep_report.json"), "r") as json_file:
                        vitisaireport = json.load(json_file)
                except Exception as e:
                    print(f"Error loading JSON file: {e}")
                    vitisaireport = {
                        "deviceStat": [
                            {"nodeNum": "N/A"},                  
                            {"name":    "CPU", "nodeNum": "N/A"},
                            {"name":    "NPU", "nodeNum": "N/A"},
                        ]
                    }


                self.measurement["vitisai"]['all'] = vitisaireport['deviceStat'][0]['nodeNum']
                vitisai_ep_report = [["total nodes", f"{vitisaireport['deviceStat'][0]['nodeNum']}"]]
                for i, device in enumerate(vitisaireport["deviceStat"][1:]):
                    vitisai_ep_report.append([f"{device['name']}", f"{device['nodeNum']}"])
                    self.measurement["vitisai"][device['name']] = device['nodeNum']

                tableprint("NODES DISTRIBUTION", vitisai_ep_report, self.TABLEW)
            print(Colors.RESET)
            save_result_json(self.measurement, "report_performance.json")
            if self.args.log_csv=="1":
                appendcsv(self.measurement, self.args)
        else:
            print(Colors.YELLOW + "Test with no inference (for power baseline estimation) completed")
            print(Colors.RESET)


if __name__ == "__main__":
    apu_type = get_apu_info()
    args, defaults = parse_args(apu_type)
        
    pm = PowerMeter(tool=args.power, suffix="_meas")
    bench = Benchmark(args, apu_type, pm, defaults=defaults)

    pm.start()
    bench.run()
    pm.stop()
    bench.report()
