# ready for public

release = 18

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
import keyboard

from utilities import *

# global variables
global latencies_sum
global inferences
global nctlist
nctlist = []
global finaltottime
finaltottime=[]

def timed_inference(fn, output_names, input_dict_B):
    global latencies_sum
    global inferences
    timed_inference_start = time.perf_counter()
    output = fn(output_names, input_dict_B)
    end = time.perf_counter()
    latencies_sum += end - timed_inference_start
    inferences += 1
    if args.verbose == "2":
        ggprint(f"end - timed_inference_start = {end - timed_inference_start}")
        ggprint(f"num_completed_tasks = {inferences}")

def process_task_queue(task_queue, session, output_names):
    while True:
        try:
            input_dict = task_queue.get(block=False)
        except queue.Empty:
            if args.verbose == "2":
                ggprint(f"Queue ended. Thread={threading.current_thread().ident} elapsed={latencies_sum} nct={inferences}")
            break
        timed_inference(session.run, output_names, input_dict)
    
def build_threads_pool(task_queue, threads, session, output_names):
    #global inferences
    #global latencies_sum
    thread_pool = []
    for i in range(threads):
        t = threading.Thread(target = process_task_queue, args = (task_queue, session, output_names))
        thread_pool.append(t)    
    return thread_pool

def profile(args, num):
    global latencies_sum
    global inferences
    latencies_sum = 0
    inferences = 0 
    so = rt.SessionOptions()
    # Enable profiling by providing the profiling options
    # so.enable_profiling = True

    EP_List = []

    if args.execution_provider == "iGPU":
        providers = [("DmlExecutionProvider", {"device_id": 0})]
        args.threads = 1  # iGPU does not take advantage of async multi-threading
        ggprint("Check with Task Manager which GPU is used!")
        session = rt.InferenceSession(args.model, so, providers=providers)
    if args.execution_provider == "dGPU":
        # providers = ['DmlExecutionProvider', {'device_id':1}]
        providers = [("DmlExecutionProvider", {"device_id": 1})]
        ggprint("Check with Task Manager which GPU is used!")
        session = rt.InferenceSession(args.model, providers=providers)
    elif args.execution_provider == "ZenDNN":
        providers = ["ZendnnExecutionProvider"]
        session = rt.InferenceSession(args.model, so, providers=providers)
    elif args.execution_provider == "CPU":
        providers = ["CPUExecutionProvider"]
        # ref: https://onnxruntime.ai/docs/performance/tune-performance/threading.html
        so.intra_op_num_threads = args.intra_op_num_threads
        #so.execution_mode = rt.ExecutionMode.ORT_PARALLEL 
        #so.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
        #so.add_session_config_entry("session.intra_op.allow_spinning", "1")
        session = rt.InferenceSession(args.model, so, providers=providers)
    elif args.execution_provider == "ZenDPU":
        providers = ["ZendpuExecutionProvider"]
        session = rt.InferenceSession(args.model, so, providers=providers)
    elif args.execution_provider == "VitisAIEP":
        ggprint(f"config path = {args.config}")
        args.model = ggquantize(args)
        
        EP_List.append("VitisAIExecutionProvider")
        cache_dir = os.path.join(Path(__file__).parent.resolve(), "cache", os.path.basename(args.model))
        so.intra_op_num_threads = args.intra_op_num_threads
        
        provider_options = [
            {
                "config_file": args.config,
                "cacheDir": str(cache_dir),
                "cacheKey": "modelcachekey",
                "num_of_dpu_runners": args.instance_count,
            }
        ]
        session = rt.InferenceSession(
            args.model, so, providers=EP_List, provider_options=provider_options
        )

    target_delay = args.min_interval
    input_nodes = session.get_inputs()
    output_nodes = session.get_outputs()
    output_names = [n.name for n in output_nodes]

    # check model compatibility with batch size
    has_string = any(isinstance(element, str) for element in input_nodes[0].shape)
    good = True
    if args.batchsize > 1:
        if has_string == False:
            good = False
    assert (
        good
    ), "The model is not compatible: if batch size > 1 the model input should have a batchsize placeholder"

    # rounding number of images in the pool to a multiple of threads and batchsize
    num = (math.ceil(args.num / args.batchsize / args.threads)* args.batchsize *  args.threads)
    r = range(int(num / args.batchsize))
    ggprint(f"Executing {int(num/args.batchsize)} steps, with each step involving the inference of {args.batchsize} images")

    # fill the input data queue
    task_queue = queue.Queue()
    tensorshape = tuple(
        [
            [args.batchsize if isinstance(s, str) or s == None else s for s in n.shape]
            for n in input_nodes
        ][0]
    )

    # build the single queue of dictionaries from where all threads will get the input data 
    for i in r:
        input_dict = {
            n.name: np.random.randn(*tensorshape).astype(
                "float16" if "float16" in n.type else "float32"
            )
            for n in input_nodes
        }
        if args.no_inference=="0":
            task_queue.put(input_dict)

    # Start profiling
    if args.infinite:
        ggprint("Infinite loop: Press q to Stop.")
    
    profile_start = time.perf_counter()
    printonce = True
    while True: 
        if args.no_inference=="1":
            if printonce:
                printonce = False
                ggprint("Running without inference for power baseline")
        else:            
            # Create a new queue and duplicate the contents of the original queue
            task_queue_t = queue.Queue()
            task_queue_t.queue.extend(task_queue.queue)            

            thread_pool = build_threads_pool(
                    task_queue_t,
                    args.threads,
                    session,
                    output_names,
                )

            # start measuring time
            start = time.perf_counter()
            # wait a minimum time
            while time.perf_counter() - start < target_delay:
                pass

            # start the thread
            # each thread creates a list of sessions of the same dimension of the data queue, then runs them.
            for thread in thread_pool:
                thread.start()
            # wait for the thread to finish
            for thread in thread_pool:
                thread.join()
            nctlist.append(inferences)
            total_run_time = time.perf_counter() - start
            
            # list of time spent to process each set of images (args.num)
            finaltottime.append(total_run_time)
            
        if args.verbose == "1" or args.verbose == "2":
            ggprint(f'{len(nctlist)} - test time = {(time.perf_counter() - profile_start):.2f} / {args.timelimit:.2f} sec')

        # break infinite loop
        if not(args.infinite):
            break
        if keyboard.is_pressed("q"):
            ggprint("Key pressed. Stopping the loop.")
            break
        if (time.perf_counter() - profile_start) >= args.timelimit:
            ggprint(f"\nLimit time {args.timelimit} s reached")
            break
    
    if args.no_inference == "0":
        lat_results = 1000 * latencies_sum / inferences    
        thr_results = (inferences * args.batchsize) / sum(finaltottime)
    else:
        lat_results = 0    
        thr_results = 0

    
    if args.verbose == "1" or args.verbose == "2":
        print("\n")
        print(f'inferences in every batch  ={int(num / args.batchsize)}')
        print(f'cumulative time spent by all threads ={latencies_sum}')
        print(f'num_completed_tasks ={inferences}')
        print(f'average time spent on each task ms={1000 * latencies_sum / inferences}')
        # WARNING - this is valid for one instance only
        print(f'average latency= {lat_results:.2f} ms')
        print("\n")
        # measurements in Profile:
        # total_run_time is the time spent in completing one batch of images
        # finaltottime is the list of times spent in completing all batches of images
        print(f'number of processed batches of images: len(finaltottime) ={len(finaltottime)}')
        print(f'total inferences (should match with completed tasks) ={int(num / args.batchsize)*len(finaltottime)}')
        # this measurement method is WRONG!! it fails when there are multiple threads
        # print(f'average latency = {1000 * sum(finaltottime) / (len(finaltottime) * int(num / args.batchsize)):.2f} ms')
        print("\n")
        print(f'average throughput = {thr_results:.2f} fps')
        
    return [thr_results, lat_results, inferences]
        
    # session.end_profiling()

if __name__ == "__main__":
    device = detect_device()
    args, defaults = parse_args(device)

    # environment control
    check_env(release, args) 
    # imput parameters control
    check_args(args, defaults)
    # delete old measurement
    del_old_meas("report_benchmark.json")
    
    # EP driven setup
    xclbin_path = ""
    if args.execution_provider == "VitisAIEP":
        os.environ['NUM_OF_DPU_RUNNERS'] = str(args.instance_count)
        os.environ['XLNX_ONNX_EP_VERBOSE'] = "0"
        os.environ['XLNX_ENABLE_STAT_LOG']= "0"
        set_engine_shape(args.core)
        
        if args.renew == "1":
            cache_dir = os.path.join(Path(__file__).parent.resolve(), "cache", os.path.basename(args.model))
            cancelcache(cache_dir)

    if args.execution_provider == "ZenDNN":
        set_ZEN_env()
    
    # Warmup. Skipping in case of infinite loop
    if args.infinite:
        ggprint("Infinite Loop: skipping warmup")
    else:
        ggprint("Single run: warmup")
        thr_results, lat_results, inferences = profile(args, args.warmup)
        nctlist = []
        finaltottime=[]

    # performance benchmark
    thr_results, lat_results, inferences = profile(args, args.num)

    if args.no_inference == "0":
        measurement = meas_init(
            args, release, thr_results, lat_results, xclbin_path
        )

        print(Colors.YELLOW + f"\nprofiling:  {args.model} running on {args.execution_provider}")
        if (args.execution_provider == "VitisAIEP"): print(f"Instances (NPU runners)  = {args.instance_count}")
        print(f"Threads                  = {args.threads}")
        print(f"Batch size               = {args.batchsize}")
        print("-" * 80)
        print(f"Throughput:              = {thr_results:.2f} fps")
        print(f"Average latency:         = {lat_results:.2f} ms")
        print(f"Inferences:              = {inferences}")
        print("-" * 80)


        if args.execution_provider == "VitisAIEP":
            cache_dir = os.path.join(Path(__file__).parent.resolve(), "cache", os.path.basename(args.model))
            try:
                with open(os.path.join(cache_dir, r"modelcachekey\vitisai_ep_report.json"), "r") as json_file:
                    vitisaireport = json.load(json_file)
            except Exception as e:
                print(f"Error loading JSON file: {e}")
            
            print(f"Total nodes {vitisaireport['deviceStat'][0]['nodeNum']}")
            measurement["vitisai"]['all'] = vitisaireport['deviceStat'][0]['nodeNum']
            for i, device in enumerate(vitisaireport["deviceStat"][1:]):
                print(f"{device['nodeNum']} nodes assigned to {device['name']}")
                measurement["vitisai"][device['name']] = device['nodeNum']

        print(Colors.RESET)

        save_result_json(measurement, "report_performance.json")
        if args.log_csv=="1":
            appendcsv(measurement, args)
    else:
        print(Colors.YELLOW + "Test with no inference (for power baseline estimation) completed")
        print(Colors.RESET)
