# ready for public

release = 19

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

    if args.no_inference == "0":
        timed_inference_start = time.perf_counter()
        output = fn(output_names, input_dict_B)
        end = time.perf_counter()
        latencies_sum += end - timed_inference_start
        inferences += 1
        #if args.verbose == "2":
        #    ggprint(f"end - timed_inference_start = {end - timed_inference_start}")
        #    ggprint(f"num_completed_tasks = {inferences}")
    
    else:
        # do almost nothing for the latency time, and keep the same code (redundant)
        # doing this to maintain the experiment duration also in case of no inference (power baseline calculation)
        timed_inference_start = time.perf_counter()
        #while time.perf_counter() - timed_inference_start < args.min_interval:
        #    pass
        
        time.sleep(args.min_interval)

        end = time.perf_counter()
        latencies_sum += end - timed_inference_start
        inferences += 1
        #if args.verbose == "2":
        #    ggprint(f"end - timed_inference_start = {end - timed_inference_start}")
        #    ggprint(f"num_completed_tasks = {inferences}")


def process_task_queue(task_queue, session, output_names):
    while True:
        try:
            input_dict = task_queue.get(block=False)
        except queue.Empty:
            #if args.verbose == "2":
            #    ggprint(f"Queue ended. Thread={threading.current_thread().ident} elapsed={latencies_sum} nct={inferences}")
            break
        timed_inference(session.run, output_names, input_dict)
    
def build_threads_pool(task_queue, threads, session, output_names):
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
    # Enable ONNX profiling by providing the profiling options
    #if args.profile=="1":
    #    so.enable_profiling = True
    #else:
    #    so.enable_profiling = False

    EP_List = []

    if args.execution_provider == "iGPU":
        providers = [("DmlExecutionProvider", {"device_id": 0})]
        args.threads = 1  # iGPU does not take advantage of async multi-threading
        ggprint("Check with Task Manager which GPU is used!")
        session = rt.InferenceSession(args.model, so, providers=providers)
    elif args.execution_provider == "dGPU":
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

        # model preprocessing
        args.model = ggquantize(args)

        EP_List.append("VitisAIExecutionProvider")
        
        # this is only valid for ryzenai
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
        
                #"ai_analyzer_visualization": True if args.profile=="1" else False,
                #"ai_analyzer_profiling": True if args.profile=="1" else False
        session = rt.InferenceSession(
            args.model, 
            so, 
            providers=EP_List, 
            provider_options=provider_options
        ) 

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
    num_bs = range(int(num / args.batchsize))
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
    for i in num_bs:
        input_dict = {
            n.name: np.random.randn(*tensorshape).astype(
                "float16" if "float16" in n.type else "float32"
            )
            for n in input_nodes
        }
        task_queue.put(input_dict)

    # Start profiling
    if args.infinite:
        ggprint("Infinite loop: Press q to Stop.")
    
    profile_start = time.perf_counter()

    if args.no_inference=="1":
        ggprint("Running without inference for power baseline")
        ggprint("Remember to manually set the latency with --min_interval option [seconds]")
    
    while True: 
        # else:            
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
        # # wait a minimum time
        # while time.perf_counter() - start < args.min_interval:
        #     pass
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
    
    lat_results = 1000 * latencies_sum / inferences    
    thr_results = (inferences * args.batchsize) / sum(finaltottime)
    
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
        

    #if args.profile=="1":
    #    profile_file = session.end_profiling()
    #    ggprint(f"ONNX Profiler report saved to {profile_file}")

    return [thr_results, lat_results, inferences]

if __name__ == "__main__":
    device = detect_device()
    args, defaults = parse_args(device)

    # environment control
    check_env(release, args) 
    
    # imput parameters control
    check_args(args, defaults)
    # delete old measurement
    del_file("report_benchmark.json")
    
    # EP driven setup
    xclbin_path = ""
    TABLEW=51
    if args.execution_provider == "VitisAIEP":
        os.environ['NUM_OF_DPU_RUNNERS'] = str(args.instance_count)
        os.environ['XLNX_ONNX_EP_VERBOSE'] = "0"
        os.environ['XLNX_ENABLE_STAT_LOG']= "0"
        os.environ["XLNX_ONNX_EP_REPORT_FILE"] = "vitisai_ep_report.json"

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

    if args.no_inference == "1":
        ggprint("TESTING WITH NO INFERENCE FOR POWER BASELINE")

    measurement = meas_init(
        args, release, thr_results, lat_results, xclbin_path
    )
    
    test_summary =  [["model", measurement['model']['name']],
                    ["instances", args.instance_count if args.execution_provider == "VitisAIEP" else ""],
                    ["threads", args.threads],
                    ["core",  "VAIP_4x4" if os.path.basename(args.config) == "vitisai_config.json" else args.core]]       
    performance  =  [["throughput", f"{thr_results:.2f} [fps]"],
                    ["latency", f"{lat_results:.2f} [ms]"],
                    ["inferences", inferences]]
    
    comp_efficency = measurement['model']['Forward_FLOPs']*thr_results/(tops_peak(device)*1e12)
    efficency    = [["model", measurement['model']['name']],
                    ["model operations", f"{measurement['model']['Forward_FLOPs']/1e6:.2f} [MFLOPs]"],
                    ["device", device],
                    ["theoretical compute peak", f"{tops_peak(device)} [TOPs]"],
                    ["compute efficency", f"{comp_efficency*100:.2f}%"]]
    measurement['model']['device'] = device
    measurement['model']['core'] = args.core
    measurement['model']['comp_peak'] = tops_peak(device)
    measurement['model']['comp_efficency'] = comp_efficency*100
    
    print(Colors.YELLOW)
    tableprint("TEST SUMMARY", test_summary, TABLEW)
    tableprint("PERFORMANCE", performance, TABLEW)
    tableprint("EFFICENCY", efficency, TABLEW)
    
    
    if args.execution_provider == "VitisAIEP":
        cache_dir = os.path.join(Path(__file__).parent.resolve(), "cache", os.path.basename(args.model))       
        try:
            with open(os.path.join(cache_dir, r"modelcachekey\vitisai_ep_report.json"), "r") as json_file:
                vitisaireport = json.load(json_file)
        except Exception as e:
            print(f"Error loading JSON file: {e}")
        
        measurement["vitisai"]['all'] = vitisaireport['deviceStat'][0]['nodeNum']
        vitisai_ep_report = [["total nodes", f"{vitisaireport['deviceStat'][0]['nodeNum']}"]]
        for i, device in enumerate(vitisaireport["deviceStat"][1:]):
            vitisai_ep_report.append([f"{device['name']}", f"{device['nodeNum']}"])
            measurement["vitisai"][device['name']] = device['nodeNum']
        
        tableprint("NODES DISTRIBUTION", vitisai_ep_report, TABLEW)
    

    print(Colors.RESET)
    save_result_json(measurement, "report_performance.json")
    if args.log_csv=="1":
        appendcsv(measurement, args)
    if args.profile=="1":
        ggprint("The AI Analyzer results can be reviewed running aianalyzer . --port=8087")

