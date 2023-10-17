import os
import json 
import copy
import pathlib as pl
yolovx_config = {
      "model_filter_id":0,
      "thread_num": 4,
      "onnx_model_path": r"\resource\nano-YOLOX_int.onnx",
      "video_file_path": r"\resource\detection.avi",
      "onnx_config": {
        "onnx_x":1,
        "onnx_y":1,
        "onnx_disable_spinning":False,
        "onnx_disable_spinning_between_run":False,
        "intra_op_thread_affinities":""
      }
    }
resnet50_config = {
      "model_filter_id":2,
      "thread_num": 4,
      "onnx_model_path": r"\resource\resnet50_pt.onnx",
      "video_file_path": r"\resource\detection.avi",
      "onnx_config": {
        "onnx_x":1,
        "onnx_y":1,
        "onnx_disable_spinning":False,
        "onnx_disable_spinning_between_run":False,
        "intra_op_thread_affinities":""
      }
    }
mobile_net_v2_config ={
      "model_filter_id":5,
      "thread_num": 4,
      "onnx_model_path": r"\resource\mobilenetv2_1.4_int.onnx",
      "confidence_threshold": 0.3,
      "video_file_path": r"\resource\detection.avi",
      "onnx_config": {
        "onnx_x":1,
        "onnx_y":1,
        "onnx_disable_spinning":False,
        "onnx_disable_spinning_between_run":False,
        "intra_op_thread_affinities":""
      }
    }
retinaface_config ={
      "model_filter_id":6,
      "thread_num": 4,
      "onnx_model_path": r"\resource\RetinaFace_int.onnx",
      "confidence_threshold": 0.3,
      "video_file_path": r"\resource\face.avi",
      "onnx_config": {
        "onnx_x":1,
        "onnx_y":1,
        "onnx_disable_spinning":False,
        "onnx_disable_spinning_between_run":False,
        "intra_op_thread_affinities":""
      }
    }
segmentation_config ={
      "model_filter_id":7,
      "thread_num": 4,
      "onnx_model_path": r"\resource\pointpainting-nus-FPN_int.onnx",
      "confidence_threshold": 0.3,
      "video_file_path": r"\resource\seg_512_288.avi",
      "onnx_config": {
        "onnx_x":1,
        "onnx_y":1,
        "onnx_disable_spinning":False,
        "onnx_disable_spinning_between_run":False,
        "intra_op_thread_affinities":""
      }
    }
# yolov5_config ={
#       "model_filter_id":8,
#       "thread_num": 4,
#       "onnx_model_path": r"\resource\yolov5s6.onnx",
#       "confidence_threshold": 0.3,
#       "video_file_path": r"\resource\detection.avi",
#       "onnx_config": {
#         "onnx_x":1,
#         "onnx_y":1,
#         "onnx_disable_spinning":False,
#         "onnx_disable_spinning_between_run":False,
#         "intra_op_thread_affinities":""
#       }
#     }

model_config={"resnet50":resnet50_config,
              "mobile_net_v2":mobile_net_v2_config,
              "retinaface":retinaface_config,
              "yolovx":yolovx_config,
              "segmentation":segmentation_config}


def set_cwd(config,raw_cwd):
    config["onnx_model_path"] = raw_cwd + config["onnx_model_path"]
    config["video_file_path"] = raw_cwd + config["video_file_path"]
def ger_raw_cwd():
    cwd = os.getcwd()
    cwd.replace("\\","\\\\")
    return r"%s"%(cwd) 
config_framework={
    "split_matrix_size": 1,
    "screen_height": 640,
    "screen_width": 1024,
    "models":{}
    }

bat_file =["set XLNX_VART_FIRMWARE=%cd%\\..\\1x4.xclbin\n",
           "set PATH=%cd%\\..\\bin;%cd%\\..\\python;%cd%\\..;%PATH%\n",
           "set DEBUG_ONNX_TASK=0\n",
           "set DEBUG_DEMO=0\n",
           "set NUM_OF_DPU_RUNNERS=4\n",
           "set XLNX_ENABLE_GRAPH_ENGINE_PAD=1\n",
           "set XLNX_ENABLE_GRAPH_ENGINE_DEPAD=1\n",
           "%cd%\\..\\bin\ipu_multi_models.exe %cd%\\config\\"]
def log_fil_generate(file_path):
  print(file_path,"generated!!")
def generate_for_one(model,raw_cwd,config_name):
    one_model_config = copy.deepcopy(config_framework)
    one_model_config["models"][model]=copy.deepcopy(model_config[model])
    one_model_config["split_matrix_size"]=1 
    file_path = "config\\"+config_name+".json"
    log_fil_generate(file_path)
    with open(file_path,"w",encoding = "utf-8") as f:
        f.write(json.dumps(one_model_config,indent=4))
    file_path ="run_"+config_name+".bat"
    log_fil_generate(file_path)
    with open(file_path,"w",encoding = "utf-8") as f:
        for line in bat_file[:-1]:
            f.write(line)
        f.write(bat_file[-1]+config_name+".json\n")  
def generate_for_four(models,raw_cwd,config_name,is_camera):
    four_model_config = copy.deepcopy(config_framework)
    four_model_config["split_matrix_size"]=2
    file_path ="config\\"+config_name+".json"
    log_fil_generate(file_path)
    for is_camera_on,model_name in zip(is_camera,models):
        model_config_index = model_name.split('.')[0]
        four_model_config["models"][model_name]=copy.deepcopy(model_config[model_config_index])
        if is_camera_on >=0:
            four_model_config["models"][model_name]["video_file_path"]=str(is_camera_on)
    with open(file_path,"w",encoding = "utf-8") as f:
            f.write(json.dumps(four_model_config,indent=4))
    file_path="run_"+config_name+".bat"
    log_fil_generate(file_path)
    with open(file_path,"w",encoding = "utf-8") as f:
        for line in bat_file[:-1]:
            f.write(line)
        f.write(bat_file[-1]+config_name+".json\n")
 
def test_pair(model_name,raw_cwd):
   all_models = model_config.keys()
   for ohter_model in all_models:
       models = [model_name+".0",ohter_model+".1"]
       generate_for_four(models,raw_cwd,models[0]+"X"+models[1],[-1,-1])

def check_file_exist(path):
    return os.path.exists(path)
def try_generate_video(name,pwd):
    video_path = "resource\\"+name+".avi"
    image_path = "resource\\"+name
    if check_file_exist(video_path):
        return 
    cmd = "python3 .\\resource\\to_video.py %s %s"%(pwd+"\\"+image_path,pwd+"\\"+video_path)
    print("exe -> ",cmd)
    os.system(cmd)
if __name__=="__main__":
    raw_cwd = ger_raw_cwd()
    # Define the directory name
    directory_name = "config"

# Check if the directory exists using os.path.exists
    if not os.path.exists(directory_name):
    # If the directory does not exist, create it using os.mkdir
      os.mkdir(directory_name)
      print(f"Directory '{directory_name}' created successfully.")
    else:
      print(f"Directory '{directory_name}' already exists.")
    # try_generate_video("detection",raw_cwd)
    # try_generate_video("face",raw_cwd)
    print("cur_dir: ",raw_cwd)
    for c in model_config.values():
        set_cwd(c,raw_cwd)
    ### one model
    all_models = model_config.keys()
    for model in all_models:
       generate_for_one(model,raw_cwd,model)
    ### four model
    models = ["yolovx","retinaface","segmentation","mobile_net_v2"]
    generate_for_four(models,raw_cwd,"modelx4",[-1,-1,-1,-1])
    # generate_for_four(models,raw_cwd,"modelx4_with_camera_on",[-1,0,-1,-1])
