import subprocess  
import time  
import signal  
import os  

# Start the process  
process = subprocess.Popen(  
    ["python", "pipeline.py", "-i", "test/test_img2img.mp4", "--npu", "--provider_config", "vaip_config.json", "--igpu"]  
)  

# Wait for 2 minutes  
try:  
    time.sleep(240)  
except KeyboardInterrupt:  
    pass  

# Terminate the process  
process.terminate()  

# Wait for the process to terminate  
try:  
    process.wait(timeout=10)  
except subprocess.TimeoutExpired:  
    process.kill()  

