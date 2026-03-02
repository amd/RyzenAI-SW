# -*- coding: utf-8 -*-
# Extracted from C:\Users\bconsolvo\code\RyzenAI-SW\docs\models-tutorials\vision\cnn-examples.mdx:167
parser = argparse.ArgumentParser()
parser.add_argument('--ep', type=str, default ='cpu',choices = ['cpu','npu'], help='EP backend selection')
opt = parser.parse_args()

providers = ['CPUExecutionProvider']
provider_options = [{}]

if opt.ep == 'npu':
   providers = ['VitisAIExecutionProvider']
   cache_dir = Path(__file__).parent.resolve()
   provider_options = [{
              'cacheDir': str(cache_dir),
              'cacheKey': 'modelcachekey',
              }]

session = ort.InferenceSession(model.SerializeToString(), providers=providers,
                               provider_options=provider_options)
