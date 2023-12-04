
# ONNX Tool

## Get started

Extract and filter data from an ONNX model with ease.

### Usage

```sh
## Command line options
python display_onnx_info.py --h

usage: display_onnx_info.py [-h] -m MODEL [-f FILTER [FILTER ...]] [-o OUTFILE] [-g] [-u] [-t {csv,xlsx}] [-n NODE_NAME]
                                                                                                                        
Python utility to extract metadata from ONNX model                                                                      
                                                                                                                        
optional arguments:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        Model Path
  -f FILTER [FILTER ...], --filter FILTER [FILTER ...]
                        Filter information with operator names
  -o OUTFILE, --outfile OUTFILE
                        Path to output file to dump data
  -g, --graph-info      Graph information
  -u, --uniq-ops        Unique operators information
  -t {csv,xlsx}, --file-type {csv,xlsx}
                        Expected type of output file
  -n NODE_NAME, --node-name NODE_NAME
                        Filter information with node names

```

### Tabulate Unique Operators in the model with associated nodes

```sh
## Show unique operators (shows associated nodes, count and total count)
python display_onnx_info.py -m <path-to-onnx-model-file> -u
```

### Tabulate Graph info

```sh
## Show model info (nodes, i/o shapes, attributes, type, values)
python display_onnx_info.py -m <path-to-onnx-model-file>
```

### Filter info based on operators

```sh
## List of Operators to filter, use "-f" or "--filter"
###
python display_onnx_info.py -m <path-to-onnx-model-file> -f Op1 Op2 Op3
###
python display_onnx_info.py -m <path-to-onnx-model-file> -f Op1 Op2 Op3 -u
```
### Write to file

```sh
## Write data to file, use "-o" or "--outfile" option
python display_onnx_info.py -m <path-to-onnx-model-file> -o <output-file-path>
```