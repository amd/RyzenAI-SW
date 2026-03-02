# Extracted from C:\Users\bconsolvo\code\RyzenAI-SW\docs\website\node_modules\html-webpack-plugin\node_modules\commander\Readme.md:321
$ extra --help
Usage: help [options]

Options:
  -t, --timeout <delay>  timeout in seconds (default: one minute)
  -d, --drink <size>     drink cup size (choices: "small", "medium", "large")
  -p, --port <number>    port number (env: PORT)
  -h, --help             display help for command

$ extra --drink huge
error: option '-d, --drink <size>' argument 'huge' is invalid. Allowed choices are small, medium, large.

$ PORT=80 extra 
Options:  { timeout: 60, port: '80' }
