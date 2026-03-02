# Extracted from C:\Users\bconsolvo\code\RyzenAI-SW\docs\website\node_modules\terser\README.md:346
$ rm -f /tmp/cache.json  # start fresh
$ terser file1.js file2.js --mangle-props --name-cache /tmp/cache.json -o part1.js
$ terser file3.js file4.js --mangle-props --name-cache /tmp/cache.json -o part2.js
