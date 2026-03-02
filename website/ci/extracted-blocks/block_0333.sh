# Extracted from C:\Users\bconsolvo\code\RyzenAI-SW\docs\website\node_modules\@babel\parser\CHANGELOG.md:723
> 1 | let { x, ...y, ...z } = { x: 1, y: 2, z: 3};
    |              ^
# Previous behavior:
# x = 1
# y = { y: 2, z: 3 }
# z = { y: 2, z: 3 }
