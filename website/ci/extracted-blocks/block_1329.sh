# Extracted from C:\Users\bconsolvo\code\RyzenAI-SW\docs\website\node_modules\source-map-support\README.md:250
$ npm install source-map-support coffeescript
$ node_modules/.bin/coffee --map --compile demo.coffee
$ node demo.js

demo.coffee:3
  bar = -> throw new Error 'this is a demo'
                     ^
Error: this is a demo
    at bar (demo.coffee:3:22)
    at foo (demo.coffee:4:3)
    at Object.<anonymous> (demo.coffee:5:1)
    at Object.<anonymous> (demo.coffee:1:1)
    at Module._compile (module.js:456:26)
    at Object.Module._extensions..js (module.js:474:10)
    at Module.load (module.js:356:32)
    at Function.Module._load (module.js:312:12)
    at Function.Module.runMain (module.js:497:10)
    at startup (node.js:119:16)
