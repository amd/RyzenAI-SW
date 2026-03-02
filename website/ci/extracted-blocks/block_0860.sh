# Extracted from C:\Users\bconsolvo\code\RyzenAI-SW\docs\website\node_modules\he\README.md:306
$ he --encode 'föo ♥ bår 𝌆 baz'
f&#xF6;o &#x2665; b&#xE5;r &#x1D306; baz

$ he --encode --use-named-refs 'föo ♥ bår 𝌆 baz'
f&ouml;o &hearts; b&aring;r &#x1D306; baz

$ he --decode 'f&ouml;o &hearts; b&aring;r &#x1D306; baz'
föo ♥ bår 𝌆 baz
