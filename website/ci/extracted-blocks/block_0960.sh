# Extracted from C:\Users\bconsolvo\code\RyzenAI-SW\docs\website\node_modules\jsesc\README.md:386
$ jsesc --json --pretty '{ "föo": "♥", "bår": "𝌆 baz" }'
{
  "f\u00F6o": "\u2665",
  "b\u00E5r": "\uD834\uDF06 baz"
}
