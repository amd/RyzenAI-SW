# Extracted from C:\Users\bconsolvo\code\RyzenAI-SW\docs\website\node_modules\is-npm\readme.md:19
$ node foo.js
# ┌──────────────────┬────────┐
# │     (index)      │ Values │
# ├──────────────────┼────────┤
# │ isPackageManager │ false  │
# │      isNpm       │ false  │
# │     isYarn       │ false  │
# │     isPnpm       │ false  │
# │      isBun       │ false  │
# └──────────────────┴────────┘
$ npm run foo
# ┌──────────────────┬────────┐
# │     (index)      │ Values │
# ├──────────────────┼────────┤
# │ isPackageManager │ true   │
# │      isNpm       │ true   │
# │     isYarn       │ false  │
# │     isPnpm       │ false  │
# │      isBun       │ false  │
# └──────────────────┴────────┘
$ yarn run foo
# ┌──────────────────┬────────┐
# │     (index)      │ Values │
# ├──────────────────┼────────┤
# │ isPackageManager │ true   │
# │      isNpm       │ false  │
# │     isYarn       │ true   │
# │     isPnpm       │ false  │
# │      isBun       │ false  │
# └──────────────────┴────────┘
$ pnpm run foo
# ┌──────────────────┬────────┐
# │     (index)      │ Values │
# ├──────────────────┼────────┤
# │ isPackageManager │ true   │
# │      isNpm       │ false  │
# │     isYarn       │ false  │
# │     isPnpm       │ true   │
# │      isBun       │ false  │
# └──────────────────┴────────┘
$ bun run foo
# ┌──────────────────┬────────┐
# │     (index)      │ Values │
# ├──────────────────┼────────┤
# │ isPackageManager │ true   │
# │      isNpm       │ false  │
# │     isYarn       │ false  │
# │     isPnpm       │ false  │
# │      isBun       │ true   │
# └──────────────────┴────────┘
