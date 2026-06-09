# Authoring conventions

Conventions every docs page follows. These are enforced/used by CI.

## 1. Owner header (required)

First line after frontmatter, a hidden comment that renders nothing:

```
{/* owner: <github-id> */}
```

Drives CODEOWNERS generation and CI failure routing. Default owner: `@dwithchenna`.

## 2. Code language tabs (Python default, C++ second)

When an example provides the same step in more than one language, show them in
tabs with **Python first (default)**, then **C++**. Use `<Tabs>`:

```mdx
<Tabs>
  <Tab title="Python">
    ```python
    from lemonade import ...
    ```
  </Tab>
  <Tab title="C++">
    ```cpp
    #include <onnxruntime_genai.h>
    ```
  </Tab>
</Tabs>
```

Order is always Python, then C++ (then any others after). Single-language
examples do not need tabs.

### Languages present in the RyzenAI-SW examples
From the live repo, examples use:
- **Python** - majority of examples (default tab)
- **C++** - OGA C++ API, CVML samples, `npu_check`, WinML ResNet (cpp)
- **Jupyter notebooks** (`.ipynb`) - hello_world, torchvision_inference
- **CMake** (`CMakeLists.txt`) and **Batchfile** (`.bat`) - build/setup snippets
- a small amount of **C**

So the two first-class code tabs are **Python** and **C++**. Notebooks are
linked to (not tabbed); CMake/Batch appear inline as build steps.

## 3. Executable code blocks (`test` tag)

Tag a block ` ```python test ` to have CI execute it on Strix / Strix Halo.
Untagged `python` blocks are syntax-checked only. Use `notest` to skip a block
entirely. See `extract_code_blocks.py`.

## 4. Page paths stay 2 levels deep

`folder/page.mdx` (e.g. `llms/overview.mdx`). The link checker does not resolve
3-level page paths. Use nested nav **groups** for deeper visual hierarchy.
