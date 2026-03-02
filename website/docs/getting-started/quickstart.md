# Quickstart

> import CIStatus from '@site/src/components/CIStatus';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import ExpectedOutput from '@site/src/components/ExpectedOutput';
import TutorialDifficulty from '@site/src/components/TutorialDifficulty';
import CodeBlock from '@theme/CodeBlock';
import QuickstartLLMSource from '!!raw-loader!@site/code-samples/getting-started/quickstart_llm.py';

import CIStatus from '@site/src/components/CIStatus';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import ExpectedOutput from '@site/src/components/ExpectedOutput';
import TutorialDifficulty from '@site/src/components/TutorialDifficulty';
import CodeBlock from '@theme/CodeBlock';
import QuickstartLLMSource from '!!raw-loader!@site/code-samples/getting-started/quickstart_llm.py';

# Quickstart <TutorialDifficulty level="beginner" />

<CIStatus validated={false} />

Run your first LLM on the AMD NPU in under 5 minutes.

## Prerequisites

- [Installation](/getting-started/installation) completed
- Ryzen AI conda environment activated (default: `ryzen-ai-1.7.0`)

## Install Dependencies

<Tabs groupId="os">
<TabItem value="windows" label="Windows">

```powershell
pip install onnxruntime-genai huggingface_hub
```

</TabItem>
<TabItem value="linux" label="Linux">

```bash
pip install onnxruntime-genai huggingface_hub
```

</TabItem>
</Tabs>

## Download a Model

<Tabs groupId="os">
<TabItem value="windows" label="Windows">

```powershell
huggingface-cli download amd/Llama-3.2-1B-Instruct-onnx-ryzenai-hybrid --local-dir models/Llama-3.2-1B-Instruct-onnx-ryzenai-hybrid
```

</TabItem>
<TabItem value="linux" label="Linux">

```bash
huggingface-cli download amd/Llama-3.2-1B-Instruct-onnx-ryzenai-hybrid --local-dir models/Llama-3.2-1B-Instruct-onnx-ryzenai-hybrid
```

</TabItem>
</Tabs>

## Run Inference

<CodeBlock language="python" title="quickstart_llm.py">{QuickstartLLMSource}</CodeBlock>

The model will stream a response to the terminal. Output will vary depending on the model.

:::tip
The first run downloads and compiles the model, which takes longer. Subsequent runs start much faster.
:::

## Next Steps

- **[Chat Application](/models-tutorials/llms/server-interface)** -- Build a full chat interface
- **[Supported Models](/models-tutorials/llms/supported-models)** -- Browse all verified LLMs
- **[Applications](/applications)** -- Pre-built apps like Lemonade and GAIA
