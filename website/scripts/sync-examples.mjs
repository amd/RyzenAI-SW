#!/usr/bin/env node
/**
 * sync-examples.mjs
 *
 * Generates docs/.../index.mdx files from examples/.../README.md files.
 * The examples/README.md is the source of truth for tutorial prose.
 * Run: node website/scripts/sync-examples.mjs
 * Check: node website/scripts/sync-examples.mjs --check
 */
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(__dirname, '../..');
const EXAMPLES_DIR = path.join(REPO_ROOT, 'examples');
const DOCS_DIR = path.join(REPO_ROOT, 'docs');

const CHECK_MODE = process.argv.includes('--check');

const EXAMPLES_MAP = [
  // [examples path, docs output path, title override (optional)]
  ['audio/whisper/README.md', 'models-tutorials/audio/whisper/index.mdx', 'Running Whisper on Ryzen AI'],
  ['llms/VLM/README.md', 'models-tutorials/llms/vlm/index.mdx', 'Vision-Language Models (VLM)'],
  ['llms/oga_api/README.md', 'models-tutorials/llms/oga-api/index.mdx', 'OGA C++ API'],
  ['llms/oga_inference/README.md', 'models-tutorials/llms/oga-inference/index.mdx', 'OGA Inference (Python)'],
  ['llms/llm-sft-deploy/README.md', 'models-tutorials/llms/llm-sft-deploy/index.mdx', 'Fine-tune and Deploy LLMs'],
  ['llms/RAG-OGA/README.md', 'models-tutorials/llms/rag-oga/index.mdx', 'RAG with OGA'],
  ['vision/hello_world/README.md', 'models-tutorials/vision/hello-world/index.mdx', 'Hello World Tutorial'],
  ['vision/getting_started_resnet/README.md', 'models-tutorials/vision/getting-started-resnet/index.mdx', 'ResNet Getting Started'],
  ['vision/getting_started_resnet/int8/README.md', 'models-tutorials/vision/getting-started-resnet/int8/index.mdx', 'ResNet INT8 Quantization'],
  ['vision/getting_started_resnet/bf16/README.md', 'models-tutorials/vision/getting-started-resnet/bf16/index.mdx', 'ResNet BF16 Tutorial'],
  ['vision/getting_started_resnet/bf16/docs/README_C++.md', 'models-tutorials/vision/getting-started-resnet/bf16/cpp-deployment.mdx', 'ResNet BF16 C++ Deployment'],
  ['vision/image_classification/README.md', 'models-tutorials/vision/image-classification/index.mdx', 'Image Classification'],
  ['vision/iGPU/getting_started/README.md', 'models-tutorials/vision/igpu-getting-started/index.mdx', 'iGPU Getting Started'],
  ['vision/object_detection/yolov8m/README.md', 'models-tutorials/vision/object-detection/yolov8m/index.mdx', 'YOLOv8m Object Detection'],
  ['vision/object_detection/yolov8s-worldv2/README.md', 'models-tutorials/vision/object-detection/yolov8s-worldv2/index.mdx', 'YOLOv8s-WorldV2'],
  ['vision/torchvision_inference/README.md', 'models-tutorials/vision/torchvision-inference/index.mdx', 'Torchvision Inference'],
  ['vision/quark_quantization/README.md', 'models-tutorials/vision/quark-quantization/index.mdx', 'Quark Quantization'],
  ['vision/quark_quantization/docs/advanced_quant_readme.md', 'models-tutorials/vision/quark-quantization/advanced.mdx', 'Advanced Quantization'],
  ['vision/super-resolution/README.md', 'models-tutorials/vision/super-resolution/index.mdx', 'Super Resolution'],
  ['vision/cvml/README.md', 'models-tutorials/vision/cvml/index.mdx', 'CVML Library'],
  ['vision/cvml/samples/cvml-sample-face-detection/README.md', 'models-tutorials/vision/cvml/face-detection.mdx', 'CVML Face Detection'],
  ['vision/cvml/samples/cvml-sample-face-mesh/README.md', 'models-tutorials/vision/cvml/face-mesh.mdx', 'CVML Face Mesh'],
  ['multimodal/npu-gpu-pipeline/README.md', 'models-tutorials/multimodal/npu-gpu-pipeline/index.mdx', 'NPU-GPU Pipeline'],
  ['nlp/distilbert/README.md', 'models-tutorials/nlp/distilbert/index.mdx', 'DistilBERT Text Classification'],
  ['tools/npu-check/README.md', 'tools/npu-check/index.mdx', 'NPU Check Utilities'],
  ['tools/benchmarking/README.md', 'tools/benchmarking/index.mdx', 'NPU Benchmark Tool'],
];

function sanitizeForMdx(content) {
  // Remove Sphinx-era <table class="sphinxhide"> header blocks
  content = content.replace(/<table\s+class="sphinxhide"[\s\S]*?<\/table>/gi, '');

  // Convert standalone HTML tables to Markdown-safe format by wrapping in
  // a raw HTML block (blank line before/after so MDX treats it as HTML block)
  content = content.replace(
    /(<table[\s\S]*?<\/table>)/gi,
    (match) => `\n<div dangerouslySetInnerHTML={{__html: \`${match.replace(/`/g, '\\`')}\`}} />\n`
  );

  // Fix self-closing tags that MDX requires (img, br, hr)
  content = content.replace(/<img\s([^>]*[^/])>/gi, '<img $1 />');
  content = content.replace(/<br\s*>/gi, '<br />');
  content = content.replace(/<hr\s*>/gi, '<hr />');

  // Remove HTML comments (MDX doesn't support them, use {/* */} instead)
  content = content.replace(/<!--[\s\S]*?-->/g, '');

  // Escape angle brackets outside fenced code blocks that look like JSX
  const lines = content.split('\n');
  const result = [];
  let inFencedBlock = false;
  for (const line of lines) {
    if (/^```/.test(line.trim())) {
      inFencedBlock = !inFencedBlock;
      result.push(line);
      continue;
    }
    if (inFencedBlock) {
      result.push(line);
      continue;
    }
    // In indented code blocks (4+ spaces), escape angle brackets, braces, and JSX expressions
    if (/^(    |\t)/.test(line)) {
      result.push(line
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/\{/g, '&#123;')
        .replace(/\}/g, '&#125;'));
    } else {
      // In regular text, escape angle brackets that are NOT valid HTML/JSX tags
      // Keep: <div>, <img />, <br />, <a href=...>, <CIStatus>, etc.
      // Escape: <s>, <<SYS>>, <cvml-depth-estimation.h>, <filename>, etc.
      let processed = line;
      // Escape << and >> patterns (template syntax)
      processed = processed.replace(/<</g, '&lt;&lt;');
      processed = processed.replace(/>>/g, '&gt;&gt;');
      // Escape <word.ext> patterns (header includes, filenames)
      processed = processed.replace(/<([a-zA-Z][a-zA-Z0-9_-]*\.[a-zA-Z]+)>/g, '`<$1>`');
      // Escape standalone <s>, </s> (tokenizer tokens)
      processed = processed.replace(/<s>/g, '`<s>`');
      processed = processed.replace(/<\/s>/g, '`</s>`');
      result.push(processed);
    }
  }
  content = result.join('\n');

  // Fix internal .md links to point to the docs-site equivalents
  content = content.replace(/\(\.\/docs\/advanced_quant_readme\.md\)/g, '(/models-tutorials/vision/quark-quantization/advanced)');
  content = content.replace(/\(\.\/docs\/README_C\+\+\.md\)/g, '(/models-tutorials/vision/getting-started-resnet/bf16/cpp-deployment)');
  content = content.replace(/\(\.\/bf16\/README\.md\)/g, '(/models-tutorials/vision/getting-started-resnet/bf16)');
  content = content.replace(/\(\.\/int8\/README\.md\)/g, '(/models-tutorials/vision/getting-started-resnet/int8)');
  content = content.replace(/\(\.\.\/README\.md\)/g, '(./)');
  content = content.replace(/\(\.\.\/README\.mdx\)/g, '(./)');

  return content;
}

function generateMdx(readmeContent, title, examplesRelPath) {
  const ghPath = `examples/${examplesRelPath.replace('/README.md', '').replace('/docs/advanced_quant_readme.md', '').replace('/docs/README_C++.md', '')}`;

  // Detect if content uses Tabs/TabItem
  const usesTabs = readmeContent.includes('<Tabs') || readmeContent.includes('<TabItem');
  const tabsImport = usesTabs
    ? `import Tabs from '@theme/Tabs';\nimport TabItem from '@theme/TabItem';\n`
    : '';

  return `---
title: "${title}"
ci_validated: false
---

{/* AUTO-GENERATED from ${examplesRelPath} -- do not edit directly. */}
{/* Run "node website/scripts/sync-examples.mjs" after updating the source. */}

import CIStatus from '@site/src/components/CIStatus';
${tabsImport}
<CIStatus validated={false} />

:::info Source Code
Clone the repo and find this example at [\`${ghPath}/\`](https://github.com/amd/RyzenAI-SW/tree/main/${ghPath}).
:::

${readmeContent}
`;
}

let driftCount = 0;
let generatedCount = 0;

for (const [exampleRel, docsRel, title] of EXAMPLES_MAP) {
  const srcPath = path.join(EXAMPLES_DIR, exampleRel);
  const dstPath = path.join(DOCS_DIR, docsRel);

  if (!fs.existsSync(srcPath)) {
    console.warn(`  WARN: Source not found: ${exampleRel}`);
    continue;
  }

  let readmeContent = fs.readFileSync(srcPath, 'utf-8');
  readmeContent = sanitizeForMdx(readmeContent);
  const mdxContent = generateMdx(readmeContent, title, exampleRel);

  if (CHECK_MODE) {
    if (fs.existsSync(dstPath)) {
      const existing = fs.readFileSync(dstPath, 'utf-8');
      if (existing !== mdxContent) {
        console.error(`  DRIFT: ${docsRel} differs from ${exampleRel}`);
        driftCount++;
      }
    } else {
      console.error(`  MISSING: ${docsRel}`);
      driftCount++;
    }
  } else {
    fs.mkdirSync(path.dirname(dstPath), { recursive: true });
    fs.writeFileSync(dstPath, mdxContent, 'utf-8');
    generatedCount++;
    console.log(`  OK: ${docsRel}`);
  }
}

if (CHECK_MODE) {
  if (driftCount > 0) {
    console.error(`\n${driftCount} file(s) out of sync. Run "node website/scripts/sync-examples.mjs" to fix.`);
    process.exit(1);
  } else {
    console.log('\nAll generated docs are in sync with examples/.');
  }
} else {
  console.log(`\nGenerated ${generatedCount} .mdx files from examples/.`);
}
