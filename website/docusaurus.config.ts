import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

const config: Config = {
  title: 'Ryzen AI',
  tagline: 'Build AI applications on AMD Ryzen AI PCs',
  favicon: 'img/favicon.ico',
  url: 'https://ryzenai.docs.amd.com',
  baseUrl: '/',

  organizationName: 'amd',
  projectName: 'RyzenAI-SW',

  onBrokenLinks: 'throw',
  trailingSlash: false,

  headTags: [
    {
      tagName: 'script',
      attributes: {type: 'application/ld+json'},
      innerHTML: JSON.stringify({
        '@context': 'https://schema.org',
        '@type': 'TechArticle',
        name: 'Ryzen AI Documentation',
        description: 'Build AI applications on AMD Ryzen AI PCs with NPU and GPU acceleration.',
        url: 'https://ryzenai.docs.amd.com',
        publisher: {
          '@type': 'Organization',
          name: 'Advanced Micro Devices, Inc.',
          url: 'https://www.amd.com',
        },
      }),
    },
  ],

  future: {
    v4: true,
    // experimental_faster requires Rspack which doesn't support raw-loader.
    // Re-enable once code-from-source imports are migrated to Rspack-compatible approach.
    experimental_faster: false,
  },

  i18n: {
    defaultLocale: 'en',
    locales: ['en', 'zh-Hans', 'fr', 'es', 'pt-BR'],
    localeConfigs: {
      en: {label: 'English', direction: 'ltr'},
      'zh-Hans': {label: '简体中文', direction: 'ltr'},
      fr: {label: 'Français', direction: 'ltr'},
      es: {label: 'Español', direction: 'ltr'},
      'pt-BR': {label: 'Português (Brasil)', direction: 'ltr'},
    },
  },

  markdown: {
    mermaid: true,
    hooks: {
      onBrokenMarkdownLinks: 'throw',
    },
  },

  presets: [
    [
      'classic',
      {
        docs: {
          path: '../docs',
          routeBasePath: '/',
          sidebarPath: './sidebars.ts',
          editUrl: 'https://github.com/amd/RyzenAI-SW/edit/main/docs/',
          showLastUpdateTime: true,
          showLastUpdateAuthor: true,
          exclude: [
            '**/templates/**',
            '**/node_modules/**',
            'CODEOWNERS',
            'README.md',
            'README.mdx',
          ],
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.scss',
        },
      } satisfies Preset.Options,
    ],
  ],

  plugins: [
    'docusaurus-plugin-sass',
    [
      '@docusaurus/plugin-client-redirects',
      {
        redirects: [
          {from: '/en/latest', to: '/'},
          {from: '/en/latest/index', to: '/'},
          {from: '/en/latest/inst', to: '/getting-started/installation'},
          {from: '/en/latest/quickstart', to: '/getting-started/quickstart'},
          {from: '/en/latest/examples', to: '/models-tutorials'},
          {from: '/en/latest/modelrun', to: '/develop/model-deployment'},
          {from: '/en/latest/app_development', to: '/develop/app-development'},
          {from: '/en/latest/model_quantization', to: '/develop/model-quantization'},
          {from: '/en/latest/oga_model_prepare', to: '/develop/onnx-model-preparation'},
          {from: '/en/latest/oga_op_prepare', to: '/develop/operator-preparation'},
          {from: '/en/latest/hybrid_oga', to: '/models-tutorials/llms/hybrid-inference'},
          {from: '/en/latest/llm/overview', to: '/models-tutorials/llms/overview'},
          {from: '/en/latest/llm/high_level_python', to: '/models-tutorials/llms/python-api'},
          {from: '/en/latest/llm/server_interface', to: '/models-tutorials/llms/server-interface'},
          {from: '/en/latest/llm_linux', to: '/models-tutorials/llms/linux-setup'},
          {from: '/en/latest/getstartex', to: '/models-tutorials/vision/cnn-examples'},
          {from: '/en/latest/sd_demo', to: '/models-tutorials/vision/stable-diffusion'},
          {from: '/en/latest/whisper_cpp', to: '/models-tutorials/audio/whisper'},
          {from: '/en/latest/ryzen_ai_libraries', to: '/develop/cvml-library'},
          {from: '/en/latest/gpu/ryzenai_gpu', to: '/develop/rocm-client-gpu'},
          {from: '/en/latest/xrt_smi', to: '/tools/npu-management'},
          {from: '/en/latest/ai_analyzer', to: '/tools/ai-analyzer'},
          {from: '/en/latest/relnotes', to: '/reference/changelog'},
          {from: '/en/latest/model_list', to: '/reference/model-list'},
          {from: '/en/latest/ops_support', to: '/reference/supported-operators'},
          {from: '/en/latest/licenses', to: '/reference/licenses'},
        ],
      },
    ],
    [
      'docusaurus-plugin-llms',
      {
        generateMarkdownFiles: true,
        docsDir: '..',
        ignoreFiles: [
          '**/node_modules/**',
          '**/website/**',
          '**/templates/**',
          '**/README.mdx',
          '**/Readme.mdx',
          '**/readme.mdx',
          '**/README_C++.mdx',
          '**/advanced_quant_readme.mdx',
        ],
      },
    ],
    '@docusaurus/plugin-ideal-image',
    [
      '@docusaurus/plugin-pwa',
      {
        offlineModeActivationStrategies: ['appInstalled', 'standalone', 'queryString'],
        pwaHead: [
          {tagName: 'link', rel: 'icon', href: '/img/amd-logo.svg'},
          {tagName: 'link', rel: 'manifest', href: '/manifest.json'},
          {tagName: 'meta', name: 'theme-color', content: '#ed1c24'},
        ],
      },
    ],
  ],

  themes: ['@docusaurus/theme-mermaid'],

  themeConfig: {
    image: 'img/ryzen-ai-social-card.png',

    colorMode: {
      defaultMode: 'dark',
      disableSwitch: false,
      respectPrefersColorScheme: true,
    },

    navbar: {
      title: 'Ryzen AI',
      logo: {
        alt: 'AMD Ryzen AI',
        src: 'img/amd-logo.svg',
      },
      items: [
        {
          type: 'dropdown',
          label: 'v1.7.0',
          position: 'right',
          items: [
            {
              label: 'v1.7.0 (current)',
              to: '/',
            },
            {
              label: 'v1.3',
              href: 'https://ryzenai.docs.amd.com/en/1.3/',
            },
            {
              label: 'v1.2',
              href: 'https://ryzenai.docs.amd.com/en/1.2/',
            },
            {
              label: 'All versions',
              to: '/versions',
            },
          ],
        },
        {
          type: 'localeDropdown',
          position: 'right',
        },
        {
          href: 'https://github.com/amd/RyzenAI-SW',
          position: 'right',
          className: 'header-github-link',
          'aria-label': 'GitHub repository',
        },
        {
          href: 'https://discord.gg/amd-dev',
          position: 'right',
          className: 'header-discord-link',
          'aria-label': 'AMD Developer Discord',
        },
      ],
    },

    footer: {
      style: 'dark',
      links: [
        {
          title: 'Documentation',
          items: [
            {label: 'Getting Started', to: '/getting-started/installation'},
            {label: 'Applications', to: '/applications'},
            {label: 'Models & Tutorials', to: '/models-tutorials'},
          ],
        },
        {
          title: 'Community',
          items: [
            {
              label: 'Discord',
              href: 'https://discord.gg/amd-dev',
            },
            {
              label: 'GitHub Discussions',
              href: 'https://github.com/amd/RyzenAI-SW/discussions',
            },
            {
              label: 'GitHub Issues',
              href: 'https://github.com/amd/RyzenAI-SW/issues',
            },
          ],
        },
        {
          title: 'AMD',
          items: [
            {
              label: 'AMD Developer',
              href: 'https://developer.amd.com',
            },
            {
              label: 'ROCm Docs',
              href: 'https://rocm.docs.amd.com',
            },
            {
              label: 'Hugging Face',
              href: 'https://huggingface.co/amd',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Advanced Micro Devices, Inc.`,
    },

    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ['powershell', 'bash', 'json', 'yaml', 'cmake', 'cpp', 'python'],
    },

    docs: {
      sidebar: {
        hideable: true,
        autoCollapseCategories: true,
      },
    },

    tableOfContents: {
      minHeadingLevel: 2,
      maxHeadingLevel: 4,
    },

    mermaid: {
      theme: {light: 'neutral', dark: 'dark'},
    },

    // Algolia DocSearch - uncomment when API key is available
    // algolia: {
    //   appId: 'YOUR_APP_ID',
    //   apiKey: 'YOUR_SEARCH_API_KEY',
    //   indexName: 'ryzenai',
    //   contextualSearch: true,
    // },
  } satisfies Preset.ThemeConfig,
};

export default config;
