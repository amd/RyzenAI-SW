import React, {useState, useCallback, useRef, useEffect} from 'react';
import Layout from '@theme-original/DocItem/Layout';
import type LayoutType from '@theme/DocItem/Layout';
import type {WrapperProps} from '@docusaurus/types';
import BrowserOnly from '@docusaurus/BrowserOnly';
import {useDoc} from '@docusaurus/plugin-content-docs/client';

type Props = WrapperProps<typeof LayoutType>;

function LLMToolbarInner(): JSX.Element | null {
  const {metadata} = useDoc();
  const [copied, setCopied] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    function handleClickOutside(e: MouseEvent) {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setMenuOpen(false);
      }
    }
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const getPageMarkdown = useCallback((): string => {
    const TurndownModule = require('turndown');
    const TurndownService = TurndownModule.default || TurndownModule;
    const turndown = new TurndownService({
      headingStyle: 'atx',
      codeBlockStyle: 'fenced',
      bulletListMarker: '-',
    });

    turndown.addRule('codeBlocks', {
      filter: (node: Element) => node.nodeName === 'PRE' && node.querySelector('code') !== null,
      replacement: (_content: string, node: Element) => {
        const code = (node as HTMLElement).querySelector('code');
        if (!code) return _content;
        const lang = Array.from(code.classList)
          .find((c: string) => c.startsWith('language-'))
          ?.replace('language-', '') || '';
        return `\n\`\`\`${lang}\n${code.textContent || ''}\n\`\`\`\n`;
      },
    });

    turndown.addRule('skipButtons', {
      filter: (node: Element) => {
        const el = node as HTMLElement;
        return el.classList?.contains('copyButton') || el.classList?.contains('llm-toolbar') || el.tagName === 'BUTTON';
      },
      replacement: () => '',
    });

    turndown.addRule('admonitions', {
      filter: (node: Element) => (node as HTMLElement).classList?.contains('admonition') ?? false,
      replacement: (content: string, node: Element) => {
        const title = (node as HTMLElement).querySelector('.admonitionHeading')?.textContent || 'Note';
        return `\n> **${title}**: ${content.trim()}\n`;
      },
    });

    const article = document.querySelector('article .markdown, article');
    if (!article) return '';
    const clone = article.cloneNode(true) as HTMLElement;
    clone.querySelectorAll('.llm-toolbar, .copyButton, button, .hash-link').forEach((el) => el.remove());
    return turndown.turndown(clone.innerHTML);
  }, []);

  const handleCopyForLLM = useCallback(async () => {
    const md = getPageMarkdown();
    const header = `# ${metadata.title}\n\nSource: ${window.location.href}\n\n---\n\n`;
    await navigator.clipboard.writeText(header + md);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }, [metadata.title, getPageMarkdown]);

  const handleViewMarkdown = useCallback(() => {
    const md = getPageMarkdown();
    const header = `# ${metadata.title}\n\nSource: ${window.location.href}\n\n---\n\n`;
    const blob = new Blob([header + md], {type: 'text/markdown;charset=utf-8'});
    window.open(URL.createObjectURL(blob), '_blank');
  }, [metadata.title, getPageMarkdown]);

  const pageTitle = encodeURIComponent(metadata.title);
  const pageUrl = typeof window !== 'undefined' ? encodeURIComponent(window.location.href) : '';

  const askOptions = [
    {
      label: 'Ask ChatGPT',
      icon: '🤖',
      getUrl: () => {
        const prompt = encodeURIComponent(
          `I'm reading the AMD Ryzen AI documentation page "${metadata.title}" at ${window.location.href}. Can you help me understand this page and answer questions about it?`
        );
        return `https://chatgpt.com/?q=${prompt}`;
      },
    },
    {
      label: 'Ask Claude',
      icon: '🧠',
      getUrl: () => {
        const prompt = encodeURIComponent(
          `I'm reading the AMD Ryzen AI documentation page "${metadata.title}" at ${window.location.href}. Can you help me understand this page and answer questions about it?`
        );
        return `https://claude.ai/new?q=${prompt}`;
      },
    },
  ];

  const btnStyle: React.CSSProperties = {
    background: 'none',
    border: '1px solid var(--ifm-color-emphasis-300)',
    borderRadius: '6px',
    padding: '5px 12px',
    cursor: 'pointer',
    color: 'var(--ifm-color-content-secondary)',
    fontSize: '13px',
    display: 'inline-flex',
    alignItems: 'center',
    gap: '4px',
    lineHeight: '1.4',
    transition: 'border-color 0.15s, color 0.15s',
  };

  return (
    <div
      className="llm-toolbar"
      style={{
        display: 'flex',
        gap: '8px',
        marginBottom: '16px',
        fontSize: '13px',
        flexWrap: 'wrap',
      }}
    >
      <button
        onClick={handleCopyForLLM}
        style={{
          ...btnStyle,
          ...(copied ? {borderColor: 'var(--ifm-color-success)', color: 'var(--ifm-color-success)'} : {}),
        }}
        title="Copy page content as clean markdown for use with an LLM"
      >
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <rect x="9" y="9" width="13" height="13" rx="2" ry="2"/>
          <path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/>
        </svg>
        {copied ? 'Copied!' : 'Copy for LLM'}
      </button>

      <button
        onClick={handleViewMarkdown}
        style={btnStyle}
        title="Open page content as plain markdown in a new tab"
      >
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/>
          <polyline points="14 2 14 8 20 8"/>
          <line x1="16" y1="13" x2="8" y2="13"/>
          <line x1="16" y1="17" x2="8" y2="17"/>
        </svg>
        View as Markdown
      </button>

      <div ref={menuRef} style={{position: 'relative'}}>
        <button
          onClick={() => setMenuOpen(!menuOpen)}
          style={btnStyle}
          title="Ask an AI assistant about this page"
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"/>
          </svg>
          Ask AI
          <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3">
            <polyline points="6 9 12 15 18 9"/>
          </svg>
        </button>

        {menuOpen && (
          <div
            style={{
              position: 'absolute',
              top: '100%',
              left: 0,
              marginTop: '4px',
              background: 'var(--ifm-background-surface-color)',
              border: '1px solid var(--ifm-color-emphasis-300)',
              borderRadius: '8px',
              boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
              zIndex: 100,
              minWidth: '180px',
              overflow: 'hidden',
            }}
          >
            {askOptions.map((opt) => (
              <a
                key={opt.label}
                href={opt.getUrl()}
                target="_blank"
                rel="noopener noreferrer"
                onClick={() => setMenuOpen(false)}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px',
                  padding: '10px 14px',
                  textDecoration: 'none',
                  color: 'var(--ifm-color-content)',
                  fontSize: '14px',
                  borderBottom: '1px solid var(--ifm-color-emphasis-200)',
                  transition: 'background 0.1s',
                }}
                onMouseEnter={(e) => {
                  (e.currentTarget as HTMLElement).style.background = 'var(--ifm-color-emphasis-100)';
                }}
                onMouseLeave={(e) => {
                  (e.currentTarget as HTMLElement).style.background = 'transparent';
                }}
              >
                <span>{opt.icon}</span>
                <span>{opt.label}</span>
                <svg
                  width="12" height="12" viewBox="0 0 24 24"
                  fill="none" stroke="currentColor" strokeWidth="2"
                  style={{marginLeft: 'auto', opacity: 0.5}}
                >
                  <path d="M18 13v6a2 2 0 01-2 2H5a2 2 0 01-2-2V8a2 2 0 012-2h6"/>
                  <polyline points="15 3 21 3 21 9"/>
                  <line x1="10" y1="14" x2="21" y2="3"/>
                </svg>
              </a>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function LLMToolbar(): JSX.Element {
  return (
    <BrowserOnly fallback={<div />}>
      {() => <LLMToolbarInner />}
    </BrowserOnly>
  );
}

export default function LayoutWrapper(props: Props): JSX.Element {
  return (
    <>
      <LLMToolbar />
      <Layout {...props} />
    </>
  );
}
