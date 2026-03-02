import React, {useState, useMemo, type ComponentProps, type ReactNode} from 'react';
import CodeBlock from '@theme-original/CodeBlock';

type Props = ComponentProps<typeof CodeBlock>;

const COLLAPSE_THRESHOLD = 10;
const PREVIEW_LINES = 4;

function extractCodeString(children: unknown): string | null {
  if (typeof children === 'string') return children;
  if (Array.isArray(children)) {
    const joined = children.filter(c => typeof c === 'string').join('');
    if (joined) return joined;
  }
  return null;
}

const floatingBtn: React.CSSProperties = {
  position: 'absolute',
  zIndex: 10,
  border: 'none',
  borderRadius: '6px',
  padding: '5px 12px',
  fontSize: '0.78rem',
  fontWeight: 600,
  cursor: 'pointer',
  color: '#e6edf3',
  background: 'rgba(30, 30, 30, 0.75)',
  backdropFilter: 'blur(4px)',
  transition: 'background 0.15s ease',
  whiteSpace: 'nowrap' as const,
  display: 'flex',
  alignItems: 'center',
  gap: '4px',
};

function CopyButton({codeString, style}: {codeString: string; style?: React.CSSProperties}) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(codeString);
    } catch {
      const ta = document.createElement('textarea');
      ta.value = codeString;
      document.body.appendChild(ta);
      ta.select();
      document.execCommand('copy');
      document.body.removeChild(ta);
    }
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <button
      type="button"
      onClick={handleCopy}
      style={{...floatingBtn, ...style}}
      aria-label="Copy code"
    >
      {copied ? '✓ Copied' : '📋 Copy'}
    </button>
  );
}

export default function CodeBlockWrapper(props: Props): ReactNode {
  const codeString = useMemo(() => extractCodeString(props.children), [props.children]);
  const lineCount = codeString ? codeString.split('\n').length : 0;
  const isCollapsible = lineCount > COLLAPSE_THRESHOLD;

  const [expanded, setExpanded] = useState(false);

  if (!codeString) {
    return <CodeBlock {...props} />;
  }

  if (!isCollapsible) {
    return (
      <div style={{position: 'relative', margin: '16px 0'}}>
        <CopyButton
          codeString={codeString}
          style={{top: '8px', right: '8px'}}
        />
        <CodeBlock {...props} />
      </div>
    );
  }

  const previewCode = codeString.split('\n').slice(0, PREVIEW_LINES).join('\n') + '\n...';

  return (
    <div style={{position: 'relative', margin: '16px 0'}}>
      {/* Copy button - always visible, top right */}
      <CopyButton
        codeString={codeString}
        style={{top: '8px', right: '130px'}}
      />

      {/* Toggle button - top right, next to copy */}
      <button
        type="button"
        onClick={() => setExpanded(prev => !prev)}
        aria-expanded={expanded}
        style={{...floatingBtn, top: '8px', right: '8px'}}
      >
        {expanded ? '⊘ Hide code' : '⊕ View code'}
      </button>

      {expanded ? (
        <CodeBlock {...props} />
      ) : (
        <div style={{position: 'relative', overflow: 'hidden', cursor: 'pointer'}} onClick={() => setExpanded(true)}>
          <div style={{pointerEvents: 'none'}}>
            <CodeBlock {...props} children={previewCode} />
          </div>
          <div
            style={{
              position: 'absolute',
              bottom: 0,
              left: 0,
              right: 0,
              height: '50px',
              background: 'linear-gradient(transparent, var(--ifm-background-color))',
              pointerEvents: 'none',
            }}
          />
        </div>
      )}
    </div>
  );
}
