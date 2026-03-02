import React, {useState, useRef, type ReactNode} from 'react';

interface ExpandableCodeProps {
  children: ReactNode;
  title?: string;
  defaultExpanded?: boolean;
  previewLines?: number;
}

export default function ExpandableCode({
  children,
  title,
  defaultExpanded = false,
  previewLines = 10,
}: ExpandableCodeProps): React.ReactElement {
  const [isExpanded, setIsExpanded] = useState(defaultExpanded);
  const containerRef = useRef<HTMLDivElement>(null);

  const previewMaxHeight = `${previewLines * 1.5}em`;

  const handleCopy = () => {
    const el = containerRef.current;
    if (!el) return;
    const codeEl = el.querySelector('code');
    const text = codeEl?.textContent || el.textContent || '';
    navigator.clipboard.writeText(text);
  };

  return (
    <div className="expandable-code-block" ref={containerRef}>
      <div className="expandable-code-block__toolbar">
        <span className="expandable-code-block__title">{title || 'Code'}</span>
        <div className="expandable-code-block__actions">
          <button
            type="button"
            className="expandable-code-block__toggle-btn"
            onClick={() => setIsExpanded(!isExpanded)}
            aria-expanded={isExpanded}
          >
            {isExpanded ? '▲ Hide code' : '▼ Show code'}
          </button>
        </div>
      </div>
      <div
        className={`expandable-code-block__content ${isExpanded ? 'expandable-code-block__content--expanded' : ''}`}
        style={isExpanded ? undefined : {maxHeight: previewMaxHeight}}
      >
        {children}
        {!isExpanded && <div className="expandable-code-block__fade" />}
      </div>
      <div className="expandable-code-block__footer">
        <button
          type="button"
          className="expandable-code-block__copy-btn"
          onClick={handleCopy}
        >
          &#x1F4CB; Copy
        </button>
      </div>
    </div>
  );
}
