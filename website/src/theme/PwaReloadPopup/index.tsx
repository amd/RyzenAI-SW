import React from 'react';

interface Props {
  onReload: () => void;
}

export default function PwaReloadPopup({onReload}: Props): JSX.Element {
  return (
    <div
      style={{
        position: 'fixed',
        bottom: '1rem',
        right: '1rem',
        padding: '0.75rem 1.25rem',
        borderRadius: '8px',
        backgroundColor: 'var(--ifm-color-primary)',
        color: '#fff',
        boxShadow: '0 4px 12px rgba(0,0,0,0.3)',
        zIndex: 1000,
        display: 'flex',
        alignItems: 'center',
        gap: '0.75rem',
        fontSize: '0.875rem',
      }}>
      <span>New version available</span>
      <button
        type="button"
        onClick={onReload}
        style={{
          background: '#fff',
          color: 'var(--ifm-color-primary)',
          border: 'none',
          borderRadius: '4px',
          padding: '0.25rem 0.75rem',
          cursor: 'pointer',
          fontWeight: 600,
        }}>
        Reload
      </button>
    </div>
  );
}
