import React from 'react';

interface CIStatusProps {
  validated: boolean;
  lastRun?: string;
}

export default function CIStatus({validated, lastRun}: CIStatusProps): React.ReactElement {
  const style: React.CSSProperties = {
    display: 'inline-flex',
    alignItems: 'center',
    gap: '6px',
    padding: '4px 10px',
    borderRadius: '6px',
    fontSize: '0.75rem',
    fontWeight: 600,
    letterSpacing: '0.02em',
    marginBottom: '16px',
    border: '1px solid',
    ...(validated
      ? {
          background: 'rgba(46, 160, 67, 0.1)',
          color: '#2ea043',
          borderColor: 'rgba(46, 160, 67, 0.3)',
        }
      : {
          background: 'rgba(210, 153, 34, 0.1)',
          color: '#d29922',
          borderColor: 'rgba(210, 153, 34, 0.3)',
        }),
  };

  const icon = validated ? '✓' : '○';
  const label = validated ? 'CI Validated' : 'Pending CI Validation';
  const date = lastRun ? ` · ${lastRun}` : '';

  return (
    <div style={style}>
      <span>{icon}</span>
      <span>{label}{date}</span>
    </div>
  );
}
