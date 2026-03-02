import React from 'react';

type FeatureLevel = 'stable' | 'beta' | 'alpha' | 'deprecated';

interface FeatureStateProps {
  level: FeatureLevel;
}

const labels: Record<FeatureLevel, string> = {
  stable: 'Stable',
  beta: 'Beta',
  alpha: 'Alpha',
  deprecated: 'Deprecated',
};

export default function FeatureState({level}: FeatureStateProps): React.ReactElement {
  return <span className={`feature-badge feature-badge--${level}`}>{labels[level]}</span>;
}
