import React from 'react';

type DifficultyLevel = 'beginner' | 'intermediate' | 'advanced';

interface TutorialDifficultyProps {
  level: DifficultyLevel;
}

const labels: Record<DifficultyLevel, string> = {
  beginner: 'Beginner',
  intermediate: 'Intermediate',
  advanced: 'Advanced',
};

export default function TutorialDifficulty({level}: TutorialDifficultyProps): React.ReactElement {
  return <span className={`difficulty-badge difficulty-badge--${level}`}>{labels[level]}</span>;
}
