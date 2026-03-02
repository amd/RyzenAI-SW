import React, {type ReactNode} from 'react';

interface ExpectedOutputProps {
  children: ReactNode;
  label?: string;
}

export default function ExpectedOutput({
  children,
  label = 'Expected Output',
}: ExpectedOutputProps): React.ReactElement {
  return (
    <div className="expected-output">
      <span className="expected-output__label">{label}</span>
      {children}
    </div>
  );
}
