import React from 'react';
import LiteYouTubeEmbed from 'react-lite-youtube-embed';
import 'react-lite-youtube-embed/dist/LiteYouTubeEmbed.css';

interface LiteYouTubeProps {
  id: string;
  title: string;
}

export default function LiteYouTube({id, title}: LiteYouTubeProps): React.ReactElement {
  return (
    <div className="lite-youtube-embed">
      <LiteYouTubeEmbed id={id} title={title} noCookie={true} />
    </div>
  );
}
