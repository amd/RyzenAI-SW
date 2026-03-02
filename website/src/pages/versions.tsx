import React from 'react';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';

const versions = [
  {
    label: '1.7.0',
    path: '/',
    date: '2026-02',
    status: 'Current',
  },
  {
    label: '1.3',
    path: 'https://ryzenai.docs.amd.com/en/1.3/',
    date: '2025',
    status: 'Previous',
    external: true,
  },
  {
    label: '1.2',
    path: 'https://ryzenai.docs.amd.com/en/1.2/',
    date: '2024',
    status: 'Previous',
    external: true,
  },
];

export default function Versions(): React.ReactElement {
  return (
    <Layout title="Versions" description="Ryzen AI Software documentation versions">
      <main style={{maxWidth: 800, margin: '0 auto', padding: '2rem 1rem'}}>
        <h1>Ryzen AI Software Versions</h1>
        <table>
          <thead>
            <tr>
              <th>Version</th>
              <th>Released</th>
              <th>Status</th>
              <th>Documentation</th>
            </tr>
          </thead>
          <tbody>
            {versions.map((v) => (
              <tr key={v.label}>
                <td><strong>{v.label}</strong></td>
                <td>{v.date}</td>
                <td>
                  <span
                    style={{
                      padding: '2px 8px',
                      borderRadius: 4,
                      fontSize: '0.85rem',
                      fontWeight: 600,
                      ...(v.status === 'Current'
                        ? {background: 'rgba(46,160,67,0.15)', color: '#2ea043'}
                        : {background: 'rgba(130,130,130,0.15)', color: '#888'}),
                    }}
                  >
                    {v.status}
                  </span>
                </td>
                <td>
                  {v.external ? (
                    <a href={v.path} target="_blank" rel="noopener noreferrer">
                      View docs &rarr;
                    </a>
                  ) : (
                    <Link to={v.path}>View docs &rarr;</Link>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </main>
    </Layout>
  );
}
