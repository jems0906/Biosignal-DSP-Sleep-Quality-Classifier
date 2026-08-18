import { useMemo } from 'react';

export function Spectrogram({ values }) {
  const cells = useMemo(() => values.flatMap((row, rowIndex) => row.map((value, columnIndex) => ({ id: `${rowIndex}-${columnIndex}`, value, rowIndex, columnIndex }))), [values]);
  const maximum = Math.max(...cells.map((cell) => cell.value), 0.00001);
  return <div className="spectrogram" role="img" aria-label="Signal spectrogram"><div className="spectrogram-grid">{cells.map((cell) => <span key={cell.id} style={{ opacity: Math.max(0.08, cell.value / maximum) }} />)}</div><div className="spectrogram-axis"><span>0 Hz</span><span>30 s</span></div></div>;
}
