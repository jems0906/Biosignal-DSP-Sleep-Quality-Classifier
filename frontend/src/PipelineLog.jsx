const stages = [
  ['01', 'Ingestion', 'EDF / CSV signal accepted', 'complete'],
  ['02', 'Preprocessing', '0.5-30 Hz bandpass + 60 Hz notch', 'complete'],
  ['03', 'Feature extraction', 'Welch PSD and four bandpowers', 'complete'],
  ['04', 'Quality gate', 'Amplitude and flatline checks', 'complete'],
  ['05', 'Inference', 'RandomForest stage probabilities', 'complete'],
];

export function PipelineLog() {
  return <div className="pipeline-page"><div className="model-intro"><span className="section-kicker">RUN HISTORY / PIPELINE LOG</span><h2>Every signal leaves a trace.</h2><p>A transparent record of the processing stages applied to the current epoch.</p></div><div className="pipeline-list">{stages.map(([number, title, detail, status]) => <div className="pipeline-step" key={number}><span className="pipeline-number">{number}</span><div><strong>{title}</strong><p>{detail}</p></div><span className="pipeline-status">{status}</span></div>)}</div></div>;
}
