import PropTypes from 'prop-types';
import { useMemo } from 'react';
import Editor from '@monaco-editor/react';

const LANGUAGE_OPTIONS = [
  { label: 'Python', value: 'python' },
  { label: 'C++', value: 'cpp' },
];

export default function EditorPane({
  problemTitle,
  language,
  code,
  onLanguageChange,
  onCodeChange,
  onRun,
  isRunning,
  tests,
  activeTestIndex,
  onSelectTest,
  lastRun,
  ariaLiveMessage,
}) {
  const activeTest = tests[activeTestIndex];

  const summaryChip = useMemo(() => {
    if (!lastRun) {
      return 'No runs yet';
    }
    return `${lastRun.passed}/${lastRun.total} passed · ${lastRun.execMs} ms`;
  }, [lastRun]);

  const rows = lastRun
    ? lastRun.results
    : tests.map((test) => ({
        caseName: test.label,
        expected: test.expected,
        actual: '—',
        pass: null,
      }));

  return (
    <section className="pane pane--right editor-pane" aria-label="Code editor">
      <div className="editor-header">
        <div>
          <h2 className="panel-title">{problemTitle}</h2>
          <p className="muted-text">Write code and run against the mock judge</p>
        </div>
        <div className="editor-controls">
          <label className="visually-hidden" htmlFor="language-select">
            Select language
          </label>
          <select
            id="language-select"
            className="language-select"
            value={language}
            onChange={(event) => onLanguageChange(event.target.value)}
          >
            {LANGUAGE_OPTIONS.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
          <button
            type="button"
            className="run-button"
            disabled={isRunning}
            onClick={onRun}
            aria-label="Run code"
          >
            {isRunning && <span className="spinner" aria-hidden="true" />} {isRunning ? 'Running…' : 'Run'}
          </button>
        </div>
      </div>

      <div className="editor-body">
        <div className="editor-monaco" role="application" aria-label="Monaco editor">
          <Editor
            value={code}
            language={language === 'cpp' ? 'cpp' : 'python'}
            theme="vs-dark"
            onChange={(nextValue) => onCodeChange(nextValue ?? '')}
            options={{
              minimap: { enabled: false },
              fontSize: 15,
              lineHeight: 22,
              lineNumbers: 'on',
              scrollBeyondLastLine: false,
              smoothScrolling: true,
            }}
          />
        </div>

        <nav className="test-tabs" aria-label="Test cases">
          {tests.map((test, index) => (
            <button
              key={test.id}
              type="button"
              className={['test-tab', index === activeTestIndex ? 'test-tab--active' : '']
                .filter(Boolean)
                .join(' ')}
              onClick={() => onSelectTest(index)}
              aria-pressed={index === activeTestIndex}
            >
              {test.label}
            </button>
          ))}
        </nav>

        <div className="result-panel" aria-live="polite">
          <div className="result-header">
            <div>
              <strong>Active Test</strong>
              <p className="muted-text">{activeTest?.inputDisplay}</p>
            </div>
            <span className="summary-chip">{summaryChip}</span>
          </div>
          <table className="results-table" aria-label="Test results">
            <thead>
              <tr>
                <th scope="col">Case</th>
                <th scope="col">Expected</th>
                <th scope="col">Actual</th>
                <th scope="col">Status</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((result) => {
                const statusLabel = result.pass === null ? 'Pending' : result.pass ? 'Pass' : 'Fail';
                const statusClass = result.pass === null ? '' : result.pass ? 'result-status--pass' : 'result-status--fail';
                return (
                  <tr key={result.caseName}>
                    <td>{result.caseName}</td>
                    <td className="muted-text">{result.expected}</td>
                    <td className="muted-text">{result.actual}</td>
                    <td>
                      <span
                        className={['result-status', statusClass].filter(Boolean).join(' ')}
                      >
                        {statusLabel}
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <div className="visually-hidden" aria-live="polite">
          {ariaLiveMessage}
        </div>
      </div>
    </section>
  );
}

EditorPane.propTypes = {
  problemTitle: PropTypes.string.isRequired,
  language: PropTypes.oneOf(['python', 'cpp']).isRequired,
  code: PropTypes.string.isRequired,
  onLanguageChange: PropTypes.func.isRequired,
  onCodeChange: PropTypes.func.isRequired,
  onRun: PropTypes.func.isRequired,
  isRunning: PropTypes.bool.isRequired,
  tests: PropTypes.arrayOf(
    PropTypes.shape({
      id: PropTypes.string.isRequired,
      label: PropTypes.string.isRequired,
      inputDisplay: PropTypes.string.isRequired,
      expected: PropTypes.string.isRequired,
    })
  ).isRequired,
  activeTestIndex: PropTypes.number.isRequired,
  onSelectTest: PropTypes.func.isRequired,
  lastRun: PropTypes.shape({
    execMs: PropTypes.number.isRequired,
    passed: PropTypes.number.isRequired,
    total: PropTypes.number.isRequired,
    results: PropTypes.arrayOf(
      PropTypes.shape({
        caseName: PropTypes.string.isRequired,
        expected: PropTypes.string.isRequired,
        actual: PropTypes.string.isRequired,
        pass: PropTypes.bool.isRequired,
      })
    ).isRequired,
  }),
  ariaLiveMessage: PropTypes.string.isRequired,
};

EditorPane.defaultProps = {
  lastRun: null,
};
