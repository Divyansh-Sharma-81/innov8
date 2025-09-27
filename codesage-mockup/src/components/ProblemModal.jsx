import { useEffect, useState } from 'react';
import PropTypes from 'prop-types';

export default function ProblemModal({ isOpen, onClose, problem }) {
  const [activeExampleIndex, setActiveExampleIndex] = useState(0);

  useEffect(() => {
    if (isOpen) {
      setActiveExampleIndex(0);
    }
  }, [isOpen, problem.id]);

  useEffect(() => {
    if (!isOpen) return undefined;

    function handleKeyDown(event) {
      if (event.key === 'Escape') {
        onClose();
      }
    }

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose]);

  if (!isOpen) {
    return null;
  }

  const activeExample = problem.examples[activeExampleIndex] ?? problem.examples[0];

  return (
    <div className="modal-overlay" role="presentation">
      <div
        className="modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="problem-modal-title"
      >
        <header className="modal-header">
          <h2 id="problem-modal-title" className="modal-title">
            {problem.title}
          </h2>
          <button type="button" className="modal-close" onClick={onClose} aria-label="Close problem modal">
            ✕
          </button>
        </header>
        <div className="modal-body">
          <section className="modal-section">
            <h4>Description</h4>
            <p className="muted-text">{problem.description}</p>
          </section>

          <section className="modal-section">
            <h4>Constraints</h4>
            <ul className="modal-constraints">
              {problem.constraints.map((line) => (
                <li key={line}>{line}</li>
              ))}
            </ul>
          </section>

          <section className="modal-section" aria-labelledby="examples-heading">
            <h4 id="examples-heading">Examples</h4>
            <div className="modal-tabs" role="tablist">
              {problem.examples.map((example, index) => (
                <button
                  key={example.name}
                  type="button"
                  role="tab"
                  className={[
                    'modal-tab',
                    index === activeExampleIndex ? 'modal-tab--active' : '',
                  ]
                    .filter(Boolean)
                    .join(' ')}
                  aria-selected={index === activeExampleIndex}
                  onClick={() => setActiveExampleIndex(index)}
                >
                  {example.name}
                </button>
              ))}
            </div>
            <article className="modal-example" role="tabpanel">
              <div>
                <strong>Input:</strong> <span className="muted-text">{activeExample.input}</span>
              </div>
              <div>
                <strong>Output:</strong> <span className="muted-text">{activeExample.output}</span>
              </div>
              <div>
                <strong>Explanation:</strong> <span className="muted-text">{activeExample.explanation}</span>
              </div>
            </article>
          </section>
        </div>
      </div>
    </div>
  );
}

ProblemModal.propTypes = {
  isOpen: PropTypes.bool.isRequired,
  onClose: PropTypes.func.isRequired,
  problem: PropTypes.shape({
    id: PropTypes.string.isRequired,
    title: PropTypes.string.isRequired,
    description: PropTypes.string.isRequired,
    constraints: PropTypes.arrayOf(PropTypes.string).isRequired,
    examples: PropTypes.arrayOf(
      PropTypes.shape({
        name: PropTypes.string.isRequired,
        input: PropTypes.string.isRequired,
        output: PropTypes.string.isRequired,
        explanation: PropTypes.string.isRequired,
      })
    ).isRequired,
  }).isRequired,
};
