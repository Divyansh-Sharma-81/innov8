import PropTypes from 'prop-types';

function renderItem(item, onAction) {
  switch (item.kind) {
    case 'chat':
      return (
        <div className="interviewer-bubble interviewer-bubble--chat">
          {item.text}
        </div>
      );
    case 'status': {
      const variantClass = item.variant ? `interviewer-chip--${item.variant}` : '';
      return (
        <span className={['interviewer-chip', variantClass].filter(Boolean).join(' ')}>
          {item.label}
        </span>
      );
    }
    case 'observer': {
      const statusClass = `interviewer-chip--observer-${item.status}`;
      const label = item.status === 'observed' ? 'Observed' : 'Observing...';
      return (
        <span className={[
          'interviewer-chip',
          'interviewer-chip--observer',
          statusClass,
        ].join(' ')}>
          {label}
        </span>
      );
    }
    case 'action':
      return (
        <button
          type="button"
          className="interviewer-action"
          onClick={() => onAction(item.action, item)}
          aria-label={item.ariaLabel ?? item.label}
        >
          <span>{item.label}</span>
          <span className="interviewer-action__icon" aria-hidden="true">
            -&gt;
          </span>
        </button>
      );
    default:
      return null;
  }
}

export default function InterviewerPane({ conversation, onAction }) {
  return (
    <section className="pane pane--left" aria-label="AI interviewer conversation">
      <header className="interviewer-header">
        <h1 className="interviewer-title">Interviewer</h1>
      </header>
      <div className="pane-scroll conversation-log" role="log" aria-live="polite">
        <ul className="conversation-list">
          {conversation.map((item) => (
            <li key={item.id} className="conversation-item">
              {renderItem(item, onAction)}
            </li>
          ))}
        </ul>
      </div>
    </section>
  );
}

const chatShape = PropTypes.shape({
  id: PropTypes.string.isRequired,
  kind: PropTypes.oneOf(['chat']).isRequired,
  text: PropTypes.string.isRequired,
});

const statusShape = PropTypes.shape({
  id: PropTypes.string.isRequired,
  kind: PropTypes.oneOf(['status']).isRequired,
  label: PropTypes.string.isRequired,
  variant: PropTypes.string,
});

const observerShape = PropTypes.shape({
  id: PropTypes.string.isRequired,
  kind: PropTypes.oneOf(['observer']).isRequired,
  status: PropTypes.oneOf(['observing', 'observed']).isRequired,
});

const actionShape = PropTypes.shape({
  id: PropTypes.string.isRequired,
  kind: PropTypes.oneOf(['action']).isRequired,
  label: PropTypes.string.isRequired,
  action: PropTypes.string.isRequired,
  ariaLabel: PropTypes.string,
});

InterviewerPane.propTypes = {
  conversation: PropTypes.arrayOf(
    PropTypes.oneOfType([chatShape, statusShape, observerShape, actionShape])
  ).isRequired,
  onAction: PropTypes.func.isRequired,
};
