import { useCallback, useEffect, useMemo, useState } from 'react';
import { Panel, PanelGroup, PanelResizeHandle } from 'react-resizable-panels';
import InterviewerPane from './components/InterviewerPane';
import ProblemModal from './components/ProblemModal';
import EditorPane from './components/EditorPane';
import { problems, defaultProblemId } from './lib/mockData';
import { mockRun } from './lib/api';

const PANEL_LAYOUT_KEY = 'codesage-panel-layout';

const CONVERSATION_TIMELINE = [
  {
    id: 'chat-intro',
    type: 'push',
    item: {
      id: 'chat-intro',
      kind: 'chat',
      text: "Ok, let's start with your intro",
    },
  },
  {
    id: 'status-intro-complete',
    type: 'push',
    item: {
      id: 'status-intro-complete',
      kind: 'status',
      label: 'Listening complete',
      variant: 'listening',
    },
  },
  {
    id: 'chat-strengths',
    type: 'push',
    item: {
      id: 'chat-strengths',
      kind: 'chat',
      text: "Ok, now what's your strengths and weaknesses",
    },
  },
  {
    id: 'status-strengths-complete',
    type: 'push',
    item: {
      id: 'status-strengths-complete',
      kind: 'status',
      label: 'Listening complete',
      variant: 'listening',
    },
  },
  {
    id: 'chat-two-sum',
    type: 'push',
    item: {
      id: 'chat-two-sum',
      kind: 'chat',
      text: "Great! Let's start with a basic coding problem. Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target. I'm streaming test cases and boilerplate on the right, so code it up when you're ready.",
    },
    effect: { setProblemId: 'two-sum' },
  },
  {
    id: 'action-two-sum-modal',
    type: 'push',
    item: {
      id: 'action-two-sum-modal',
      kind: 'action',
      label: 'See full problem',
      action: 'see-problem',
      ariaLabel: 'Open full problem statement',
    },
  },
  {
    id: 'observer-initial',
    type: 'push',
    item: {
      id: 'observer-initial',
      kind: 'observer',
      status: 'observing',
    },
  },
  {
    id: 'observer-initial-update',
    type: 'update',
    targetId: 'observer-initial',
    updates: { status: 'observed' },
  },
  {
    id: 'chat-follow-up',
    type: 'push',
    item: {
      id: 'chat-follow-up',
      kind: 'chat',
      text: 'Follow-up: I notice that you managed to solve it in O(n^2) complexity. Can you come up with an algorithm that is less than O(n^2) time complexity?',
    },
  },
  {
    id: 'observer-secondary',
    type: 'push',
    item: {
      id: 'observer-secondary',
      kind: 'observer',
      status: 'observing',
    },
  },
  {
    id: 'chat-coach',
    type: 'push',
    item: {
      id: 'chat-coach',
      kind: 'chat',
      text: "Hmm, interesting thinking approach, try to think a bit more in this direction. Don't feel nervous.",
    },
  },
  {
    id: 'chat-duplicates',
    type: 'push',
    item: {
      id: 'chat-duplicates',
      kind: 'chat',
      text: "Great! Let's tackle a new prompt: Implement a function to find duplicates in an array. I'll provide fresh cases and boilerplate in the editor.",
    },
    effect: { setProblemId: 'find-duplicates' },
  },
  {
    id: 'action-duplicates-modal',
    type: 'push',
    item: {
      id: 'action-duplicates-modal',
      kind: 'action',
      label: 'See full problem',
      action: 'see-problem',
      ariaLabel: 'Open find duplicates problem statement',
    },
  },
  {
    id: 'observer-tertiary',
    type: 'push',
    item: {
      id: 'observer-tertiary',
      kind: 'observer',
      status: 'observing',
    },
  },
  {
    id: 'chat-parentheses-intro',
    type: 'push',
    item: {
      id: 'chat-parentheses-intro',
      kind: 'chat',
      text: "Once duplicates feel solid, we'll pivot to validating parentheses strings to test stack intuition.",
    },
  },
  {
    id: 'chat-parentheses',
    type: 'push',
    item: {
      id: 'chat-parentheses',
      kind: 'chat',
      text: 'Next up: determine if a parentheses string is valid. Tests and boilerplate are ready when you are.',
    },
    effect: { setProblemId: 'valid-parentheses' },
  },
  {
    id: 'action-parentheses-modal',
    type: 'push',
    item: {
      id: 'action-parentheses-modal',
      kind: 'action',
      label: 'See full problem',
      action: 'see-problem',
      ariaLabel: 'Open valid parentheses problem statement',
    },
  },
];

function usePersistedLayout() {
  const [layout, setLayout] = useState(() => {
    if (typeof window === 'undefined') return [38, 62];
    try {
      const saved = window.localStorage.getItem(PANEL_LAYOUT_KEY);
      if (!saved) return [38, 62];
      const parsed = JSON.parse(saved);
      if (!Array.isArray(parsed) || parsed.length !== 2) {
        return [38, 62];
      }
      return parsed;
    } catch (error) {
      console.error('Failed to load panel layout', error);
      return [38, 62];
    }
  });

  const handleLayoutChange = useCallback((sizes) => {
    setLayout(sizes);
    if (typeof window !== 'undefined') {
      window.localStorage.setItem(PANEL_LAYOUT_KEY, JSON.stringify(sizes));
    }
  }, []);

  return [layout, handleLayoutChange];
}

function buildInitialCode() {
  return problems.reduce((acc, problem) => {
    acc[problem.id] = {
      python: problem.boilerplate.python,
      cpp: problem.boilerplate.cpp,
    };
    return acc;
  }, {});
}

export default function App() {
  const [layout, handleLayoutChange] = usePersistedLayout();
  const [activeProblemId, setActiveProblemId] = useState(defaultProblemId);
  const [language, setLanguage] = useState('python');
  const [codeByProblem, setCodeByProblem] = useState(() => buildInitialCode());
  const [activeTestIndex, setActiveTestIndex] = useState(0);
  const [isProblemModalOpen, setProblemModalOpen] = useState(false);
  const [isRunning, setIsRunning] = useState(false);
  const [lastRun, setLastRun] = useState(null);
  const [snapshots, setSnapshots] = useState([]);
  const [ariaLiveMessage, setAriaLiveMessage] = useState('Ready');
  const [conversation, setConversation] = useState([]);

  const activeProblem = useMemo(
    () => problems.find((problem) => problem.id === activeProblemId) ?? problems[0],
    [activeProblemId]
  );

  const tests = activeProblem.tests;
  const code = codeByProblem[activeProblemId]?.[language] ?? '';

  const handleProblemChange = useCallback((nextProblemId) => {
    setActiveProblemId(nextProblemId);
    setActiveTestIndex(0);
    setLastRun(null);
    setAriaLiveMessage('Problem changed');
  }, []);

  const processConversationEvent = useCallback(
    (event) => {
      if (event.type === 'push' && event.item) {
        setConversation((prev) => [
          ...prev,
          {
            ...event.item,
            timestamp: Date.now(),
          },
        ]);
      }

      if (event.type === 'update' && event.targetId) {
        setConversation((prev) =>
          prev.map((item) =>
            item.id === event.targetId
              ? {
                  ...item,
                  ...(event.updates ?? {}),
                }
              : item
          )
        );
      }

      if (event.effect?.setProblemId) {
        handleProblemChange(event.effect.setProblemId);
      }
    },
    [handleProblemChange]
  );

  useEffect(() => {
    let index = 0;
    if (CONVERSATION_TIMELINE.length > 0) {
      processConversationEvent(CONVERSATION_TIMELINE[0]);
      index = 1;
    }

    const interval = setInterval(() => {
      const nextEvent = CONVERSATION_TIMELINE[index];
      if (!nextEvent) {
        clearInterval(interval);
        return;
      }
      processConversationEvent(nextEvent);
      index += 1;
    }, 4200);

    return () => clearInterval(interval);
  }, [processConversationEvent]);

  const handleLanguageChange = useCallback((nextLanguage) => {
    setLanguage(nextLanguage);
    setAriaLiveMessage(`Switched to ${nextLanguage === 'cpp' ? 'C++' : 'Python'}`);
  }, []);

  const handleCodeChange = useCallback(
    (nextCode) => {
      setCodeByProblem((prev) => ({
        ...prev,
        [activeProblemId]: {
          ...prev[activeProblemId],
          [language]: nextCode,
        },
      }));
    },
    [activeProblemId, language]
  );

  const handleRun = useCallback(async () => {
    if (isRunning) return;
    setIsRunning(true);
    setAriaLiveMessage('Running tests…');
    try {
      const response = await mockRun({
        problemId: activeProblemId,
        language,
        code,
        tests,
      });
      setLastRun(response);
      setAriaLiveMessage(`${response.passed}/${response.total} tests passed in ${response.execMs} ms`);
      const snapshot = {
        id: `snapshot-${Date.now()}`,
        timestamp: Date.now(),
        language: language === 'cpp' ? 'C++' : 'Python',
        case: tests[activeTestIndex]?.label ?? 'All cases',
      };
      setSnapshots((prev) => [...prev, snapshot].slice(-8));
    } catch (error) {
      console.error('Mock run failed', error);
      setAriaLiveMessage('Run failed. Please try again.');
    } finally {
      setIsRunning(false);
    }
  }, [activeProblemId, activeTestIndex, code, isRunning, language, tests]);

  useEffect(() => {
    function handleKeydown(event) {
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'enter') {
        event.preventDefault();
        handleRun();
      }
      if (event.altKey) {
        const index = Number.parseInt(event.key, 10) - 1;
        if (!Number.isNaN(index) && index >= 0 && index < tests.length) {
          event.preventDefault();
          setActiveTestIndex(index);
        }
      }
    }

    window.addEventListener('keydown', handleKeydown);
    return () => window.removeEventListener('keydown', handleKeydown);
  }, [handleRun, tests.length]);

  const handleInterviewerAction = useCallback((action) => {
    if (action === 'see-problem') {
      setProblemModalOpen(true);
    }
  }, []);

  return (
    <div className="app-shell">
      <PanelGroup
        direction="horizontal"
        defaultLayout={layout}
        onLayout={handleLayoutChange}
        className="main-panels"
      >
        <Panel defaultSize={layout[0]} minSize={28}>
          <div className="panel-content">
            <InterviewerPane
              conversation={conversation}
              onAction={handleInterviewerAction}
            />
          </div>
        </Panel>
        <PanelResizeHandle className="resize-handle" aria-label="Resize interviewer and editor panels" tabIndex={0} />
        <Panel defaultSize={layout[1]} minSize={35}>
          <div className="panel-content">
            <EditorPane
              problemTitle={activeProblem.title}
              language={language}
              code={code}
              onLanguageChange={handleLanguageChange}
              onCodeChange={handleCodeChange}
              onRun={handleRun}
              isRunning={isRunning}
              tests={tests}
              activeTestIndex={activeTestIndex}
              onSelectTest={setActiveTestIndex}
              lastRun={lastRun}
              ariaLiveMessage={ariaLiveMessage}
            />
          </div>
        </Panel>
      </PanelGroup>

      <ProblemModal
        isOpen={isProblemModalOpen}
        onClose={() => setProblemModalOpen(false)}
        problem={activeProblem}
      />
    </div>
  );
}
