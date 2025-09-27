import { problems } from './mockData';

const PLACEHOLDER_PATTERNS = [/TODO/i, /\bpass\b/, /return\s*\{\}/];

function hasPlaceholder(code = '') {
  return PLACEHOLDER_PATTERNS.some((pattern) => pattern.test(code));
}

function simulateTwoSum({ nums = [], target = 0 }) {
  const map = new Map();
  for (let i = 0; i < nums.length; i += 1) {
    const diff = target - nums[i];
    if (map.has(diff)) {
      return `[${map.get(diff)}, ${i}]`;
    }
    map.set(nums[i], i);
  }
  return 'No solution';
}

function simulateFindDuplicates({ nums = [] }) {
  const counts = new Map();
  const seen = new Set();
  nums.forEach((value) => {
    const nextCount = (counts.get(value) || 0) + 1;
    counts.set(value, nextCount);
    if (nextCount > 1) {
      seen.add(value);
    }
  });
  const duplicates = Array.from(seen).sort((a, b) => a - b);
  return `[${duplicates.join(', ')}]`;
}

function simulateValidParentheses({ s = '' }) {
  const stack = [];
  const pairs = { ')': '(', ']': '[', '}': '{' };
  for (const char of s) {
    if (pairs[char]) {
      if (!stack.length || stack.pop() !== pairs[char]) {
        return 'false';
      }
    } else {
      stack.push(char);
    }
  }
  return stack.length === 0 ? 'true' : 'false';
}

function evaluateCase(problemId, testInput) {
  switch (problemId) {
    case 'two-sum':
      return simulateTwoSum(testInput);
    case 'find-duplicates':
      return simulateFindDuplicates(testInput);
    case 'valid-parentheses':
      return simulateValidParentheses(testInput);
    default:
      return '';
  }
}

function getProblem(problemId) {
  return problems.find((problem) => problem.id === problemId);
}

export async function mockRun({ problemId, language, code, tests }) {
  const problem = getProblem(problemId);
  if (!problem) {
    throw new Error('Unknown problem id');
  }

  const delay = 800 + Math.floor(Math.random() * 400);
  const execMs = 60 + Math.floor(Math.random() * 60);
  const placeholder = hasPlaceholder(code);

  const results = tests.map((test) => {
    const actual = placeholder ? 'Not implemented' : evaluateCase(problemId, test.input);
    const pass = !placeholder && actual === test.expected;
    return {
      caseName: test.label,
      expected: test.expected,
      actual,
      pass,
    };
  });

  const passed = results.filter((item) => item.pass).length;
  const total = results.length;
  const stdout = results.map((item) => `${item.caseName}: ${item.actual}`).join('\n');
  const stderr = placeholder ? 'Solution still contains placeholders.' : '';

  return new Promise((resolve) => {
    setTimeout(() => {
      resolve({
        stdout,
        stderr,
        execMs,
        language,
        passed,
        total,
        results,
      });
    }, delay);
  });
}
