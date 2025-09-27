export const problems = [
  {
    id: 'two-sum',
    title: 'Two Sum',
    description:
      'Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target. You may assume that each input would have exactly one solution, and you may not use the same element twice.',
    constraints: [
      '2 ≤ nums.length ≤ 10^4',
      '-10^9 ≤ nums[i] ≤ 10^9',
      '-10^9 ≤ target ≤ 10^9',
      'Exactly one valid answer exists',
    ],
    examples: [
      {
        name: 'Example 1',
        input: 'nums = [2,7,11,15], target = 9',
        output: '[0, 1]',
        explanation: 'nums[0] + nums[1] = 2 + 7 = 9',
      },
      {
        name: 'Example 2',
        input: 'nums = [3,2,4], target = 6',
        output: '[1, 2]',
        explanation: 'nums[1] + nums[2] = 2 + 4 = 6',
      },
      {
        name: 'Example 3',
        input: 'nums = [3,3], target = 6',
        output: '[0, 1]',
        explanation: 'nums[0] + nums[1] = 3 + 3 = 6',
      },
    ],
    tests: [
      {
        id: 'ts-1',
        label: 'Case 1',
        input: { nums: [2, 7, 11, 15], target: 9 },
        inputDisplay: 'nums = [2,7,11,15]\ntarget = 9',
        expected: '[0, 1]',
      },
      {
        id: 'ts-2',
        label: 'Case 2',
        input: { nums: [3, 2, 4], target: 6 },
        inputDisplay: 'nums = [3,2,4]\ntarget = 6',
        expected: '[1, 2]',
      },
      {
        id: 'ts-3',
        label: 'Case 3',
        input: { nums: [3, 3], target: 6 },
        inputDisplay: 'nums = [3,3]\ntarget = 6',
        expected: '[0, 1]',
      },
    ],
    boilerplate: {
      python: `from typing import List\n\n\ndef two_sum(nums: List[int], target: int) -> List[int]:\n    \"\"\"Return the indices of the two numbers that add up to target.\"\"\"\n    # TODO: replace with your implementation\n    pass\n\n\nif __name__ == \"__main__\":\n    import sys, json\n\n    payload = json.loads(sys.stdin.read() or '{}')\n    nums = payload.get('nums', [])\n    target = payload.get('target', 0)\n    print(two_sum(nums, target))\n`,
      cpp: `#include <bits/stdc++.h>\nusing namespace std;\n\nvector<int> twoSum(vector<int>& nums, int target) {\n    // TODO: replace with your implementation\n    return {};\n}\n\nint main() {\n    ios::sync_with_stdio(false);\n    cin.tie(nullptr);\n\n    // Input format (JSON): { \"nums\": [...], \"target\": int }\n    // Parsing is omitted in this mock runner.\n    return 0;\n}\n`,
    },
  },
  {
    id: 'find-duplicates',
    title: 'Find Duplicates',
    description:
      'Given an integer array nums, return all the elements that appear more than once. Return the result in ascending order.',
    constraints: [
      '1 ≤ nums.length ≤ 10^5',
      '1 ≤ nums[i] ≤ 10^5',
      'Output should be sorted in ascending order',
    ],
    examples: [
      {
        name: 'Example 1',
        input: 'nums = [4,3,2,7,8,2,3,1]',
        output: '[2, 3]',
        explanation: 'Both 2 and 3 appear twice in the array.',
      },
      {
        name: 'Example 2',
        input: 'nums = [1,1,2]',
        output: '[1]',
        explanation: 'Only 1 appears more than once.',
      },
      {
        name: 'Example 3',
        input: 'nums = [1]',
        output: '[]',
        explanation: 'No duplicates present.',
      },
    ],
    tests: [
      {
        id: 'fd-1',
        label: 'Case 1',
        input: { nums: [4, 3, 2, 7, 8, 2, 3, 1] },
        inputDisplay: 'nums = [4,3,2,7,8,2,3,1]',
        expected: '[2, 3]',
      },
      {
        id: 'fd-2',
        label: 'Case 2',
        input: { nums: [1, 1, 2] },
        inputDisplay: 'nums = [1,1,2]',
        expected: '[1]',
      },
      {
        id: 'fd-3',
        label: 'Case 3',
        input: { nums: [1] },
        inputDisplay: 'nums = [1]',
        expected: '[]',
      },
    ],
    boilerplate: {
      python: `from typing import List\n\n\ndef find_duplicates(nums: List[int]) -> List[int]:\n    \"\"\"Return elements that appear more than once, sorted ascending.\"\"\"\n    # TODO: replace with your implementation\n    pass\n\n\nif __name__ == \"__main__\":\n    import sys, json\n\n    payload = json.loads(sys.stdin.read() or '{}')\n    nums = payload.get('nums', [])\n    print(find_duplicates(nums))\n`,
      cpp: `#include <bits/stdc++.h>\nusing namespace std;\n\nvector<int> findDuplicates(const vector<int>& nums) {\n    // TODO: replace with your implementation\n    return {};\n}\n\nint main() {\n    ios::sync_with_stdio(false);\n    cin.tie(nullptr);\n\n    // Input format (JSON): { \"nums\": [...] }\n    return 0;\n}\n`,
    },
  },
  {
    id: 'valid-parentheses',
    title: 'Valid Parentheses',
    description:
      'Given a string s containing just the characters "(){}[]", determine if the input string is valid. An input string is valid if open brackets are closed by the same type of brackets, and open brackets are closed in the correct order.',
    constraints: [
      '1 ≤ s.length ≤ 10^4',
      's consists of parentheses characters only',
    ],
    examples: [
      {
        name: 'Example 1',
        input: 's = "()"',
        output: 'true',
        explanation: 'The string closes its only open bracket.',
      },
      {
        name: 'Example 2',
        input: 's = "()[]{}"',
        output: 'true',
        explanation: 'Every bracket closes in order.',
      },
      {
        name: 'Example 3',
        input: 's = "(]"',
        output: 'false',
        explanation: 'There is a mismatched closing bracket.',
      },
    ],
    tests: [
      {
        id: 'vp-1',
        label: 'Case 1',
        input: { s: '()' },
        inputDisplay: 's = "()"',
        expected: 'true',
      },
      {
        id: 'vp-2',
        label: 'Case 2',
        input: { s: '()[]{}' },
        inputDisplay: 's = "()[]{}"',
        expected: 'true',
      },
      {
        id: 'vp-3',
        label: 'Case 3',
        input: { s: '(]' },
        inputDisplay: 's = "(]"',
        expected: 'false',
      },
    ],
    boilerplate: {
      python: `def is_valid(s: str) -> bool:\n    \"\"\"Return whether the string has valid parentheses.\"\"\"\n    # TODO: replace with your implementation\n    pass\n\n\nif __name__ == \"__main__\":\n    import sys, json\n\n    payload = json.loads(sys.stdin.read() or '{}')\n    s = payload.get('s', '')\n    print(str(is_valid(s)).lower())\n`,
      cpp: `#include <bits/stdc++.h>\nusing namespace std;\n\nbool isValid(const string& s) {\n    // TODO: replace with your implementation\n    return false;\n}\n\nint main() {\n    ios::sync_with_stdio(false);\n    cin.tie(nullptr);\n\n    // Input format (JSON): { \"s\": \"...\" }\n    return 0;\n}\n`,
    },
  },
];

export const defaultProblemId = problems[0].id;
