def fibonacci(n):
  if n <= 1:
    return n
  # Two recursive calls for each execution
  return fibonacci(n - 1) + fibonacci(n - 2)

# --- Main function to run test cases ---
if __name__ == "__main__":
  # Use a small n, as this gets very slow very fast!
  n = 10 
  result = fibonacci(n)
  print(f"O(2^n) Recursive Fibonacci for n={n}: {result}")
  # Output: O(2^n) Recursive Fibonacci for n=10: 55