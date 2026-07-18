import os
import sys
from saber.llm_engine import LLMEngine

TEST_CASES = [
    # ==========================================================
    # DEBUGGING
    # ==========================================================
    {
        "id": 1,
        "category": "Debugging",
        "question": """Find and fix the bug.

def remove_even(nums):
    for n in nums:
        if n % 2 == 0:
            nums.remove(n)
    return nums
"""
    },
    {
        "id": 2,
        "category": "Debugging",
        "question": """Why does this recursive Fibonacci implementation become extremely slow for n=40? How would you improve it?

def fib(n):
    if n <= 1:
        return n
    return fib(n-1) + fib(n-2)
"""
    },
    {
        "id": 3,
        "category": "Debugging",
        "question": """This code sometimes throws KeyError.

counts = {}
for word in words:
    counts[word] += 1

Explain why and fix it."""
    },
    # ==========================================================
    # ALGORITHMS
    # ==========================================================
    {
        "id": 4,
        "category": "Algorithms",
        "question": "Implement Dijkstra's shortest path algorithm."
    },
    {
        "id": 5,
        "category": "Algorithms",
        "question": "Given a list of intervals, merge all overlapping intervals."
    },
    {
        "id": 6,
        "category": "Algorithms",
        "question": "Explain the difference between DFS and BFS. When would you choose one over the other?"
    },
    {
        "id": 7,
        "category": "Algorithms",
        "question": "Design an LRU Cache supporting O(1) get() and put()."
    },
    # ==========================================================
    # COMPLEXITY
    # ==========================================================
    {
        "id": 8,
        "category": "Complexity",
        "question": """Analyze the time and space complexity.

for i in range(n):
    for j in range(i,n):
        print(i,j)
"""
    },
    {
        "id": 9,
        "category": "Complexity",
        "question": "When is O(n log n) preferable to O(n²)? Give practical examples."
    },
    # ==========================================================
    # PYTHON
    # ==========================================================
    {
        "id": 10,
        "category": "Python",
        "question": "Explain generators. When should they be used instead of lists?"
    },
    {
        "id": 11,
        "category": "Python",
        "question": "Explain decorators with an example."
    },
    {
        "id": 12,
        "category": "Python",
        "question": "What is the difference between deep copy and shallow copy?"
    },
    {
        "id": 13,
        "category": "Python",
        "question": "Explain __slots__. What problem does it solve?"
    },
    {
        "id": 14,
        "category": "Python",
        "question": "What are context managers? Implement one yourself."
    },
    # ==========================================================
    # OOP
    # ==========================================================
    {
        "id": 15,
        "category": "OOP",
        "question": "Explain SOLID principles with Python examples."
    },
    {
        "id": 16,
        "category": "OOP",
        "question": "Difference between composition and inheritance?"
    },
    # ==========================================================
    # DATABASES
    # ==========================================================
    {
        "id": 17,
        "category": "Databases",
        "question": "Explain indexing in SQL. Why can too many indexes hurt performance?"
    },
    {
        "id": 18,
        "category": "Databases",
        "question": "Optimize a slow SQL query returning millions of rows."
    },
    # ==========================================================
    # CONCURRENCY
    # ==========================================================
    {
        "id": 19,
        "category": "Concurrency",
        "question": "Explain Python's GIL. When does it matter?"
    },
    {
        "id": 20,
        "category": "Concurrency",
        "question": "When should you use multiprocessing instead of multithreading?"
    },
    {
        "id": 21,
        "category": "Concurrency",
        "question": "What is a race condition? Show how to prevent one in Python."
    },
    # ==========================================================
    # SYSTEM DESIGN
    # ==========================================================
    {
        "id": 22,
        "category": "System Design",
        "question": "Design a URL shortener backend."
    },
    {
        "id": 23,
        "category": "System Design",
        "question": "Design a rate limiter supporting one million users."
    },
    # ==========================================================
    # SECURITY
    # ==========================================================
    {
        "id": 24,
        "category": "Security",
        "question": "Explain SQL Injection and how parameterized queries prevent it."
    },
    {
        "id": 25,
        "category": "Security",
        "question": "Why is using eval() on user input dangerous?"
    },
    # ==========================================================
    # CODE REVIEW
    # ==========================================================
    {
        "id": 26,
        "category": "Code Review",
        "question": """Review this code.

def average(nums):
    total=0
    for i in range(len(nums)):
        total+=nums[i]
    return total/len(nums)

Suggest improvements."""
    },
    {
        "id": 27,
        "category": "Code Review",
        "question": "How would you refactor a 1000-line Python file?"
    },
    # ==========================================================
    # API / BACKEND
    # ==========================================================
    {
        "id": 28,
        "category": "Backend",
        "question": "Design a REST API for an e-commerce checkout service."
    },
    {
        "id": 29,
        "category": "Backend",
        "question": "Explain idempotency in REST APIs."
    },
    # ==========================================================
    # ENGINEERING
    # ==========================================================
    {
        "id": 30,
        "category": "Engineering",
        "question": "A production service suddenly becomes 10x slower after deployment. Walk through your debugging process."
    }
]

def main():
    model_path = "models/coding_v2"
    
    print("=========================================================")
    print(f" SABER Coding Evaluation Suite (30 Cases)")
    print(f" Loading Model: {model_path}")
    print("=========================================================")
    
    if not os.path.exists(model_path):
        print(f"Error: Coding specialist model path '{model_path}' not found.")
        sys.exit(1)
        
    try:
        engine = LLMEngine(model_path)
        engine.__enter__()
    except Exception as e:
        print(f"Failed to load model: {e}")
        sys.exit(1)

    print("\nModel loaded successfully! Beginning evaluation...\n")
    
    for case in TEST_CASES:
        print(f"---------------------------------------------------------")
        print(f"CASE [{case['id']}/30] | Category: {case['category']}")
        print(f"Q: {case['question']}")
        print("---------------------------------------------------------")
        
        history = [
            {
                "role": "system",
                "content": (
                    "You are a coding specialist with expertise in Python, algorithms, "
                    "data structures, and software engineering. Write clean, optimized "
                    "code with clear explanations. Think through your approach step "
                    "by step before writing code."
                )
            }
        ]
        
        try:
            ans = engine.generate_with_history(history, new_user_message=case['question'])
            print(f"SABER: {ans}\n\n")
        except Exception as e:
            print(f"SABER: [FAILED TO GENERATE] {e}\n\n")
            
    engine.__exit__(None, None, None)
    print("=========================================================")
    print(" Evaluation Complete! Model unloaded from VRAM.")
    print("=========================================================")

if __name__ == "__main__":
    main()
