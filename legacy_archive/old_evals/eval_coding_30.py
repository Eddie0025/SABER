import os
import sys
from saber.llm_engine import LLMEngine

TEST_CASES = [
    # ==========================================================
    # EXPLANATION & REASONING (1-8)
    # ==========================================================
    {
        "id": 1,
        "category": "Explanation",
        "question": "Without writing any code, explain why recursion is generally avoided for production implementations of graph traversal in very large graphs. Discuss stack usage, memory, debugging, and maintainability."
    },
    {
        "id": 2,
        "category": "Explanation",
        "question": "Explain why Python's asyncio can outperform multithreading for I/O-bound workloads. Do not provide code—focus on how the scheduler works."
    },
    {
        "id": 3,
        "category": "Explanation",
        "question": "A senior engineer says, 'Premature optimization is the root of all evil.' Explain what this means and when optimization should actually begin."
    },
    {
        "id": 4,
        "category": "Explanation",
        "question": "Explain why immutability often reduces bugs in concurrent software systems."
    },
    {
        "id": 5,
        "category": "Explanation",
        "question": "Explain why composition is generally preferred over inheritance in modern software engineering. Include situations where inheritance is still appropriate."
    },
    {
        "id": 6,
        "category": "Explanation",
        "question": "Why can adding more threads actually reduce application performance? Discuss context switching, lock contention, cache coherence, and CPU scheduling."
    },
    {
        "id": 7,
        "category": "Explanation",
        "question": "Explain why microservices are not automatically better than monolithic architectures."
    },
    {
        "id": 8,
        "category": "Explanation",
        "question": "Explain why distributed systems are fundamentally harder to build than single-machine software."
    },
    # ==========================================================
    # TRADEOFFS (9-14)
    # ==========================================================
    {
        "id": 9,
        "category": "Tradeoffs",
        "question": "When would a relational database outperform a NoSQL database? Explain the tradeoffs rather than choosing one universally."
    },
    {
        "id": 10,
        "category": "Tradeoffs",
        "question": "Compare gRPC and REST for an internal microservice architecture. Which situations favor each?"
    },
    {
        "id": 11,
        "category": "Tradeoffs",
        "question": "Explain when optimistic locking should be preferred over pessimistic locking."
    },
    {
        "id": 12,
        "category": "Tradeoffs",
        "question": "When would breadth-first search be preferred over depth-first search? Include shortest-path considerations."
    },
    {
        "id": 13,
        "category": "Tradeoffs",
        "question": "Explain the advantages and disadvantages of event-driven architectures compared with synchronous request-response systems."
    },
    {
        "id": 14,
        "category": "Tradeoffs",
        "question": "A company wants to replace every monolith with microservices. Explain why this may be a poor engineering decision."
    },
    # ==========================================================
    # DATABASES (15-18)
    # ==========================================================
    {
        "id": 15,
        "category": "Database",
        "question": "Explain why adding indexes can speed up SELECT queries but slow INSERT, UPDATE, and DELETE operations."
    },
    {
        "id": 16,
        "category": "Database",
        "question": "A SQL query became ten times slower after the dataset grew from one million rows to fifty million rows. Walk through your optimization process before rewriting the query."
    },
    {
        "id": 17,
        "category": "Database",
        "question": "Explain the N+1 query problem and how ORMs commonly create it."
    },
    {
        "id": 18,
        "category": "Database",
        "question": "Why is database normalization useful, and when is deliberate denormalization the better engineering choice?"
    },
    # ==========================================================
    # CONCURRENCY (19-22)
    # ==========================================================
    {
        "id": 19,
        "category": "Concurrency",
        "question": "Explain the difference between deadlock, livelock, and starvation. Give practical software examples."
    },
    {
        "id": 20,
        "category": "Concurrency",
        "question": "A multithreaded service occasionally returns corrupted data even though no exceptions occur. Describe your debugging strategy."
    },
    {
        "id": 21,
        "category": "Concurrency",
        "question": "Why do race conditions often disappear during debugging but reappear in production?"
    },
    {
        "id": 22,
        "category": "Concurrency",
        "question": "Explain memory visibility problems in multithreaded applications and why atomic operations alone are not always sufficient."
    },
    # ==========================================================
    # SECURITY (23-25)
    # ==========================================================
    {
        "id": 23,
        "category": "Security",
        "question": "Explain why using eval() on user-controlled input can lead to remote code execution. Do not simply say 'it's dangerous'—describe the execution path."
    },
    {
        "id": 24,
        "category": "Security",
        "question": "Explain why parameterized SQL queries prevent SQL injection while manual string concatenation does not."
    },
    {
        "id": 25,
        "category": "Security",
        "question": "A developer stores passwords using SHA-256. Explain why this is insecure despite SHA-256 being cryptographically strong."
    },
    # ==========================================================
    # DEBUGGING (26-28)
    # ==========================================================
    {
        "id": 26,
        "category": "Debugging",
        "question": "A production API suddenly became five times slower after deployment. CPU usage is unchanged, memory usage is stable, but database latency increased. Describe your debugging methodology."
    },
    {
        "id": 27,
        "category": "Debugging",
        "question": "A distributed application intermittently loses messages between services. Explain the systematic debugging process you would follow."
    },
    {
        "id": 28,
        "category": "Debugging",
        "question": "An application works perfectly on one machine but consistently fails in production. List the engineering hypotheses you would investigate before modifying code."
    },
    # ==========================================================
    # CODE REVIEW & DESIGN (29-30)
    # ==========================================================
    {
        "id": 29,
        "category": "Code Review",
        "question": "You are reviewing a pull request containing 3,500 lines of code implementing five unrelated features. Describe your review process and the engineering concerns you would raise."
    },
    {
        "id": 30,
        "category": "Software Design",
        "question": "Design a scalable backend for a URL shortening service capable of serving one billion redirects per day. Focus on architecture, scalability, databases, caching, and fault tolerance rather than implementation code."
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
