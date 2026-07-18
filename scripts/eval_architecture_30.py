import sys
from saber.llm_engine import LLMEngine

QUESTIONS = [
    # =========================================================
    # Category 1: Fundamental Recall (6)
    # =========================================================
    "What are the three pillars of software architecture?",
    "What is the primary purpose of an architecture decision record (ADR)?",
    "Define loose coupling and explain why it is desirable.",
    "What is the difference between scalability and elasticity?",
    "What is eventual consistency in distributed systems?",
    "What is the CAP theorem?",

    # =========================================================
    # Category 2: Core Architecture Patterns (6)
    # =========================================================
    "Differentiate monolithic architecture from microservices.",
    "When would you choose an event-driven architecture over a request-response architecture?",
    "Explain the publisher-subscriber (Pub/Sub) pattern.",
    "What problem does CQRS solve?",
    "Explain the purpose of an API Gateway.",
    "What is the Saga pattern and why is it used?",

    # =========================================================
    # Category 3: Mechanisms (5)
    # =========================================================
    "Explain how horizontal scaling improves system capacity.",
    "Why do distributed systems require consensus algorithms such as Raft or Paxos?",
    "Explain how caching reduces latency.",
    "Why can synchronous microservice communication reduce system resilience?",
    "Explain why idempotency is important in distributed APIs.",

    # =========================================================
    # Category 4: Design Decisions (5)
    # =========================================================
    "A system expects millions of read requests but very few writes. What architecture decisions would you recommend?",
    "A payment service must never process the same transaction twice. How would you design for this requirement?",
    "Your application experiences traffic spikes during product launches. How would you architect for this scenario?",
    "A company wants to migrate from a monolith to microservices. What challenges should they expect?",
    "How would you architect a globally distributed application requiring low latency across continents?",

    # =========================================================
    # Category 5: Failure & Reliability (3)
    # =========================================================
    "A downstream service becomes unavailable. How should a resilient architecture respond?",
    "Explain the purpose of a circuit breaker pattern.",
    "How does bulkheading improve fault isolation?",

    # =========================================================
    # Category 6: Tradeoffs & Reasoning (3)
    # =========================================================
    "Compare SQL and NoSQL databases for an e-commerce platform.",
    "Compare eventual consistency and strong consistency. When is each appropriate?",
    "When should a system favor availability over consistency?",

    # =========================================================
    # Category 7: Misconceptions & Red-Herring Resistance (2)
    # =========================================================
    "True or False: Microservices are always better than monolithic architectures. Explain.",
    "A system is slow. A developer suggests replacing the monolith with microservices. Is that necessarily the correct solution? Explain your reasoning."
]

def main():
    model_path = "models/architecture_v2"
    
    print("=========================================================")
    print(f" SABER Architecture Evaluation Suite (30 Cases)")
    print(f" Loading Model: {model_path}")
    print("=========================================================")
    
    try:
        engine = LLMEngine(model_path)
        engine.__enter__()
    except Exception as e:
        print(f"Failed to load model: {e}")
        sys.exit(1)

    print("\nModel loaded successfully! Beginning evaluation...\n")
    
    for i, question in enumerate(QUESTIONS, 1):
        print(f"---------------------------------------------------------")
        print(f"CASE [{i}/30]")
        print(f"Q: {question}")
        print("---------------------------------------------------------")
        
        # Reset context for each question
        history = [
            {"role": "system", "content": "You are a highly skilled Software Architect. Provide a thorough, accurate, and evidence-based architectural design answer."}
        ]
        
        try:
            ans = engine.generate_with_history(history, new_user_message=question)
            print(f"SABER: {ans}\n\n")
        except Exception as e:
            print(f"SABER: [FAILED TO GENERATE] {e}\n\n")
            
    engine.__exit__(None, None, None)
    print("=========================================================")
    print(" Evaluation Complete! Model unloaded from VRAM.")
    print("=========================================================")

if __name__ == "__main__":
    main()
