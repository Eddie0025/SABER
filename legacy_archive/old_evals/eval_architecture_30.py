import sys
from saber.llm_engine import LLMEngine

TEST_CASES = [

# ==========================================================
# DISTRIBUTED SYSTEMS (1-6)
# ==========================================================

{
    "id": 1,
    "category": "Distributed Systems",
    "question": "Design a globally distributed messaging platform serving one billion daily users. Explain how you would address latency, replication, consistency, and regional failures."
},

{
    "id": 2,
    "category": "Distributed Systems",
    "question": "Explain why distributed systems cannot simply behave like a single large computer. Discuss network partitions, latency, and independent failures."
},

{
    "id": 3,
    "category": "Distributed Systems",
    "question": "A service must remain available even if an entire data center fails. Design the architecture and justify each major component."
},

{
    "id": 4,
    "category": "Distributed Systems",
    "question": "Design a distributed logging platform capable of ingesting one million events per second."
},

{
    "id": 5,
    "category": "Distributed Systems",
    "question": "Design a globally distributed object storage service similar to Amazon S3."
},

{
    "id": 6,
    "category": "Distributed Systems",
    "question": "Design an architecture for coordinating thousands of autonomous AI agents operating across multiple geographic regions."
},

# ==========================================================
# CONSENSUS & CONSISTENCY (7-12)
# ==========================================================

{
    "id": 7,
    "category": "Consensus",
    "question": "Explain why distributed consensus is difficult even when every machine is functioning correctly."
},

{
    "id": 8,
    "category": "Consensus",
    "question": "Compare Raft, Paxos, and leaderless replication. Explain when each approach is appropriate."
},

{
    "id": 9,
    "category": "Consistency",
    "question": "Compare strong consistency, eventual consistency, causal consistency, and session consistency. Provide practical scenarios for each."
},

{
    "id": 10,
    "category": "CAP Theorem",
    "question": "A hospital system cannot tolerate stale patient data but must remain operational during network partitions. Discuss the architectural tradeoffs without assuming an ideal solution."
},

{
    "id": 11,
    "category": "Replication",
    "question": "Explain why multi-region replication improves availability but can introduce consistency challenges."
},

{
    "id": 12,
    "category": "Consistency",
    "question": "Design a financial transaction system where consistency requirements differ between balance updates, analytics, and reporting."
},

# ==========================================================
# SCALABILITY (13-18)
# ==========================================================

{
    "id": 13,
    "category": "Scalability",
    "question": "An application serving 100,000 users is expected to grow to 100 million users. Describe your scaling strategy over time rather than presenting a final architecture."
},

{
    "id": 14,
    "category": "Load Balancing",
    "question": "Compare round-robin, least-connections, weighted routing, and consistent hashing. Explain when each is preferable."
},

{
    "id": 15,
    "category": "Caching",
    "question": "Explain why caching improves performance but can create correctness problems."
},

{
    "id": 16,
    "category": "Storage",
    "question": "Design a storage architecture for an AI platform generating five petabytes of data annually."
},

{
    "id": 17,
    "category": "Performance",
    "question": "A distributed application experiences excellent CPU utilization but poor user response times. Describe your investigation process."
},

{
    "id": 18,
    "category": "Scalability",
    "question": "Design a notification service capable of delivering one billion notifications per day."
},

# ==========================================================
# ARCHITECTURAL TRADEOFFS (19-24)
# ==========================================================

{
    "id": 19,
    "category": "Tradeoffs",
    "question": "Explain why microservices increase organizational complexity even when they improve technical scalability."
},

{
    "id": 20,
    "category": "Tradeoffs",
    "question": "Compare event-driven architectures with synchronous request-response systems. Discuss latency, coupling, observability, and failure handling."
},

{
    "id": 21,
    "category": "Tradeoffs",
    "question": "Explain when a modular monolith is a better architectural choice than microservices."
},

{
    "id": 22,
    "category": "Tradeoffs",
    "question": "Should every service have its own database? Explain the architectural consequences of both approaches."
},

{
    "id": 23,
    "category": "Tradeoffs",
    "question": "A company wants to move all workloads to Kubernetes. Explain situations where this migration may reduce overall engineering efficiency."
},

{
    "id": 24,
    "category": "Tradeoffs",
    "question": "Compare SQL, NoSQL, graph databases, and time-series databases. Explain which workload each is optimized for."
},

# ==========================================================
# ENTERPRISE DESIGN (25-30)
# ==========================================================

{
    "id": 25,
    "category": "Enterprise Design",
    "question": "Design the architecture for a nationwide electronic health record platform supporting hundreds of millions of patients."
},

{
    "id": 26,
    "category": "Enterprise Design",
    "question": "Design an AI-native operating system capable of orchestrating thousands of specialized AI agents."
},

{
    "id": 27,
    "category": "Enterprise Design",
    "question": "Design the backend architecture for an autonomous vehicle fleet operating across multiple countries."
},

{
    "id": 28,
    "category": "Enterprise Design",
    "question": "Design the architecture of a global stock exchange capable of processing millions of trades per second while maintaining regulatory compliance."
},

{
    "id": 29,
    "category": "Enterprise Design",
    "question": "Design the architecture of a planetary-scale satellite telemetry processing system handling continuous real-time streams."
},

{
    "id": 30,
    "category": "Architecture Review",
    "question": "A startup proposes: API Gateway → Microservices → Shared Database → Redis Cache → Kubernetes. Critique this architecture, identify hidden bottlenecks, single points of failure, scalability concerns, and operational risks, then propose improvements."
}

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
    
    for i, case in enumerate(TEST_CASES, 1):
        question = case["question"]
        category = case["category"]
        print(f"---------------------------------------------------------")
        print(f"CASE [{i}/30] - Category: {category}")
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
