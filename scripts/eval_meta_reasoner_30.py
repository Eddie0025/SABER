import sys
from saber.llm_engine import LLMEngine

QUESTIONS = [
    # =========================================================
    # Category 1 – Agreement Detection (5)
    # =========================================================
    "Expert A:\nAES-256 is symmetric encryption.\nExpert B:\nAES-256 is symmetric encryption.\nEvaluate agreements/conflicts.",
    "Expert A:\nHTTP is stateless.\nExpert B:\nHTTP is stateless.\nEvaluate agreements/conflicts.",
    "Expert A:\nCAP stands for Consistency, Availability, Partition tolerance.\nExpert B:\nCAP stands for Consistency, Availability, Partition tolerance.\nEvaluate agreements/conflicts.",
    "Expert A:\nSQL databases support ACID.\nExpert B:\nSQL databases support ACID.\nEvaluate agreements/conflicts.",
    "Expert A:\nMicroservices communicate over APIs.\nExpert B:\nMicroservices communicate over APIs.\nEvaluate agreements/conflicts.",

    # =========================================================
    # Category 2 – Direct Contradictions (8)
    # =========================================================
    "Expert A:\nEventual consistency guarantees immediate synchronization.\nExpert B:\nEventual consistency allows temporary inconsistency.\nResolve this contradiction.",
    "Expert A:\nMicroservices are always better.\nExpert B:\nIt depends on requirements.\nResolve this contradiction.",
    "Expert A:\nCAP allows all three simultaneously.\nExpert B:\nCAP allows only two guarantees at the same time.\nResolve this contradiction.",
    "Expert A:\nAES is asymmetric.\nExpert B:\nAES is symmetric.\nResolve this contradiction.",
    "Expert A:\nREST requires HTTP.\nExpert B:\nREST is an architectural style independent of protocols.\nResolve this contradiction.",
    "Expert A:\nDocker replaces Kubernetes.\nExpert B:\nDocker and Kubernetes solve different problems and work together.\nResolve this contradiction.",
    "Expert A:\nSQL scales only vertically.\nExpert B:\nSQL supports horizontal scaling strategies like sharding and replication.\nResolve this contradiction.",
    "Expert A:\nMonoliths cannot scale.\nExpert B:\nMonoliths can scale vertically and horizontally (replicated monoliths).\nResolve this contradiction.",

    # =========================================================
    # Category 3 – Partial Conflicts (6)
    # =========================================================
    "Expert A recommends CQRS.\nExpert B recommends caching.\nArbitrate this partial conflict.",
    "Expert A recommends Redis.\nExpert B recommends Memcached.\nArbitrate this partial conflict.",
    "Expert A prefers PostgreSQL.\nExpert B prefers Cassandra.\nArbitrate this partial conflict.",
    "Expert A suggests synchronous communication.\nExpert B suggests asynchronous messaging.\nArbitrate this partial conflict.",
    "Expert A suggests JWT.\nExpert B suggests OAuth.\nArbitrate this partial conflict.",
    "Expert A recommends Kubernetes.\nExpert B recommends ECS.\nArbitrate this partial conflict.",

    # =========================================================
    # Category 4 – Preference Arbitration (5)
    # =========================================================
    "Expert A solution:\nFast but less secure.\nExpert B:\nSlower but secure.\nArbitrate these preferences based on security priorities.",
    "Expert A:\nSimple architecture.\nExpert B:\nHighly scalable architecture.\nArbitrate these preferences based on scalability needs.",
    "Expert A:\nLow latency.\nExpert B:\nHigh consistency.\nArbitrate these preferences based on consistency requirements.",
    "Expert A:\nLow cost.\nExpert B:\nHigh availability.\nArbitrate these preferences based on availability SLAs.",
    "Expert A:\nRapid development.\nExpert B:\nMaintainability.\nArbitrate these preferences based on long-term maintenance goals.",

    # =========================================================
    # Category 5 – Hallucination Detection (3)
    # =========================================================
    "Expert A:\nPANCA-3 is used to diagnose pancreatitis.\nExpert B:\nSerum lipase is preferred.\nDetermine which claim is a hallucination.",
    "Expert A:\nDocker is an operating system.\nExpert B:\nDocker is a container platform.\nDetermine which claim is a hallucination.",
    "Expert A:\nCAP has four guarantees.\nExpert B:\nCAP has three guarantees.\nDetermine which claim is a hallucination.",

    # =========================================================
    # Category 6 – Multi-Expert Arbitration (3)
    # =========================================================
    "Architecture:\nUse microservices.\nCyber:\nUse monolith for simplicity and reduced attack surface.\nOperations:\nPrefer Kubernetes.\nProduce a unified recommendation.",
    "Medical:\nImmediate surgery is required.\nAI:\nGather more diagnostic information first to rule out myocarditis.\nProduce a final decision.",
    "Architecture:\nUse Kafka.\nAI:\nUse RabbitMQ.\nOperations:\nUse NATS.\nSelect the most appropriate solution based on system throughput requirements."
]

def main():
    model_path = "models/meta_reasoner_v2"
    
    print("=========================================================")
    print(f" SABER Meta-Reasoner Evaluation Suite (30 Cases)")
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
            {"role": "system", "content": "You are the SABER Meta-Reasoner. Resolve contradictions, arbitrate preferences, identify hallucinations, and output a unified recommendation."}
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
