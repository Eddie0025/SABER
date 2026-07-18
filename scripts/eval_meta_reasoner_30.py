import sys
from saber.llm_engine import LLMEngine

TEST_CASES = [
    # ==========================================================
    # UNDER-SPECIFIED DECISIONS (1-8)
    # ==========================================================
    {
        "id": 1,
        "category": "Insufficient Information",
        "question": "Which cloud provider is best: AWS, Azure, or Google Cloud?"
    },
    {
        "id": 2,
        "category": "Insufficient Information",
        "question": "Should a startup choose PostgreSQL or MongoDB?"
    },
    {
        "id": 3,
        "category": "Insufficient Information",
        "question": "Should a company build a mobile app or a web application first?"
    },
    {
        "id": 4,
        "category": "Insufficient Information",
        "question": "Is Kubernetes always better than Docker Compose?"
    },
    {
        "id": 5,
        "category": "Insufficient Information",
        "question": "Should an AI startup use Python or Rust?"
    },
    {
        "id": 6,
        "category": "Insufficient Information",
        "question": "Should microservices always replace a monolith?"
    },
    {
        "id": 7,
        "category": "Insufficient Information",
        "question": "Should data always be encrypted end-to-end?"
    },
    {
        "id": 8,
        "category": "Insufficient Information",
        "question": "Should every production system implement Zero Trust?"
    },
    # ==========================================================
    # TRADEOFF REASONING (9-15)
    # ==========================================================
    {
        "id": 9,
        "category": "Tradeoffs",
        "question": "Explain the tradeoff between consistency, availability, and partition tolerance in distributed systems. Do not recommend one universally."
    },
    {
        "id": 10,
        "category": "Tradeoffs",
        "question": "A distributed cache improves latency but risks stale reads. Explain the engineering tradeoff."
    },
    {
        "id": 11,
        "category": "Tradeoffs",
        "question": "Higher encryption strength often increases computational overhead. Explain when stronger encryption is not automatically the best engineering decision."
    },
    {
        "id": 12,
        "category": "Tradeoffs",
        "question": "Explain why maximizing accuracy in an AI model may reduce fairness or interpretability."
    },
    {
        "id": 13,
        "category": "Tradeoffs",
        "question": "Explain why maximizing availability can reduce consistency in distributed architectures."
    },
    {
        "id": 14,
        "category": "Tradeoffs",
        "question": "Should engineering teams optimize for developer productivity or runtime performance? Explain the conditions affecting the answer."
    },
    {
        "id": 15,
        "category": "Tradeoffs",
        "question": "Explain the tradeoff between rapid product delivery and software quality."
    },
    # ==========================================================
    # THREE-EXPERT SYNTHESIS (16-22)
    # ==========================================================
    {
        "id": 16,
        "category": "Multi-Expert",
        "question": "Architecture recommends microservices, Finance recommends reducing infrastructure costs, and Coding recommends keeping the existing monolith. Produce a synthesized recommendation."
    },
    {
        "id": 17,
        "category": "Multi-Expert",
        "question": "Medical recommends maximum patient safety, Finance recommends minimizing operational cost, and Cybersecurity recommends strict authentication. Produce a balanced decision."
    },
    {
        "id": 18,
        "category": "Multi-Expert",
        "question": "Science recommends further experimentation, Finance wants immediate commercialization, and Architecture believes the technology is not production ready. Resolve the disagreement."
    },
    {
        "id": 19,
        "category": "Multi-Expert",
        "question": "Coding recommends rapid implementation, Cybersecurity recommends delaying release for security review, and Finance demands launch before quarter-end."
    },
    {
        "id": 20,
        "category": "Multi-Expert",
        "question": "Coding recommends PostgreSQL, Architecture recommends Cassandra, and Finance recommends minimizing infrastructure cost."
    },
    {
        "id": 21,
        "category": "Multi-Expert",
        "question": "Medical recommends collecting more patient data, Cybersecurity recommends minimizing stored data, and Science requests larger datasets for research."
    },
    {
        "id": 22,
        "category": "Multi-Expert",
        "question": "Coding recommends PostgreSQL, Architecture recommends Cassandra, and Finance recommends minimizing infrastructure cost."
    },
    # ==========================================================
    # PRIORITIZATION (23-26)
    # ==========================================================
    {
        "id": 23,
        "category": "Prioritization",
        "question": "A hospital system has security vulnerabilities, poor user experience, slow databases, and inaccurate AI predictions. Prioritize remediation."
    },
    {
        "id": 24,
        "category": "Prioritization",
        "question": "An AI startup has technical debt, limited funding, increasing customer demand, and compliance deadlines. Determine the priority order."
    },
    {
        "id": 25,
        "category": "Prioritization",
        "question": "A cloud platform suffers from latency, increasing costs, and occasional outages. Which problem should be addressed first, and why?"
    },
    {
        "id": 26,
        "category": "Prioritization",
        "question": "A national healthcare platform has identified privacy concerns, performance bottlenecks, missing features, and usability complaints. Produce a justified priority sequence."
    },
    # ==========================================================
    # META REASONING (27-30)
    # ==========================================================
    {
        "id": 27,
        "category": "Meta Reasoning",
        "question": "Describe a situation where deliberately delaying a technical decision produces a better long-term outcome."
    },
    {
        "id": 28,
        "category": "Meta Reasoning",
        "question": "Explain why insufficient information should sometimes result in asking additional questions rather than making recommendations."
    },
    {
        "id": 29,
        "category": "Meta Reasoning",
        "question": "How should an AI system decide when two expert recommendations are genuinely incompatible versus merely optimizing different objectives?"
    },
    {
        "id": 30,
        "category": "Meta Reasoning",
        "question": "Design a reasoning strategy for resolving conflicts among eight domain specialists without assuming that any one specialist is always correct."
    }
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
    
    for case in TEST_CASES:
        question = case["question"]
        print(f"---------------------------------------------------------")
        print(f"CASE [{case['id']}/30] | Category: {case['category']}")
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
