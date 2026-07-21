import sys
import json
from saber.llm_engine import LLMEngine

TEST_CASES = [
    # ==========================================================
    # ROUTING TRAPS (1-8)
    # ==========================================================
    {
        "id": 1,
        "category": "Routing Bias",
        "question": "Design an AI-powered ICU patient monitoring platform with encrypted telemetry, anomaly detection, and predictive alerts."
    },
    {
        "id": 2,
        "category": "Routing Bias",
        "question": "Develop a fraud detection system using graph neural networks for a multinational bank."
    },
    {
        "id": 3,
        "category": "Routing Bias",
        "question": "Create an autonomous laboratory capable of conducting biological experiments with robotic automation."
    },
    {
        "id": 4,
        "category": "Routing Bias",
        "question": "Design a secure electronic health record platform supporting remote diagnosis."
    },
    {
        "id": 5,
        "category": "Routing Bias",
        "question": "Build a nationwide vaccine logistics prediction platform."
    },
    {
        "id": 6,
        "category": "Routing Bias",
        "question": "Create an AI system for predicting stock market manipulation."
    },
    {
        "id": 7,
        "category": "Routing Bias",
        "question": "Develop a satellite image analysis system for crop disease prediction."
    },
    {
        "id": 8,
        "category": "Routing Bias",
        "question": "Build an AI-powered legal document review platform with compliance monitoring."
    },
    # ==========================================================
    # MULTI-DOMAIN ROUTING (9-16)
    # ==========================================================
    {
        "id": 9,
        "category": "Multi Domain",
        "question": "Build a telemedicine platform supporting AI diagnosis, cloud deployment, billing, secure messaging, and medical imaging."
    },
    {
        "id": 10,
        "category": "Multi Domain",
        "question": "Design a self-driving taxi fleet management platform."
    },
    {
        "id": 11,
        "category": "Multi Domain",
        "question": "Create a smart manufacturing plant using predictive maintenance and robotics."
    },
    {
        "id": 12,
        "category": "Multi Domain",
        "question": "Develop a cryptocurrency exchange for institutional investors."
    },
    {
        "id": 13,
        "category": "Multi Domain",
        "question": "Build a nationwide electronic voting platform."
    },
    {
        "id": 14,
        "category": "Multi Domain",
        "question": "Develop an AI-powered university management system."
    },
    {
        "id": 15,
        "category": "Multi Domain",
        "question": "Design a digital twin platform for an international airport."
    },
    {
        "id": 16,
        "category": "Multi Domain",
        "question": "Create an autonomous maritime shipping management platform."
    },
    # ==========================================================
    # AMBIGUITY (17-21)
    # ==========================================================
    {
        "id": 17,
        "category": "Clarification",
        "question": "Build an AI assistant for hospitals."
    },
    {
        "id": 18,
        "category": "Clarification",
        "question": "Improve cybersecurity for my company."
    },
    {
        "id": 19,
        "category": "Clarification",
        "question": "I want to digitize my business."
    },
    {
        "id": 20,
        "category": "Clarification",
        "question": "Help me build an intelligent healthcare platform."
    },
    {
        "id": 21,
        "category": "Clarification",
        "question": "Design an enterprise AI platform."
    },
    # ==========================================================
    # TASK DECOMPOSITION (22-26)
    # ==========================================================
    {
        "id": 22,
        "category": "Planning",
        "question": "A government wants to build a nationwide digital healthcare ecosystem."
    },
    {
        "id": 23,
        "category": "Planning",
        "question": "A Fortune 500 company wants to migrate all infrastructure to the cloud."
    },
    {
        "id": 24,
        "category": "Planning",
        "question": "Develop an AI-powered financial trading platform from scratch."
    },
    {
        "id": 25,
        "category": "Planning",
        "question": "Design an autonomous drone delivery ecosystem."
    },
    {
        "id": 26,
        "category": "Planning",
        "question": "Build an international digital identity platform."
    },
    # ==========================================================
    # EXECUTION ORDER (27-30)
    # ==========================================================
    {
        "id": 27,
        "category": "Execution Order",
        "question": "Develop a medical imaging AI startup from concept to deployment. Produce the correct execution order for specialist involvement."
    },
    {
        "id": 28,
        "category": "Execution Order",
        "question": "Create a military cyber-defense platform. Determine the optimal specialist execution sequence."
    },
    {
        "id": 29,
        "category": "Execution Order",
        "question": "Build a Mars habitat management system. Decide which specialists should be involved first and justify the dependency chain."
    },
    {
        "id": 30,
        "category": "Execution Order",
        "question": "Develop an AI operating system capable of managing distributed autonomous agents. Determine routing, dependencies, and execution order."
    }
]

def main():
    model_path = "models/orchestrator_v2"
    
    print("=========================================================")
    print(f" SABER Orchestrator Evaluation Suite v3 (Realistic Queries)")
    print(f" Loading Model: {model_path}")
    print("=========================================================")
    
    from saber.config import SaberConfig
    from saber.registry import SpecialistRegistry
    from saber.audit import AuditLogger
    from saber.orchestrator import Orchestrator
    
    # Initialize components to check programmatic routing
    config = SaberConfig()
    registry = SpecialistRegistry()
    registry.auto_discover()
    audit = AuditLogger()
    orch = Orchestrator(config=config, registry=registry, audit=audit)
    
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
        
        # Calculate programmatic routing
        domain_scores = orch.classify_domains(question)
        routed_specialists = orch.select_specialists(domain_scores)
        print(f"PROGRAMMATIC ROUTING: {routed_specialists}")
        print("Scores:", {d: round(s, 2) for d, s in domain_scores.items() if s > 0.05})
        print("- - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
        
        # Use the exact prompt the model was fine-tuned on
        available_domains = ", ".join(registry.list_domains())
        system_prompt = (
            "You are the SABER Orchestrator. Your sole responsibility is to evaluate "
            "the user's prompt and route it to the correct specialist domains based on the required technical expertise. "
            "For complex system-building, application design, or pipeline engineering requests, you must route to "
            "both 'architecture' (for design) and 'coding' (for implementation) in addition to the specific domain "
            f"(e.g., 'finance', 'medical', 'science'). Available specialists: {available_domains}. "
            "DO NOT answer the user's question. You must output strict JSON matching "
            "the following schema: "
            "{\"route\": [\"domain1\", \"domain2\"], \"confidence\": 0.99, \"multi_domain\": true, \"query_summary\": \"...\"}"
        )
        
        history = [
            {"role": "system", "content": system_prompt}
        ]
        
        try:
            raw_ans = engine.generate_with_history(history, new_user_message=question)
            print(f"Raw Output: {raw_ans}")
            
            # Robust JSON cleaning and extraction
            clean_ans = raw_ans.replace("```json", "").replace("```", "").strip()
            # Find first '{' and last '}' to strip prefixes/suffixes
            start_idx = clean_ans.find("{")
            end_idx = clean_ans.rfind("}")
            if start_idx != -1 and end_idx != -1:
                clean_ans = clean_ans[start_idx:end_idx+1]
                
            parsed = json.loads(clean_ans)
            print(f"Parsed JSON Plan:\n{json.dumps(parsed, indent=2)}\n\n")
        except Exception as e:
            print(f"SABER: [FAILED TO PARSE OR GENERATE] {e}\n\n")
            
    engine.__exit__(None, None, None)
    print("=========================================================")
    print(" Evaluation Complete! Model unloaded from VRAM.")
    print("=========================================================")

if __name__ == "__main__":
    main()
