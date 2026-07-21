import sys
from saber.llm_engine import LLMEngine

TEST_CASES = [

# ==========================================================
# UNCERTAINTY MANAGEMENT (1-6)
# ==========================================================

{
    "id": 1,
    "category": "Uncertainty",
    "question": "A CEO asks whether the company should migrate to microservices. You know nothing about the current architecture, team size, traffic, or business goals. Explain why making a recommendation now would be irresponsible, identify the critical missing information, and describe how you would proceed."
},

{
    "id": 2,
    "category": "Uncertainty",
    "question": "Two equally credible studies reach opposite conclusions about the same medical treatment. Describe a systematic process for determining which evidence should carry more weight."
},

{
    "id": 3,
    "category": "Uncertainty",
    "question": "An AI system reports 95% confidence in a prediction, but the input data differs significantly from the training distribution. Explain how this should influence decision-making."
},

{
    "id": 4,
    "category": "Uncertainty",
    "question": "Your specialists disagree because each has incomplete information. Explain when it is better to delay a decision rather than optimize with uncertainty."
},

{
    "id": 5,
    "category": "Uncertainty",
    "question": "A user requests the 'best database' for a new product. Explain why the question itself is underspecified and identify the minimum information required before making a recommendation."
},

{
    "id": 6,
    "category": "Uncertainty",
    "question": "Describe a framework for reasoning under uncertainty when no additional information can be obtained."
},

# ==========================================================
# CONFLICT RESOLUTION (7-12)
# ==========================================================

{
    "id": 7,
    "category": "Conflict Resolution",
    "question": "Architecture recommends eventual consistency, Finance requires strict transactional guarantees, and Product insists on sub-100 ms responses. Produce a balanced recommendation and justify the compromises."
},

{
    "id": 8,
    "category": "Conflict Resolution",
    "question": "Cybersecurity recommends mandatory MFA for every action, while UX argues this will drive users away. Explain how you would reconcile the disagreement."
},

{
    "id": 9,
    "category": "Conflict Resolution",
    "question": "Coding recommends rewriting an application from scratch, Architecture recommends incremental modernization, and Finance refuses additional budget. Resolve the conflict."
},

{
    "id": 10,
    "category": "Conflict Resolution",
    "question": "Medical recommends collecting extensive patient data for better diagnosis, while Cybersecurity recommends aggressive data minimization. Explain how you would balance both objectives."
},

{
    "id": 11,
    "category": "Conflict Resolution",
    "question": "Science recommends delaying product launch for further validation, while Marketing insists on launching immediately. Explain how evidence quality should influence the decision."
},

{
    "id": 12,
    "category": "Conflict Resolution",
    "question": "Five specialists each propose different priorities. Design a repeatable decision process that avoids simply choosing the loudest opinion."
},

# ==========================================================
# DECISION QUALITY (13-18)
# ==========================================================

{
    "id": 13,
    "category": "Decision Quality",
    "question": "Explain the difference between a good decision and a good outcome. Provide examples where a correct decision still produces poor results."
},

{
    "id": 14,
    "category": "Decision Quality",
    "question": "Describe how hindsight bias can distort evaluations of previous engineering decisions."
},

{
    "id": 15,
    "category": "Decision Quality",
    "question": "Explain why maximizing a single metric often harms the overall system."
},

{
    "id": 16,
    "category": "Decision Quality",
    "question": "Describe a situation where doing nothing is the optimal strategic decision."
},

{
    "id": 17,
    "category": "Decision Quality",
    "question": "A recommendation appears optimal today but limits future flexibility. Explain how option value should influence the decision."
},

{
    "id": 18,
    "category": "Decision Quality",
    "question": "Explain how short-term optimization can produce long-term systemic failure."
},

# ==========================================================
# META-COGNITION (19-24)
# ==========================================================

{
    "id": 19,
    "category": "Meta Cognition",
    "question": "Describe how an AI should recognize when it is reasoning beyond its competence."
},

{
    "id": 20,
    "category": "Meta Cognition",
    "question": "Explain why confidence should not always increase with answer length."
},

{
    "id": 21,
    "category": "Meta Cognition",
    "question": "How should an AI detect that it may be hallucinating despite producing a coherent explanation?"
},

{
    "id": 22,
    "category": "Meta Cognition",
    "question": "Describe methods an AI can use to verify its own reasoning before presenting a conclusion."
},

{
    "id": 23,
    "category": "Meta Cognition",
    "question": "Explain why internally consistent reasoning can still produce incorrect conclusions."
},

{
    "id": 24,
    "category": "Meta Cognition",
    "question": "Design a self-critique process that every specialist should perform before submitting its recommendation."
},

# ==========================================================
# SYSTEM-LEVEL REASONING (25-30)
# ==========================================================

{
    "id": 25,
    "category": "System Reasoning",
    "question": "Design a reasoning framework capable of integrating recommendations from Coding, Architecture, Cybersecurity, Medical, Finance, Science, and Orchestrator specialists without assuming any specialist is always correct."
},

{
    "id": 26,
    "category": "System Reasoning",
    "question": "Explain how conflicting optimization objectives should be prioritized when no solution satisfies every stakeholder."
},

{
    "id": 27,
    "category": "System Reasoning",
    "question": "Describe how dependencies between specialist recommendations should influence execution order."
},

{
    "id": 28,
    "category": "System Reasoning",
    "question": "An orchestrator routes a task incorrectly but every downstream specialist performs perfectly. Explain why the final outcome can still fail."
},

{
    "id": 29,
    "category": "System Reasoning",
    "question": "Design a scoring framework that evaluates recommendation quality based on evidence, uncertainty, risk, trade-offs, and reversibility rather than confidence."
},

{
    "id": 30,
    "category": "System Reasoning",
    "question": "Your AI ecosystem now contains eight expert specialists with conflicting recommendations, incomplete evidence, limited computational resources, strict deadlines, and changing user requirements. Design the complete reasoning strategy for producing a final recommendation while minimizing bias, overconfidence, and irreversible mistakes."
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
