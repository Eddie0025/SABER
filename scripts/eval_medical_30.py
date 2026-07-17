import sys
from saber.llm_engine import LLMEngine

QUESTIONS = [
    # Category 1: Classic presentations
    "Sudden tearing chest pain radiating to the back, unequal blood pressures in both arms. What is the diagnosis?",
    "Fever, neck stiffness, petechial rash in a young adult. What is the diagnosis?",
    "Crushing substernal chest pain, diaphoresis, radiation to left arm, ST elevation. What is the diagnosis?",
    "Polyuria, polydipsia, fruity breath odor, Kussmaul breathing. What is the diagnosis?",
    "Unilateral facial droop, arm drift, slurred speech, sudden onset. What is the diagnosis?",

    # Category 2: Classic-but-underweighted presentations
    "Recurrent DVTs, livedo reticularis, recurrent miscarriages, positive lupus anticoagulant. What is the diagnosis?",
    "Hyperpigmentation, hypotension, hyponatremia, hyperkalemia, fatigue. What is the diagnosis?",
    "Rapidly progressive muscle weakness, absent reflexes, ascending paralysis 2 weeks post-GI infection. What is the diagnosis?",
    "Copper-colored rash on palms and soles, painless genital ulcer weeks earlier. What is the diagnosis?",
    "Colicky abdominal pain, hypertension, kidney stones, constipation, and elevated calcium. What is the diagnosis?",
    "Young woman, recurrent oral/genital ulcers, uveitis, positive pathergy test. What is the diagnosis?",
    "Triad of hypotension, JVD, muffled heart sounds after chest trauma. What is the diagnosis?",

    # Category 3: Decoy-prone / close differentials
    "Patient presents with confusion, hypotension, hyperkalemia, hyponatremia, and eosinophilia. Differentiate between Addisonian crisis, SIADH, and diabetes insipidus.",
    "Patient presents with sudden severe headache, focal neuro deficit, and history of anticoagulant use. Differentiate between SAH, ischemic stroke, and intracerebral hemorrhage.",
    "Patient presents with fever, jaundice, and RUQ pain. Differentiate between simple cholecystitis and ascending cholangitis.",
    "Patient presents with chest pain worse lying down, relieved leaning forward, and a friction rub. Differentiate between pericarditis, MI, and PE.",
    "Young patient presents with tachycardia, hyperthermia, and muscle rigidity after starting an antipsychotic. Differentiate between NMS, serotonin syndrome, and malignant hyperthermia.",

    # Category 4: Mechanism / "why" questions
    "Why is oxygen given cautiously in COPD patients with chronic CO2 retention? Explain the mechanisms involving V/Q mismatch and the Haldane effect.",
    "Why does rapid correction of hyponatremia cause central pontine myelinolysis mechanistically?",
    "Why are ACE inhibitors contraindicated in bilateral renal artery stenosis mechanistically?",
    "Why does metoclopramide cause extrapyramidal symptoms mechanistically?",
    "Why can succinylcholine cause hyperkalemia in burn/crush injury patients specifically?",

    # Category 5: Drug interactions / avoid-this-class questions
    "A patient on warfarin is started on an antibiotic. Which antibiotic classes raise INR significantly, and why?",
    "In a patient with G6PD deficiency, which drug classes precipitate hemolysis?",
    "In a patient on an MAOI, what food/drug combinations are dangerous and why?",
    "In a patient with prolonged QT, which antibiotic and antiemetic classes should be avoided?",

    # Category 6: Multi-step reasoning
    "A patient presents with an elevated TSH, low free T4, and symptoms of fatigue and weight gain. What is the diagnosis, how do you confirm it, what is the treatment, and what do you recheck and when?",
    "A patient presents with microcytic anemia. What is the next test to distinguish iron deficiency vs. thalassemia trait, how do you confirm it, and what is the treatment implication if wrong?",

    # Category 7: Red herrings
    "A patient with a strong family history of pancreatic cancer presents with a straightforward viral URI, rhinorrhea, and a sore throat for 2 days. What is the diagnosis and management?",
    "A patient recently started a new medication (statins) and presents with acute right lower quadrant abdominal pain, rebound tenderness, and a fever. What is the likely diagnosis and management?"
]

def main():
    model_path = "models/medical_v2"
    
    print("=========================================================")
    print(f" SABER Medical Evaluation Suite (30 Cases)")
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
        
        # Reset context for each question so previous answers don't bias the model
        history = [
            {"role": "system", "content": "You are a highly skilled Medical AI specialist. Provide a thorough, accurate, and evidence-based clinical answer."}
        ]
        
        try:
            response = engine.generate_with_history(history, new_user_message=question)
            print(f"SABER: {response}\n\n")
        except Exception as e:
            print(f"SABER: [FAILED TO GENERATE] {e}\n\n")
            
    engine.__exit__(None, None, None)
    print("=========================================================")
    print(" Evaluation Complete! Model unloaded from VRAM.")
    print("=========================================================")

if __name__ == "__main__":
    main()
