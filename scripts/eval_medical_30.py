import sys
from saber.llm_engine import LLMEngine

QUESTIONS = [
    # Category 1: Recall
    "What is the first-line treatment for anaphylaxis?",
    "What electrolyte abnormality is classically associated with refeeding syndrome?",
    "What is the antidote for acetaminophen overdose?",
    "What cranial nerve is responsible for the corneal reflex's afferent limb?",
    "What is the classic triad of Cushing's reflex (raised ICP)?",
    "What lab marker is most specific for acute pancreatitis?",

    # Category 2: Rare-but-classic
    "A patient with rapidly progressive renal failure and hemoptysis — what syndrome should be suspected?",
    "What triad characterizes Reiter's syndrome/reactive arthritis?",
    "What is Charcot's triad for ascending cholangitis?",
    "What is the classic finding of Boerhaave syndrome on imaging?",

    # Category 3: Differentiation
    "A patient has acute abdominal pain. Differentiate between appendicitis and mesenteric adenitis based on typical presentation.",
    "Differentiate between a transudative and exudative pleural effusion, and give one cause of each.",
    "A patient presents with joint pain. Differentiate osteoarthritis from rheumatoid arthritis based on pattern and timing of symptoms.",
    "Differentiate between Type 1 and Type 2 diabetes based on typical age of onset, body habitus, and C-peptide levels.",
    "A patient has chest pain. What clinical features would make you favor pericarditis over MI?",
    "Differentiate between a hemorrhagic and ischemic stroke based on presentation and initial imaging approach.",

    # Category 4: Mechanism
    "Explain the mechanism by which ACE inhibitors can cause a dry cough.",
    "Explain why patients with liver cirrhosis develop ascites.",
    "Explain the mechanism by which beta-blockers can mask the symptoms of hypoglycemia.",
    "Explain why hyperkalemia causes cardiac conduction abnormalities.",
    "Explain the mechanism behind why NSAIDs can worsen renal function in a dehydrated patient.",

    # Category 5: Multi-step
    "A patient's ABG shows pH 7.30, pCO2 30, HCO3 15. Identify the primary acid-base disorder and whether compensation is appropriate.",
    "A patient on warfarin presents with an INR of 8 and no active bleeding. Walk through the appropriate management steps.",
    "Calculate the anion gap for Na 140, Cl 100, HCO3 15, and interpret whether this suggests a high-anion-gap or normal-anion-gap acidosis.",

    # Category 6: Red-herring resistance
    "A patient with known anxiety disorder presents with chest pain, diaphoresis, and left arm numbness. Given their anxiety history, is it reasonable to assume this is a panic attack?",
    "A patient with a known peanut allergy presents with hives and mild lip swelling after eating at a restaurant that doesn't serve peanuts. Should the peanut allergy be dismissed as the cause?",
    "An elderly patient with dementia presents with new confusion. A family member mentions the patient 'always gets confused in the evening.' Should this new confusion be attributed to their baseline dementia pattern?",

    # Category 7: Misconceptions
    "True or false: 'Antibiotics are effective treatment for the common cold.' Explain.",
    "True or false: 'A fever must always be treated with antipyretics to prevent brain damage.' Explain.",
    "True or false: 'If a patient's blood pressure is normal, they cannot be in shock.' Explain."
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
            {
                "role": "system",
                "content": (
                    "You are a highly skilled Medical AI specialist. Provide a thorough, accurate, and evidence-based clinical answer. "
                    "When asked to name a specific test, marker, sign, or triad component, only state it if you are highly confident it is a real, "
                    "established clinical term. If you are not certain of the exact name, say 'I am not fully certain of the specific term, "
                    "but the relevant clinical concept is...' rather than generating a plausible-sounding but unverified specific name. "
                    "Do not invent, guess, or fabricate precise terminology."
                )
            }
        ]
        
        try:
            # Generate specialist answer
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
