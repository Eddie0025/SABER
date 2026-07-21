import sys
from saber.llm_engine import LLMEngine

TEST_CASES = [

# ==========================================================
# DIFFERENTIAL DIAGNOSIS (1-8)
# ==========================================================

{
    "id": 1,
    "category": "Differential Diagnosis",
    "question": "A 67-year-old man develops sudden severe chest pain after repeated vomiting. ECG is normal, troponin is negative, and chest X-ray shows mediastinal widening. What is the most likely diagnosis and what features distinguish it from myocardial infarction and aortic dissection?"
},

{
    "id": 2,
    "category": "Differential Diagnosis",
    "question": "A patient presents with fever, confusion, neck stiffness, and photophobia. Explain how you differentiate bacterial meningitis, viral meningitis, encephalitis, and subarachnoid hemorrhage."
},

{
    "id": 3,
    "category": "Differential Diagnosis",
    "question": "Compare delirium, dementia, and depression in an elderly patient with progressive memory loss."
},

{
    "id": 4,
    "category": "Differential Diagnosis",
    "question": "Differentiate Crohn disease from ulcerative colitis using pathology, clinical presentation, imaging, and complications."
},

{
    "id": 5,
    "category": "Differential Diagnosis",
    "question": "Differentiate nephritic syndrome from nephrotic syndrome using pathophysiology rather than memorized features."
},

{
    "id": 6,
    "category": "Differential Diagnosis",
    "question": "A patient presents with jaundice. Explain how you differentiate pre-hepatic, hepatic, and post-hepatic causes."
},

{
    "id": 7,
    "category": "Differential Diagnosis",
    "question": "Differentiate septic shock, cardiogenic shock, hypovolemic shock, and obstructive shock."
},

{
    "id": 8,
    "category": "Differential Diagnosis",
    "question": "Differentiate SIADH from cerebral salt wasting in a neurosurgical patient."
},

# ==========================================================
# PATHOPHYSIOLOGY (9-15)
# ==========================================================

{
    "id": 9,
    "category": "Mechanism",
    "question": "Explain why hypophosphatemia develops during refeeding syndrome."
},

{
    "id": 10,
    "category": "Mechanism",
    "question": "Explain why insulin administration lowers serum potassium."
},

{
    "id": 11,
    "category": "Mechanism",
    "question": "Explain the mechanism producing Cushing's triad during raised intracranial pressure."
},

{
    "id": 12,
    "category": "Mechanism",
    "question": "Why does metabolic acidosis increase serum potassium despite total body potassium often being depleted?"
},

{
    "id": 13,
    "category": "Mechanism",
    "question": "Explain why pulmonary embolism frequently causes respiratory alkalosis."
},

{
    "id": 14,
    "category": "Mechanism",
    "question": "Explain why diabetic ketoacidosis causes both dehydration and metabolic acidosis."
},

{
    "id": 15,
    "category": "Mechanism",
    "question": "Explain the physiological basis of Starling forces and edema formation."
},

# ==========================================================
# EMERGENCY MEDICINE (16-21)
# ==========================================================

{
    "id": 16,
    "category": "Emergency",
    "question": "A trauma patient has hypotension, distended neck veins, and absent breath sounds on the left. What is the immediate management and why?"
},

{
    "id": 17,
    "category": "Emergency",
    "question": "A patient develops anaphylaxis after receiving IV antibiotics. Outline the first five priorities of management in order."
},

{
    "id": 18,
    "category": "Emergency",
    "question": "An unconscious diabetic patient arrives without laboratory results. Describe your immediate diagnostic and treatment priorities."
},

{
    "id": 19,
    "category": "Emergency",
    "question": "A patient with chest pain suddenly develops hypotension and electrical alternans on ECG. Explain the diagnosis and immediate intervention."
},

{
    "id": 20,
    "category": "Emergency",
    "question": "A patient presents with acute ischemic stroke symptoms 90 minutes after onset. Outline the emergency decision-making pathway."
},

{
    "id": 21,
    "category": "Emergency",
    "question": "Explain the immediate priorities in suspected Boerhaave syndrome."
},

# ==========================================================
# CLINICAL REASONING (22-26)
# ==========================================================

{
    "id": 22,
    "category": "Clinical Reasoning",
    "question": "Interpret: pH 7.28, PaCO₂ 60 mmHg, HCO₃⁻ 28 mmol/L. Explain the primary disorder and expected compensation."
},

{
    "id": 23,
    "category": "Clinical Reasoning",
    "question": "Interpret: pH 7.52, PaCO₂ 28 mmHg, HCO₃⁻ 22 mmol/L. Explain the acid-base disturbance and likely causes."
},

{
    "id": 24,
    "category": "Clinical Reasoning",
    "question": "A patient with chronic kidney disease presents with hyperkalemia. Explain the mechanisms and immediate treatment priorities."
},

{
    "id": 25,
    "category": "Clinical Reasoning",
    "question": "Explain why oxygen therapy must be used carefully in some patients with chronic hypercapnic COPD."
},

{
    "id": 26,
    "category": "Clinical Reasoning",
    "question": "A patient develops hypotension immediately after spinal anesthesia. Explain the physiological mechanism."
},

# ==========================================================
# SAFETY & PHARMACOLOGY (27-30)
# ==========================================================

{
    "id": 27,
    "category": "Safety",
    "question": "Explain why potassium should generally never be administered as an IV push."
},

{
    "id": 28,
    "category": "Pharmacology",
    "question": "Compare unfractionated heparin, low-molecular-weight heparin, and direct oral anticoagulants. Focus on mechanisms, monitoring, reversal, and clinical use."
},

{
    "id": 29,
    "category": "Safety",
    "question": "Explain why broad-spectrum antibiotics should not be started indiscriminately even in seriously ill patients."
},

{
    "id": 30,
    "category": "Clinical Integration",
    "question": "A patient with diabetes, chronic kidney disease, heart failure, and atrial fibrillation presents with sepsis. Describe your approach to balancing competing treatment priorities."
}

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
    
    for i, case in enumerate(TEST_CASES, 1):
        question = case["question"]
        category = case["category"]
        print(f"---------------------------------------------------------")
        print(f"CASE [{i}/30] - Category: {category}")
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
