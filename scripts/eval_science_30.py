import sys
from saber.llm_engine import LLMEngine

TEST_CASES = [

# ==========================================================
# SCIENTIFIC REASONING (1-6)
# ==========================================================

{
    "id": 1,
    "category": "Scientific Reasoning",
    "question": "A correlation study shows that people who drink more coffee live longer. Explain why this result alone does not establish causation and identify possible confounding variables."
},

{
    "id": 2,
    "category": "Scientific Reasoning",
    "question": "A published experiment cannot be reproduced by five independent laboratories. Explain the possible scientific explanations before concluding misconduct."
},

{
    "id": 3,
    "category": "Scientific Reasoning",
    "question": "Explain why a statistically significant result may still lack practical or clinical significance."
},

{
    "id": 4,
    "category": "Scientific Reasoning",
    "question": "Describe the role of falsifiability in the scientific method and why it is essential for scientific progress."
},

{
    "id": 5,
    "category": "Scientific Reasoning",
    "question": "Explain why increasing sample size generally improves statistical power but does not eliminate systematic bias."
},

{
    "id": 6,
    "category": "Scientific Reasoning",
    "question": "A machine learning model predicts disease with 99% accuracy. Explain why this number alone is insufficient to judge its scientific usefulness."
},

# ==========================================================
# EXPERIMENTAL DESIGN (7-12)
# ==========================================================

{
    "id": 7,
    "category": "Experimental Design",
    "question": "Design an experiment to evaluate whether a new antibiotic is more effective than the current standard of care. Specify controls, randomization, blinding, primary endpoints, and sources of bias."
},

{
    "id": 8,
    "category": "Experimental Design",
    "question": "Design an experiment to determine whether a fertilizer increases crop yield while minimizing environmental confounders."
},

{
    "id": 9,
    "category": "Experimental Design",
    "question": "An experiment reports positive results but has no control group. Explain why the conclusions are unreliable."
},

{
    "id": 10,
    "category": "Experimental Design",
    "question": "Explain the purpose of randomization, blinding, and placebo controls, and discuss situations where each may not be feasible."
},

{
    "id": 11,
    "category": "Experimental Design",
    "question": "Design a study to evaluate the long-term safety of an implanted medical device over ten years."
},

{
    "id": 12,
    "category": "Experimental Design",
    "question": "Explain the strengths and weaknesses of observational studies compared with randomized controlled trials."
},

# ==========================================================
# GENETICS (13-18)
# ==========================================================

{
    "id": 13,
    "category": "Genetics",
    "question": "Explain the difference between genotype, phenotype, penetrance, and expressivity. Provide examples demonstrating why these concepts should not be confused."
},

{
    "id": 14,
    "category": "Genetics",
    "question": "A disease appears in every generation but not every child of an affected parent develops symptoms. Explain the possible genetic mechanisms."
},

{
    "id": 15,
    "category": "Genetics",
    "question": "Compare autosomal dominant, autosomal recessive, X-linked dominant, X-linked recessive, and mitochondrial inheritance using pedigree reasoning."
},

{
    "id": 16,
    "category": "Genetics",
    "question": "Explain why identical twins can develop different diseases despite having nearly identical genomes."
},

{
    "id": 17,
    "category": "Genetics",
    "question": "Describe how CRISPR-Cas9 performs gene editing and identify its major technical limitations and ethical concerns."
},

{
    "id": 18,
    "category": "Genetics",
    "question": "Explain the relationship between DNA, RNA, transcription, translation, and protein synthesis without reducing the explanation to memorized definitions."
},

# ==========================================================
# STATISTICS (19-24)
# ==========================================================

{
    "id": 19,
    "category": "Statistics",
    "question": "Explain Type I error, Type II error, statistical power, and effect size. Describe how changing the significance threshold influences these quantities."
},

{
    "id": 20,
    "category": "Statistics",
    "question": "A clinical trial reports p = 0.04. Explain what this value means and identify three common misconceptions about p-values."
},

{
    "id": 21,
    "category": "Statistics",
    "question": "A diagnostic test has 99% sensitivity and 99% specificity. Explain why its positive predictive value may still be poor in a low-prevalence population."
},

{
    "id": 22,
    "category": "Statistics",
    "question": "Explain why confidence intervals often provide more useful information than p-values alone."
},

{
    "id": 23,
    "category": "Statistics",
    "question": "Researchers perform twenty independent statistical tests without correcting for multiple comparisons. Explain why this increases the risk of false discoveries."
},

{
    "id": 24,
    "category": "Statistics",
    "question": "A model achieves excellent performance on training data but poor performance on new data. Explain this result using statistical learning concepts."
},

# ==========================================================
# INTEGRATED SCIENCE (25-30)
# ==========================================================

{
    "id": 25,
    "category": "Integrated Science",
    "question": "Climate scientists, economists, and engineers disagree about the best strategy for reducing carbon emissions. Explain how scientific evidence should be integrated into multidisciplinary decision-making."
},

{
    "id": 26,
    "category": "Integrated Science",
    "question": "A pharmaceutical company wants to accelerate clinical trials using AI. Explain the scientific opportunities and risks, including bias, validation, and regulatory considerations."
},

{
    "id": 27,
    "category": "Integrated Science",
    "question": "Explain how computational simulations complement laboratory experiments and discuss situations where simulations may produce misleading conclusions."
},

{
    "id": 28,
    "category": "Integrated Science",
    "question": "A published paper reports extraordinary results that contradict decades of established evidence. Describe the scientific process you would follow before accepting or rejecting the findings."
},

{
    "id": 29,
    "category": "Integrated Science",
    "question": "Explain why interdisciplinary collaboration has become increasingly important in modern scientific research."
},

{
    "id": 30,
    "category": "Research Leadership",
    "question": "You are leading a multidisciplinary research program involving biology, AI, materials science, and medicine. Design a framework for prioritizing hypotheses, allocating resources, validating discoveries, and deciding when evidence is sufficient to move toward commercialization."
}

]

def main():
    model_path = "models/science_v2"
    
    print("=========================================================")
    print(f" SABER Science Evaluation Suite (30 Cases)")
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
            {"role": "system", "content": "You are a highly skilled Scientific AI specialist. Provide a thorough, accurate, and evidence-based clinical/scientific answer."}
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
