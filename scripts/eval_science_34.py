import sys
from saber.llm_engine import LLMEngine

QUESTIONS = [
    # Category 1: Classic/high-yield recall (baseline — should be near 100%)
    "What law explains why a gas's pressure and volume are inversely proportional at constant temperature?",
    "What particle mediates the strong nuclear force?",
    "What is the name of the process by which plants convert light energy into chemical energy?",
    "Which organelle is responsible for ATP production in eukaryotic cells?",
    "What is the term for a reaction that absorbs heat from its surroundings?",
    "What is the primary greenhouse gas responsible for the majority of anthropogenic radiative forcing?",
    "What subatomic particle was discovered using the gold foil experiment, and what did it reveal about atomic structure?",

    # Category 2: Underweighted-but-classic concepts
    "What is the Mpemba effect, and under what conditions has it been observed?",
    "What is the Franck-Hertz experiment, and what did it demonstrate about atomic energy levels?",
    "What is genetic linkage, and how does crossover frequency relate to map distance in centimorgans?",
    "What is the Chandrasekhar limit, and what does exceeding it determine about a star's fate?",
    "What is allopatric vs. sympatric speciation, and what is a specific example of each?",
    "What is the difference between a codominant and incompletely dominant trait, with an example of each?",

    # Category 3: Differentiation tasks
    "Differentiate between an SN1 and SN2 reaction mechanism — what determines which pathway a given substrate favors?",
    "Differentiate between fission and fusion in terms of energy release, binding energy curve position, and required conditions.",
    "Differentiate between a virus and a viroid — structural composition, replication mechanism, and host range.",
    "Differentiate between mitosis and meiosis I specifically at the point of homologous chromosome behavior (not just 'one makes 2 cells, one makes 4').",
    "Differentiate between weather and climate — and specifically, why a single cold winter doesn't disprove long-term warming trends.",
    "Differentiate between an ionic bond, covalent bond, and metallic bond in terms of electron behavior — not just examples.",

    # Category 4: Mechanism / "why" questions
    "Why does ice float on water mechanistically, in terms of hydrogen bonding and molecular geometry?",
    "Why do noble gases have such low reactivity — explain in terms of electron configuration, not just 'they're stable.'",
    "Why does increasing altitude lower the boiling point of water mechanistically?",
    "Why does a catalyst speed up a reaction without being consumed — explain via activation energy and transition state, not just 'it helps the reaction go faster.'",
    "Why is the CMB (cosmic microwave background) considered strong evidence for the Big Bang specifically, mechanistically?",
    "Why do enzymes lose function at high temperatures — explain via protein structure, not just 'they get destroyed.'",

    # Category 5: Close differentials / decoy-prone comparisons
    "A star's light is redshifted. Differentiate between cosmological redshift, gravitational redshift, and Doppler redshift as possible causes, and what evidence would distinguish them.",
    "A population shows reduced genetic diversity. Differentiate between a bottleneck effect and founder effect as possible causes.",
    "A chemical reaction reaches equilibrium. Differentiate between a reaction that's genuinely at equilibrium vs. one that has simply stopped due to a limiting reagent.",
    "An organism shows convergent traits with another distantly related species. Differentiate convergent evolution from homology/common ancestry as the explanation.",

    # Category 6: Multi-step reasoning
    "A radioactive isotope has a half-life of 5,730 years (carbon-14). A sample shows 25% of the original C-14 remaining. Calculate the age, and explain the limits of radiocarbon dating for very old vs. very young samples.",
    "Given a Punnett square scenario (two heterozygous parents for a recessive trait), calculate offspring genotype/phenotype ratios, then explain how this changes if the gene is X-linked instead of autosomal.",

    # Category 7: Red herrings
    "A chemist has a strong personal interest in organic chemistry and asks: what is the boiling point of water at sea level?",
    "A star is observed to have unusually high metallicity for its estimated age, and the observing telescope was recently recalibrated. What does the high metallicity suggest about the star's formation history?",
    "A researcher studying photosynthesis efficiency also mentions the plant species is endangered. What is the rate-limiting step of the light-independent reactions?",

    # Category 8: Common misconception traps
    "Does the Coriolis effect determine which way water drains in a sink or toilet? Explain why or why not.",
    "Do humans only use 10% of their brains? Address the actual neuroscience.",
    "Is evolution a 'theory' in the colloquial sense of 'just a guess,' or in the scientific sense — explain the distinction.",
    "Does glass flow slowly over centuries, explaining thicker bottoms in old windows? Explain the actual reason for the pane thickness variation."
]

def main():
    model_path = "models/science_v2"
    
    print("=========================================================")
    print(f" SABER Science Evaluation Suite (34 Cases)")
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
        print(f"CASE [{i}/34]")
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
