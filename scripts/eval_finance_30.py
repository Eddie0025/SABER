import sys
from saber.llm_engine import LLMEngine

QUESTIONS = [
    # Category 1: Straightforward recall
    "What is the formula for calculating Net Present Value (NPV)?",
    "Define 'beta' in the context of the Capital Asset Pricing Model (CAPM).",
    "What is the difference between a bond's coupon rate and its yield to maturity?",
    "What does EBITDA stand for, and what does it measure?",
    "What is the difference between a call option and a put option?",
    "What is the debt-to-equity ratio, and what does it indicate about a company?",
    "Define 'quantitative easing' and its primary intended effect on the economy.",
    "What is the difference between nominal GDP and real GDP?",

    # Category 2: Rare-but-classic scenarios
    "A company's inventory is rising faster than its sales growth over several quarters. What does this typically signal about the business?",
    "What is a 'yield curve inversion,' and what has it historically been associated with?",
    "What is the 'Gordon Growth Model' used for, and what is its key limitation?",
    "Explain what happens to bond prices when interest rates rise, and why.",

    # Category 3: Differentiation/reasoning
    "Company A has high revenue growth but negative free cash flow. Company B has slower revenue growth but strong positive free cash flow. Which is the better investment, and what additional information would you want before deciding?",
    "A firm is deciding between debt financing and equity financing for a new project. Under what conditions would debt financing be preferable, and under what conditions would equity be preferable?",
    "Two companies have identical P/E ratios. Does this mean they are equally valued? What else would you need to check?",
    "A country's currency is depreciating rapidly. Explain two distinct scenarios — one where this benefits the economy, and one where it harms it.",
    "Explain the difference between systematic risk and unsystematic risk, and why diversification only addresses one of them.",
    "A company reports strong net income growth, but its operating cash flow is declining. What could explain this discrepancy, and why should an investor be cautious?",

    # Category 4: Mechanistic explanation
    "Explain the mechanism by which central bank interest rate hikes reduce inflation.",
    "Why does a stock's price typically drop by approximately the dividend amount on the ex-dividend date?",
    "Explain why an inverted yield curve can actually cause the recession it predicts, not just forecast it.",
    "Explain the mechanism of how short selling can lead to a 'short squeeze.'",

    # Category 5: Multi-step reasoning
    "A company has $10M in EBITDA, $2M in depreciation, is taxed at 25%, and has $1M in interest expense. Calculate approximate net income and explain each adjustment step.",
    "An investor buys a bond at a discount to face value with 5 years to maturity. Walk through how yield to maturity would compare to the coupon rate, and why.",
    "A portfolio is 60% stocks (expected return 10%) and 40% bonds (expected return 4%). Calculate the expected portfolio return, then explain how adding more bonds changes both expected return and risk.",

    # Category 6: Red herring resistance
    "A company just announced a stock buyback program and its CEO recently sold a large number of personal shares. Does the CEO's stock sale indicate the buyback is a bad sign for the company? Explain your reasoning.",
    "A company's revenue beat analyst expectations, but its stock price dropped 8% the next day. An analyst on TV says this is 'clearly irrational market behavior.' Is that the most likely explanation? What else could explain it?",
    "A country has very low unemployment. Does this necessarily mean its economy is healthy? What could this low number be masking?",

    # Category 7: Misconceptions
    "Someone claims that 'diversification guarantees you won't lose money in a downturn.' Is this true? Explain what diversification actually does and doesn't protect against.",
    "Someone claims that 'a stock split makes a company more valuable because you now own more shares.' Is this true? Explain what actually happens in a stock split."
]

def main():
    model_path = "models/finance_v2"
    
    print("=========================================================")
    print(f" SABER Finance Evaluation Suite (30 Cases)")
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
            {"role": "system", "content": "You are a highly skilled Financial AI specialist. Provide a thorough, accurate, and evidence-based analytical answer."}
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
