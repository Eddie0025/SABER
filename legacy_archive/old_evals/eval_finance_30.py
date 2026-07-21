import sys
from saber.llm_engine import LLMEngine

TEST_CASES = [

# ==========================================================
# FINANCIAL STATEMENTS (1-6)
# ==========================================================

{
    "id": 1,
    "category": "Financial Statements",
    "question": "A company reports Revenue of $250M, EBITDA of $60M, Depreciation of $12M, Interest Expense of $8M, Tax Rate of 25%, and no extraordinary items. Calculate Operating Income, Pre-Tax Income, Tax Expense, and Net Income. Show your reasoning."
},

{
    "id": 2,
    "category": "Financial Statements",
    "question": "Explain why EBITDA should never be interpreted as cash flow."
},

{
    "id": 3,
    "category": "Financial Statements",
    "question": "A company reports strong profits but continuously negative operating cash flow. Explain at least five possible reasons."
},

{
    "id": 4,
    "category": "Financial Statements",
    "question": "Explain how a $100 million increase in depreciation affects the income statement, cash flow statement, and balance sheet."
},

{
    "id": 5,
    "category": "Financial Statements",
    "question": "Explain why revenue growth alone may actually reduce shareholder value."
},

{
    "id": 6,
    "category": "Financial Statements",
    "question": "Compare gross margin, operating margin, EBITDA margin, and net margin. Explain what each reveals about a business."
},

# ==========================================================
# VALUATION (7-12)
# ==========================================================

{
    "id": 7,
    "category": "Valuation",
    "question": "Compare Discounted Cash Flow (DCF), Comparable Company Analysis, and Precedent Transactions. When is each most appropriate?"
},

{
    "id": 8,
    "category": "Valuation",
    "question": "Explain why two companies with identical revenue can have dramatically different enterprise values."
},

{
    "id": 9,
    "category": "Valuation",
    "question": "Interest rates increase substantially. Explain how this affects company valuations and why."
},

{
    "id": 10,
    "category": "Valuation",
    "question": "Explain why a rapidly growing startup may still deserve a lower valuation than a slower-growing competitor."
},

{
    "id": 11,
    "category": "Valuation",
    "question": "Compare Price-to-Earnings, EV/EBITDA, Price-to-Sales, and Price-to-Book. Explain the situations where each metric is misleading."
},

{
    "id": 12,
    "category": "Valuation",
    "question": "A SaaS company has exceptional revenue growth but consistently negative free cash flow. Explain how you would evaluate it."
},

# ==========================================================
# CORPORATE FINANCE (13-18)
# ==========================================================

{
    "id": 13,
    "category": "Corporate Finance",
    "question": "Explain the tradeoffs between financing expansion through debt versus issuing new equity."
},

{
    "id": 14,
    "category": "Corporate Finance",
    "question": "A company has $5 billion in excess cash. Compare dividends, share buybacks, acquisitions, debt repayment, and internal investment."
},

{
    "id": 15,
    "category": "Corporate Finance",
    "question": "Explain why increasing leverage may both increase shareholder returns and increase bankruptcy risk."
},

{
    "id": 16,
    "category": "Corporate Finance",
    "question": "Explain the relationship between Return on Equity, leverage, and profitability."
},

{
    "id": 17,
    "category": "Corporate Finance",
    "question": "Why is maximizing quarterly earnings not always aligned with maximizing long-term shareholder value?"
},

{
    "id": 18,
    "category": "Corporate Finance",
    "question": "Explain why reducing costs indiscriminately can destroy long-term company value."
},

# ==========================================================
# UNIT ECONOMICS (19-24)
# ==========================================================

{
    "id": 19,
    "category": "Unit Economics",
    "question": "A SaaS company has CAC of $800, annual gross profit per customer of $400, annual churn of 40%, and gross margin of 80%. Evaluate whether the business model is sustainable."
},

{
    "id": 20,
    "category": "Unit Economics",
    "question": "Explain why high revenue growth combined with poor unit economics often leads to startup failure."
},

{
    "id": 21,
    "category": "Unit Economics",
    "question": "Compare customer lifetime value (LTV) and customer acquisition cost (CAC). Explain why both metrics should be interpreted together."
},

{
    "id": 22,
    "category": "Unit Economics",
    "question": "A company doubles marketing spend but revenue barely increases. Explain the possible financial explanations."
},

{
    "id": 23,
    "category": "Unit Economics",
    "question": "A marketplace business has rapidly growing GMV but declining profitability. Explain why these two trends can coexist."
},

{
    "id": 24,
    "category": "Unit Economics",
    "question": "Explain how churn influences valuation, profitability, and growth simultaneously."
},

# ==========================================================
# STRATEGIC FINANCE (25-30)
# ==========================================================

{
    "id": 25,
    "category": "Strategic Finance",
    "question": "A technology company must choose between maximizing revenue growth, profitability, and free cash flow. Explain how this decision depends on business stage and market conditions."
},

{
    "id": 26,
    "category": "Strategic Finance",
    "question": "A recession begins unexpectedly. Explain how capital allocation priorities should change."
},

{
    "id": 27,
    "category": "Strategic Finance",
    "question": "A company reports record earnings while losing market share. Explain why investors may react negatively."
},

{
    "id": 28,
    "category": "Strategic Finance",
    "question": "Explain why financial decisions should always be evaluated alongside operational, technological, and strategic considerations."
},

{
    "id": 29,
    "category": "Strategic Finance",
    "question": "A CEO proposes maximizing EBITDA at all costs. Critically evaluate this strategy."
},

{
    "id": 30,
    "category": "Integrated Finance",
    "question": "An AI startup is deciding whether to expand internationally, acquire a competitor, invest in new R&D, or preserve cash. Develop a structured financial decision-making framework rather than recommending a single option."
}

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
    
    for i, case in enumerate(TEST_CASES, 1):
        question = case["question"]
        category = case["category"]
        print(f"---------------------------------------------------------")
        print(f"CASE [{i}/30] - Category: {category}")
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
