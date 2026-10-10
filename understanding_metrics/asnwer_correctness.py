
from langchain.agents import create_agent
from deepeval import evaluate
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval.evaluate import CacheConfig
from dotenv import load_dotenv

# Step 1: Load environment variables
load_dotenv()


# Step 2: Define tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


# Step 3: Create LangChain agent
agent = create_agent(
    model="openai:gpt-4",
    tools=[multiply],
    system_prompt="Be concise. Use tools for any math questions.",
)


# Step 4: Create evaluation dataset
dataset = EvaluationDataset(
    goldens=[
        Golden(
            input="What is 8123 multiplied by 62323?",
            expected_output="506249729",
        ),
        Golden(
            input="What is 7 multiplied by 92323?",
            expected_output="646261",
        ),
    ]
)


# Step 5: Execute agent and create test cases
for golden in dataset.goldens:

    print(f"\nEvaluating: {golden.input}")

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": golden.input,
                }
            ]
        }
    )

    actual_output = result["messages"][-1].content

    # Convert Golden into LLMTestCase
    test_case = LLMTestCase(
        input=golden.input,
        actual_output=actual_output,
        expected_output=golden.expected_output,
    )

    dataset.add_test_case(test_case)

    print("Agent Response:", actual_output)
    print("Expected Response:", golden.expected_output)


# Step 6: Configure Answer Correctness metric
metric = GEval(
    name="Answer Correctness",
    evaluation_params=[
        SingleTurnParams.INPUT,
        SingleTurnParams.ACTUAL_OUTPUT,
        SingleTurnParams.EXPECTED_OUTPUT,
    ],
    evaluation_steps=[
        "Understand the mathematical question in the input.",
        "Compare the numerical result in the actual output "
        "with the expected output.",
        "Ignore differences in wording, formatting, and commas.",
        "Penalize incorrect numerical values heavily.",
        "Assign the highest score only when the final "
        "numerical answer is correct.",
    ],
    threshold=0.8,
    model="gpt-4o",
    verbose_mode=True,
)


# Step 7: Run evaluation
evaluate(
    test_cases=dataset.test_cases,
    metrics=[metric],
    cache_config=CacheConfig(
        use_cache=False,
        write_cache=False,
    ),
)
