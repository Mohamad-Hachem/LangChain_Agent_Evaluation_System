
from dotenv import load_dotenv
from langchain.agents import create_agent
from deepeval import evaluate
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.integrations.langchain import CallbackHandler
from deepeval.evaluate import CacheConfig
from deepeval.metrics import (
    TaskCompletionMetric,
    StepEfficiencyMetric,
    ArgumentCorrectnessMetric,
    ToolCorrectnessMetric,
    PlanAdherenceMetric,
    PlanQualityMetric,
    GEval,
)
from deepeval.test_case import (
    LLMTestCase,
    ToolCall,
    ToolCallParams,
    SingleTurnParams,
)

load_dotenv()


# 1. Agent
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b


agent = create_agent(
    model="openai:gpt-4",
    tools=[multiply],
    system_prompt="Be concise. Use tools for any math questions.",
)


# 2. Ten goldens
examples = [
    (8123, 62323),
    (7, 92323),
    (12, 34),
    (15, 15),
    (0, 420),
    (-9, 8),
    (123, 456),
    (101, 99),
    (2500, 16),
    (-12, -13),
]

dataset = EvaluationDataset(
    goldens=[
        Golden(
            input=f"What is {a} multiplied by {b}?",
            expected_output=str(a * b),
            expected_tools=[
                ToolCall(
                    name="multiply",
                    input_parameters={"a": a, "b": b},
                )
            ],
        )
        for a, b in examples
    ]
)


# 3. Four trace-based metrics
trace_metrics = [
    TaskCompletionMetric(
        threshold=0.7, model="gpt-4o"
    ),
    StepEfficiencyMetric(
        threshold=0.7, model="gpt-4o"
    ),
    PlanAdherenceMetric(
        threshold=0.7, model="gpt-4o"
    ),
    PlanQualityMetric(
        threshold=0.7, model="gpt-4o"
    ),
]


# 4. Execute agent ONCE per golden
#    Trace the run and collect data for other metrics
test_cases = []

for golden in dataset.evals_iterator(metrics=trace_metrics):

    result = agent.invoke(
        {
            "messages": [
                {"role": "user", "content": golden.input}
            ]
        },
        config={"callbacks": [CallbackHandler()]},
    )

    actual_output = result["messages"][-1].content

    if not isinstance(actual_output, str):
        actual_output = str(actual_output)

    tools_called = []

    for message in result["messages"]:
        for call in getattr(message, "tool_calls", []) or []:
            tools_called.append(
                ToolCall(
                    name=call["name"],
                    input_parameters=call.get("args", {}),
                )
            )

    test_case = LLMTestCase(
        input=golden.input,
        actual_output=actual_output,
        expected_output=golden.expected_output,
        tools_called=tools_called,
        expected_tools=golden.expected_tools,
    )

    test_cases.append(test_case)

    print(f"\nInput: {golden.input}")
    print(f"Actual: {actual_output}")
    print(f"Tools: {tools_called}")


# 5. Three single-turn metrics
single_turn_metrics = [
    ArgumentCorrectnessMetric(
        threshold=0.8,
        model="gpt-4o",
        include_reason=True,
    ),

    ToolCorrectnessMetric(
        threshold=1.0,
        evaluation_params=[
            ToolCallParams.INPUT_PARAMETERS
        ],
        should_exact_match=True,
        include_reason=True,
    ),

    GEval(
        name="Answer Correctness",
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.EXPECTED_OUTPUT,
        ],
        evaluation_steps=[
            "Identify the numerical answer requested.",
            "Compare the actual answer with the expected answer.",
            "Ignore commas, spacing, and explanatory wording.",
            "Give full credit only if the numerical values match.",
        ],
        threshold=0.8,
        model="gpt-4o",
        include_reason=True,
    ),
]


# 6. Evaluate remaining three metrics
evaluate(
    test_cases=test_cases,
    metrics=single_turn_metrics,
    cache_config=CacheConfig(
        use_cache=False,
        write_cache=False,
    ),
)

print("\nCompleted evaluation of all 7 metrics.")
