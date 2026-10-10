from langchain.agents import create_agent
from deepeval.integrations.langchain import CallbackHandler
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import TaskCompletionMetric, StepEfficiencyMetric, PlanAdherenceMetric
from dotenv import load_dotenv

load_dotenv()

def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[multiply],
    system_prompt="Be concise.",
)

# Goldens are the inputs you want to evaluate.
dataset = EvaluationDataset(goldens=[
    Golden(input="What is 8 multiplied by 6?"),
    Golden(input="What is 7 multiplied by 9?"),
])

metrics = [PlanAdherenceMetric()]


# The `TaskCompletionMetric` is passed into the `evals_iterator`.
for golden in dataset.evals_iterator(metrics=metrics):
    agent.invoke(
        {"messages": [{"role": "user", "content": golden.input}]},
        config={"callbacks": [CallbackHandler()]},
    )