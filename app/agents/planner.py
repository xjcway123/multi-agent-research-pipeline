from langchain_core.prompts import ChatPromptTemplate

from app.models.schemas import PipelineState
from app.observability import observe, trace_llm, update_current

PLAN_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a research planner. Given a topic, produce a numbered list of 3-5 specific research questions to answer. Be concise."),
    ("user", "Topic: {topic}\nDepth: {depth}\n\nProduce the research plan as a numbered list."),
])


@observe(name="planner", as_type="agent", capture_input=False, capture_output=False)
def run_planner(state: PipelineState) -> PipelineState:
    from app.config import get_planner_llm

    use_opus = bool(state.get("use_opus_planner"))
    model_label = "claude-opus-4-6" if use_opus else "claude-sonnet-4-5"
    llm = get_planner_llm(use_opus=use_opus)
    chain = PLAN_PROMPT | llm
    payload = {"topic": state["topic"], "depth": state["depth"]}

    with trace_llm("planner.invoke", model=model_label, input=payload):
        response = chain.invoke(payload)
        update_current(output=response.content)

    plan_lines = [line.strip() for line in response.content.split("\n") if line.strip()]
    state["plan"] = plan_lines
    return state
