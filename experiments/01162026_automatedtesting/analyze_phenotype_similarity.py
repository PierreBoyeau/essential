import os
import glob
from typing import TypedDict, Optional

from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage


# Define the state for the graph
class AnalysisState(TypedDict):
    file_path: str
    report_content: Optional[str]
    analysis_output: Optional[str]


# Define nodes


def load_report(state: AnalysisState) -> AnalysisState:
    """Reads the target report.json."""
    if "GOOGLE_API_KEY" not in os.environ:
        raise ValueError("GOOGLE_API_KEY environment variable not set.")

    file_path = state["file_path"]
    try:
        with open(file_path, "r") as f:
            content = f.read()
        return {"report_content": content}
    except Exception as e:
        print(f"Error loading report {file_path}: {e}")
        return {"report_content": None}


def generate_analysis(state: AnalysisState) -> AnalysisState:
    """Generates analysis using Gemini 3 Pro."""
    report_content = state.get("report_content")
    if not report_content:
        return {"analysis_output": None}
    prompt_path = "/workspace/experiments/01162026_automatedtesting/prompts/phenotype_similarity_analysis.md"
    try:
        with open(prompt_path, "r") as f:
            system_prompt = f.read()
    except Exception as e:
        print(f"Error loading prompt {prompt_path}: {e}")
        return {"analysis_output": None}

    llm = ChatGoogleGenerativeAI(model="gemini-3-pro-preview")

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Here is the report data:\n{report_content}"),
    ]

    try:
        response = llm.invoke(messages)
        content = response.text
        return {"analysis_output": content}
    except Exception as e:
        print(f"Error generating analysis for {state['file_path']}: {e}")
        return {"analysis_output": None}


def save_analysis(state: AnalysisState) -> AnalysisState:
    """Saves the analysis to analysis.md in the same directory."""
    analysis_output = state.get("analysis_output")
    file_path = state["file_path"]

    if not analysis_output:
        print(f"No analysis to save for {file_path}")
        return {}

    output_dir = os.path.dirname(file_path)
    output_path = os.path.join(output_dir, "analysis.md")

    try:
        with open(output_path, "w") as f:
            f.write(analysis_output)
        print(f"Saved analysis to {output_path}")
    except Exception as e:
        print(f"Error saving analysis to {output_path}: {e}")

    return {}


# Build the graph
builder = StateGraph(AnalysisState)

builder.add_node("load_report", load_report)
builder.add_node("generate_analysis", generate_analysis)
builder.add_node("save_analysis", save_analysis)

builder.set_entry_point("load_report")

builder.add_edge("load_report", "generate_analysis")
builder.add_edge("generate_analysis", "save_analysis")
builder.add_edge("save_analysis", END)

graph = builder.compile()


def main():
    base_dir = "/workspace/experiments/01162026_automatedtesting/outputs/phenotype_similarity/test"
    pattern = os.path.join(base_dir, "*", "report.json")
    report_files = glob.glob(pattern)

    print(f"Found {len(report_files)} report files.")

    for file_path in report_files:
        print(f"Processing {file_path}...")
        initial_state = {"file_path": file_path, "report_content": None, "analysis_output": None}
        graph.invoke(initial_state)


if __name__ == "__main__":
    main()
