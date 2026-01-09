from langgraph.graph import StateGraph, END
from .state import ResumeState
from .nodes import (
    extract_keywords_node, 
    filter_keywords_node,
    rewrite_resume_node, 
    validate_content_node,
    compile_node, 
    score_node
)

def should_continue_compilation(state: ResumeState):
    """Determine if we should continue fixing LaTeX or move to validation."""
    if state.get('compilation_error') is None:
        return "validate"
    
    if state.get('iteration_count', 0) >= 3:
        return "validate"
        
    return "rewrite"

def should_continue_tailoring(state: ResumeState):
    """Determine if we should rewrite due to content validation failure."""
    if state.get('validation_reason') is None:
        return "score"
    
    if state.get('iteration_count', 0) >= 5: # Total cap for both syntax and content
        return "score"
        
    return "rewrite"

def create_resume_agent():
    workflow = StateGraph(ResumeState)

    # Add Nodes
    workflow.add_node("extract_keywords", extract_keywords_node)
    workflow.add_node("filter_keywords", filter_keywords_node)
    workflow.add_node("rewrite", rewrite_resume_node)
    workflow.add_node("validate", validate_content_node)
    workflow.add_node("compile", compile_node)
    workflow.add_node("score", score_node)

    # Set Entry Point
    workflow.set_entry_point("extract_keywords")

    # Connect Nodes
    workflow.add_edge("extract_keywords", "filter_keywords")
    workflow.add_edge("filter_keywords", "rewrite")
    workflow.add_edge("rewrite", "compile")

    # Conditional Edges for Compilation
    workflow.add_conditional_edges(
        "compile",
        should_continue_compilation,
        {
            "rewrite": "rewrite",
            "validate": "validate"
        }
    )
    
    # Conditional Edges for Content Validation
    workflow.add_conditional_edges(
        "validate",
        should_continue_tailoring,
        {
            "rewrite": "rewrite",
            "score": "score"
        }
    )

    workflow.add_edge("score", END)

    return workflow.compile()
