from langgraph.graph import StateGraph, END
from .state import ResumeState
from .nodes import (
    sanitize_jd_node,
    extract_keywords_node, 
    filter_keywords_node,
    rewrite_resume_node, 
    validate_content_node,
    compile_node, 
    score_node
)

def should_continue_compilation(state: ResumeState):
    """Phase 5: Compile LaTeX. If error -> retry rewrite. If pass -> END."""
    if state.get('compilation_error') is None:
        return END
    
    if state.get('iteration_count', 0) >= 7:
        return END
        
    return "rewrite"

def should_continue_tailoring(state: ResumeState):
    """Determine if we should rewrite due to content validation failure."""
    if state.get('validation_reason') is None:
        return "continue"
    
    if state.get('iteration_count', 0) >= 5:
        return "continue"
        
    return "rewrite"

def create_resume_agent():
    workflow = StateGraph(ResumeState)

    # Add Nodes
    workflow.add_node("sanitize_jd", sanitize_jd_node)
    workflow.add_node("extract_keywords", extract_keywords_node)
    workflow.add_node("filter_keywords", filter_keywords_node)
    workflow.add_node("rewrite", rewrite_resume_node)
    workflow.add_node("validate", validate_content_node)
    workflow.add_node("compile", compile_node)
    workflow.add_node("score", score_node)

    # Set Entry Point
    workflow.set_entry_point("sanitize_jd")

    # Connect Nodes
    workflow.add_edge("sanitize_jd", "extract_keywords")
    workflow.add_edge("extract_keywords", "filter_keywords")
    workflow.add_edge("filter_keywords", "rewrite")
    workflow.add_edge("rewrite", "validate")

    # Conditional Edges for Content Validation (Phase 3)
    workflow.add_conditional_edges(
        "validate",
        should_continue_tailoring,
        {
            "rewrite": "rewrite",
            "continue": "score"
        }
    )
    
    # Conditional Edges for Factual Verification (Phase 4)
    workflow.add_conditional_edges(
        "score",
        should_continue_tailoring,
        {
            "rewrite": "rewrite",
            "continue": "compile"
        }
    )

    # Conditional Edges for Compilation (Phase 5)
    workflow.add_conditional_edges(
        "compile",
        should_continue_compilation,
        {
            "rewrite": "rewrite",
            "__end__": END
        }
    )

    return workflow.compile()
