"""End-to-end LLM gateway over LiteLLM (S2).

Public API:
- Gateway — the single point for `completion` + `embedding`, config-agnostic (values via
  the constructor). See gateway.py and §5 Rag_pipeline_architecture.md.
"""
from llm_gateway.gateway import Gateway

__all__ = ["Gateway"]
