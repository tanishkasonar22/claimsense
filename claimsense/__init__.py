"""ClaimSense: a multimodal AI claims processing platform (portfolio project).

Package layout (one package, several runnable services):

    claimsense.api          FastAPI claims API (also runs the agent in-process in lite mode)
    claimsense.agent        LangGraph workflow with a human-approval interrupt
    claimsense.mcp_server   MCP server exposing the read-only tools
    claimsense.pipeline     The claim-processing steps as plain functions (no framework)
    claimsense.speech       Voice note -> transcript
    claimsense.documents    PDFs / photos of documents -> text -> fields
    claimsense.vision       Damage photos -> DamageReport
    claimsense.rag          Policy retrieval and coverage assessment
    claimsense.fraud        Fraud features and scoring
    claimsense.guardrails   PII redaction, prompt-injection detection, output checks
    claimsense.llm          One interface over Gemini, Ollama/vLLM/TGI and an offline rules baseline
    claimsense.storage      Database and file storage
    claimsense.streaming    Kafka workers (Tier 3)
"""

__version__ = "0.1.0"
