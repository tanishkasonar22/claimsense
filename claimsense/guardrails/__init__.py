"""Safety layers (defense in depth):

  1. pii.redact            mask personal data before anything leaves for an external LLM
  2. injection.scan        flag text that tries to instruct the AI (in narratives, voice, documents)
  3. output.enforce        the agent may only RECOMMEND; it can never approve or pay
  4. tools are read-only   (see claimsense/tools.py) - no tool can change data
"""
