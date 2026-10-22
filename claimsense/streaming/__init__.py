"""Event-driven processing with Kafka (booklet Phase 3.1, study guide E2).

Set PROCESSING=kafka and the API publishes each new claim to `claims.submitted` instead of
running the agent in-process. Then run the workers (each is its own process / container):

    python -m claimsense.streaming.run speech
    python -m claimsense.streaming.run documents
    python -m claimsense.streaming.run vision          # scale this one: run it 4 times
    python -m claimsense.streaming.run orchestrator

    claims.submitted ─┬─ speech    ─ claims.speech.done    ─┐
                      ├─ documents ─ claims.documents.done ─┼─ orchestrator ─ claims.ready_for_review
                      └─ vision    ─ claims.vision.done    ─┘
"""
