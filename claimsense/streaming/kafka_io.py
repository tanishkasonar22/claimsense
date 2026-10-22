"""Kafka plumbing: topics, producer, and a worker loop with retries, a dead-letter topic
and idempotency. Needs: pip install confluent-kafka (and redis for multi-process idempotency).

Delivery is at-least-once: a worker may see the same event twice (for example if it crashes
after processing but before committing). Every handler is therefore idempotent: the
(claim_id, worker) pair is recorded after success and skipped if seen again.
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Callable

from ..config import get_settings

log = logging.getLogger("claimsense.kafka")

SUBMITTED = "claims.submitted"
SPEECH_DONE = "claims.speech.done"
DOCUMENTS_DONE = "claims.documents.done"
VISION_DONE = "claims.vision.done"
READY = "claims.ready_for_review"
DECIDED = "claims.decided"
DEAD_LETTER = "claims.dead_letter"
ALL_TOPICS = [SUBMITTED, SPEECH_DONE, DOCUMENTS_DONE, VISION_DONE, READY, DECIDED, DEAD_LETTER]
MAX_ATTEMPTS = 3


def _conf() -> dict:
    return {"bootstrap.servers": get_settings().kafka_bootstrap}


def create_topics(partitions: int = 4) -> None:
    from confluent_kafka.admin import AdminClient, NewTopic
    admin = AdminClient(_conf())
    existing = set(admin.list_topics(timeout=10).topics)
    new = [NewTopic(t, num_partitions=partitions, replication_factor=1) for t in ALL_TOPICS if t not in existing]
    for t, f in (admin.create_topics(new).items() if new else []):
        f.result()
        log.info("created topic %s", t)


_producer = None


def publish(topic: str, key: str, value: dict) -> None:
    global _producer
    from confluent_kafka import Producer
    if _producer is None:
        _producer = Producer(_conf() | {"enable.idempotence": True, "acks": "all"})
    _producer.produce(topic, key=key.encode(), value=json.dumps(value).encode())
    _producer.flush(10)


class Seen:
    """Idempotency store. Redis when REDIS_URL is set (shared by all worker processes)."""

    def __init__(self, namespace: str):
        self.ns = namespace
        url = os.environ.get("REDIS_URL")
        self.r = __import__("redis").from_url(url) if url else None
        self.local: set[str] = set()

    def done(self, key: str) -> bool:
        k = f"{self.ns}:{key}"
        return bool(self.r.exists(k)) if self.r else k in self.local

    def mark(self, key: str) -> None:
        k = f"{self.ns}:{key}"
        if self.r:
            self.r.set(k, 1, ex=7 * 24 * 3600)
        else:
            self.local.add(k)


def run_worker(group: str, topics: list[str], handler: Callable[[str, dict], tuple[str, dict] | None]) -> None:
    """Consume `topics`; for each event call handler(topic, event) -> (out_topic, out_event) or None."""
    from confluent_kafka import Consumer
    c = Consumer(_conf() | {"group.id": group, "enable.auto.commit": False, "auto.offset.reset": "earliest"})
    c.subscribe(topics)
    seen = Seen(group)
    log.info("worker %s listening on %s", group, topics)
    try:
        while True:
            msg = c.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                log.warning("kafka error: %s", msg.error())
                continue
            event = json.loads(msg.value())
            claim_id = event["claim_id"]
            key = f"{msg.topic()}:{claim_id}"
            if seen.done(key):
                c.commit(msg)                       # duplicate delivery: skip safely
                continue
            attempts = event.get("_attempts", 0) + 1
            try:
                t0 = time.perf_counter()
                out = handler(msg.topic(), event)
                if out:
                    publish(out[0], claim_id, out[1])
                seen.mark(key)
                log.info("%s processed %s in %.0f ms", group, claim_id, (time.perf_counter() - t0) * 1000)
            except Exception as e:
                log.exception("%s failed on %s (attempt %d)", group, claim_id, attempts)
                if attempts < MAX_ATTEMPTS:
                    publish(msg.topic(), claim_id, event | {"_attempts": attempts})
                else:
                    publish(DEAD_LETTER, claim_id, {"claim_id": claim_id, "worker": group, "error": str(e), "event": event})
            c.commit(msg)                           # commit only after handling (success, retry or DLQ)
    finally:
        c.close()
