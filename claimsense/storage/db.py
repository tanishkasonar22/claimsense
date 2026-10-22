"""Claims database (SQLAlchemy 2.0). SQLite in lite mode, PostgreSQL in full mode:
the same code, only DATABASE_URL changes.

Tables:
  claims        one row per claim: status, submission, pipeline result, adjuster decision
  audit_events  append-only log of everything that happened to a claim (who, what, when)
  preferences   (agent summary, adjuster-edited summary) pairs -> DPO training data (Phase 2.5)
"""
from __future__ import annotations

import datetime as dt
from functools import lru_cache

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String, Text, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from ..config import get_settings


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


class Base(DeclarativeBase):
    pass


class Claim(Base):
    __tablename__ = "claims"
    claim_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    status: Mapped[str] = mapped_column(String(32), index=True)
    customer_id: Mapped[str] = mapped_column(String(16), index=True)
    product_code: Mapped[str] = mapped_column(String(16))
    submission: Mapped[dict] = mapped_column(JSON)
    result: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    decision: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)


class AuditEvent(Base):
    __tablename__ = "audit_events"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    claim_id: Mapped[str] = mapped_column(ForeignKey("claims.claim_id"), index=True)
    event: Mapped[str] = mapped_column(String(64))
    actor: Mapped[str] = mapped_column(String(64), default="system")
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now)


class Preference(Base):
    __tablename__ = "preferences"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    claim_id: Mapped[str] = mapped_column(ForeignKey("claims.claim_id"))
    prompt: Mapped[str] = mapped_column(Text)
    chosen: Mapped[str] = mapped_column(Text)     # adjuster's edited summary
    rejected: Mapped[str] = mapped_column(Text)   # agent's original summary
    at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now)


class ClaimRepo:
    def __init__(self, url: str):
        kw = {"connect_args": {"check_same_thread": False}} if url.startswith("sqlite") else {"pool_pre_ping": True}
        self.engine = create_engine(url, **kw)
        Base.metadata.create_all(self.engine)   # use Alembic migrations once the schema settles

    def session(self) -> Session:
        return Session(self.engine, expire_on_commit=False)

    def create(self, submission: dict) -> Claim:
        with self.session() as s:
            c = Claim(claim_id=submission["claim_id"], status="submitted", customer_id=submission["customer_id"],
                      product_code=submission["product_code"], submission=submission)
            s.merge(c)
            s.add(AuditEvent(claim_id=c.claim_id, event="submitted", payload={"files": len(submission.get("documents", []))}))
            s.commit()
            return c

    def get(self, claim_id: str) -> Claim | None:
        with self.session() as s:
            return s.get(Claim, claim_id)

    def list(self, status: str | None = None, limit: int = 200) -> list[Claim]:
        with self.session() as s:
            q = select(Claim).order_by(Claim.updated_at.desc()).limit(limit)
            if status:
                q = q.where(Claim.status == status)
            return list(s.scalars(q))

    def update(self, claim_id: str, *, actor: str = "system", event: str | None = None,
               payload: dict | None = None, **fields) -> None:
        with self.session() as s:
            c = s.get(Claim, claim_id)
            if c is None:
                raise KeyError(claim_id)
            for k, v in fields.items():
                setattr(c, k, v)
            s.add(AuditEvent(claim_id=claim_id, event=event or f"status:{c.status}", actor=actor, payload=payload or {}))
            s.commit()

    def add_preference(self, claim_id: str, prompt: str, chosen: str, rejected: str) -> None:
        with self.session() as s:
            s.add(Preference(claim_id=claim_id, prompt=prompt, chosen=chosen, rejected=rejected))
            s.commit()

    def preferences(self) -> list[Preference]:
        with self.session() as s:
            return list(s.scalars(select(Preference)))

    def audit(self, claim_id: str) -> list[AuditEvent]:
        with self.session() as s:
            return list(s.scalars(select(AuditEvent).where(AuditEvent.claim_id == claim_id).order_by(AuditEvent.id)))


@lru_cache
def get_repo() -> ClaimRepo:
    return ClaimRepo(get_settings().database_url)
