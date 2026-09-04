"""
store.py - THE unified database.

Three sources -- Human, Agentic AI, and the base Transformer -- each
write through a "Memory Store" (MS) into ONE shared database rather than
three silos. That accumulated log *is* the embedding: not a separate
step, the record itself.

Backed by SQLite. By default main.py points this at a local file
(companion.db) so your profile, facts, and event log survive between
runs instead of vanishing every time the process exits -- pass
db_path=":memory:" if you want a throwaway session instead.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from privacy import contains_pii
from state import SemanticFact
from belief import BetaBelief

SOURCE_HUMAN = "human"
SOURCE_AGENT = "agentic_ai"
SOURCE_TRANSFORMER = "transformer"
VALID_SOURCES = {SOURCE_HUMAN, SOURCE_AGENT, SOURCE_TRANSFORMER}


def _bit_repr(payload: str, length: int = 32) -> str:
    """Cosmetic bit-level encoding -- mirrors the notebook's '01101101'
    motif. This is a *display* representation, not the actual storage
    format; real payloads are stored as JSON for correctness."""
    h = abs(hash(payload))
    bits = bin(h)[2:].zfill(length)[:length]
    return " ".join(bits[i:i + 8] for i in range(0, length, 8))


@dataclass
class EpisodicEvent:
    id: str
    user_id: str
    source: str                 # human | agentic_ai | transformer
    event_type: str              # e.g. "input", "advice", "feedback", "correction"
    payload: Dict[str, Any]
    ts: float = field(default_factory=time.time)
    pii: bool = False
    bits: str = ""
    domain: str = "general"
    conversation_id: str = "legacy"  # groups turns into one chat thread for the sidebar

    @classmethod
    def new(
        cls, user_id: str, source: str, event_type: str, payload: Dict[str, Any],
        domain: str = "general", conversation_id: str = "legacy",
    ) -> "EpisodicEvent":
        assert source in VALID_SOURCES, f"invalid source: {source}"
        raw = json.dumps(payload, sort_keys=True)
        return cls(
            id=str(uuid.uuid4()),
            user_id=user_id,
            source=source,
            event_type=event_type,
            payload=payload,
            pii=contains_pii(raw),
            bits=_bit_repr(raw),
            domain=domain,
            conversation_id=conversation_id,
        )


class UnifiedMemoryStore:
    """The single shared database fed by all three Memory Stores."""

    def __init__(self, path: str = ":memory:") -> None:
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS episodic_log (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                source TEXT NOT NULL,
                event_type TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                bits TEXT NOT NULL,
                pii INTEGER NOT NULL DEFAULT 0,
                ts REAL NOT NULL,
                consumed INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS semantic_profile (
                user_id TEXT NOT NULL,
                key TEXT NOT NULL,
                fact_json TEXT NOT NULL,
                updated_at REAL NOT NULL,
                PRIMARY KEY (user_id, key)
            );

            CREATE INDEX IF NOT EXISTS idx_episodic_user ON episodic_log(user_id, ts);
            """
        )
        self._migrate_episodic_log_columns()
        self._conn.commit()

    def _migrate_episodic_log_columns(self) -> None:
        """Older databases predate the domain/conversation_id columns --
        add them in place rather than requiring a fresh db. Existing rows
        get domain='general', conversation_id='legacy' (the best label
        available in hindsight, since the old schema never recorded
        either) so they still surface as one browsable past chat instead
        of vanishing."""
        existing = {row["name"] for row in self._conn.execute("PRAGMA table_info(episodic_log)")}
        if "domain" not in existing:
            self._conn.execute("ALTER TABLE episodic_log ADD COLUMN domain TEXT NOT NULL DEFAULT 'general'")
        if "conversation_id" not in existing:
            self._conn.execute("ALTER TABLE episodic_log ADD COLUMN conversation_id TEXT NOT NULL DEFAULT 'legacy'")
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_episodic_conversation "
            "ON episodic_log(user_id, domain, conversation_id, ts)"
        )

    # ---------- episodic (raw, source-tagged log) ----------

    def write_episodic(self, event: EpisodicEvent) -> None:
        """Idempotent write: retried writes never duplicate an event."""
        self._conn.execute(
            "INSERT OR IGNORE INTO episodic_log "
            "(id, user_id, source, event_type, payload_json, bits, pii, ts, domain, conversation_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                event.id, event.user_id, event.source, event.event_type,
                json.dumps(event.payload), event.bits, int(event.pii), event.ts,
                event.domain, event.conversation_id,
            ),
        )
        self._conn.commit()

    def read_episodic(
        self, user_id: str, limit: int = 50, unconsumed_only: bool = False,
        domain: Optional[str] = None, conversation_id: Optional[str] = None,
    ) -> List[EpisodicEvent]:
        query = "SELECT * FROM episodic_log WHERE user_id = ?"
        params: List[Any] = [user_id]
        if domain is not None:
            query += " AND domain = ?"
            params.append(domain)
        if conversation_id is not None:
            query += " AND conversation_id = ?"
            params.append(conversation_id)
        if unconsumed_only:
            query += " AND consumed = 0"
        query += " ORDER BY ts DESC LIMIT ?"
        params.append(limit)
        rows = self._conn.execute(query, params).fetchall()
        return [
            EpisodicEvent(
                id=r["id"], user_id=r["user_id"], source=r["source"],
                event_type=r["event_type"], payload=json.loads(r["payload_json"]),
                ts=r["ts"], pii=bool(r["pii"]), bits=r["bits"],
                domain=r["domain"], conversation_id=r["conversation_id"],
            )
            for r in rows
        ]

    def list_conversations(self, user_id: str, domain: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Groups this user+domain's episodic log by conversation_id --
        the unit the sidebar's chat list shows. Title is the
        conversation's first human message (truncated); a conversation
        with no human input at all is dropped (nothing to show or click
        into)."""
        rows = self._conn.execute(
            "SELECT conversation_id, source, event_type, payload_json, ts "
            "FROM episodic_log WHERE user_id = ? AND domain = ? ORDER BY ts ASC",
            (user_id, domain),
        ).fetchall()
        conversations: Dict[str, Dict[str, Any]] = {}
        for r in rows:
            cid = r["conversation_id"]
            conv = conversations.setdefault(
                cid, {"conversation_id": cid, "title": None, "last_ts": 0.0, "turns": 0}
            )
            conv["last_ts"] = max(conv["last_ts"], r["ts"])
            if r["source"] == SOURCE_HUMAN and r["event_type"] == "input":
                conv["turns"] += 1
                if conv["title"] is None:
                    text = json.loads(r["payload_json"]).get("text", "").strip()
                    conv["title"] = (text[:48] + "...") if len(text) > 48 else (text or "New chat")
        result = [c for c in conversations.values() if c["turns"] > 0]
        result.sort(key=lambda c: c["last_ts"], reverse=True)
        return result[:limit]

    def mark_consumed(self, event_ids: List[str]) -> None:
        if not event_ids:
            return
        placeholders = ",".join("?" for _ in event_ids)
        self._conn.execute(
            f"UPDATE episodic_log SET consumed = 1 WHERE id IN ({placeholders})", event_ids
        )
        self._conn.commit()

    def live_feed_tail(self, user_id: str, n: int = 8) -> List[EpisodicEvent]:
        return self.read_episodic(user_id, limit=n)

    # ---------- semantic (distilled, editable Internal State) ----------

    def upsert_semantic(self, user_id: str, fact: SemanticFact) -> None:
        d = fact.as_dict()
        d["value"] = fact.value
        if fact.belief is not None:
            d["belief"] = {"alpha": fact.belief.alpha, "beta": fact.belief.beta}
        self._conn.execute(
            "INSERT INTO semantic_profile (user_id, key, fact_json, updated_at) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(user_id, key) DO UPDATE SET fact_json = excluded.fact_json, "
            "updated_at = excluded.updated_at",
            (user_id, fact.key, json.dumps(d), fact.updated_at),
        )
        self._conn.commit()

    def read_semantic(self, user_id: str) -> Dict[str, SemanticFact]:
        rows = self._conn.execute(
            "SELECT key, fact_json FROM semantic_profile WHERE user_id = ?", (user_id,)
        ).fetchall()
        facts = {}
        for r in rows:
            d = json.loads(r["fact_json"])
            belief: Optional[BetaBelief] = None
            if "belief" in d:
                belief = BetaBelief(alpha=d["belief"]["alpha"], beta=d["belief"]["beta"])
            facts[r["key"]] = SemanticFact(
                id=d["id"], key=d["key"], label=d["label"], value=d["value"],
                confidence=d["confidence"], updated_at=d["updated_at"],
                status=d.get("status", "active"), pii=d.get("pii", False),
                belief=belief,
                pending_value=d.get("pending_value"), pending_label=d.get("pending_label"),
            )
        return facts

    def delete_semantic(self, user_id: str, key: str) -> bool:
        """User-initiated forget -- part of the privacy/trust layer."""
        cur = self._conn.execute(
            "DELETE FROM semantic_profile WHERE user_id = ? AND key = ?", (user_id, key)
        )
        self._conn.commit()
        return cur.rowcount > 0

    def close(self) -> None:
        self._conn.close()
