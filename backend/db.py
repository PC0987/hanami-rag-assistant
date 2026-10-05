"""SQLite storage: one row per query (with latency and tokens) plus thumbs up/down feedback."""
import os
import sqlite3
from contextlib import contextmanager

DB_PATH = os.getenv("DB_PATH", "app.db")


@contextmanager
def conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    try:
        yield c
        c.commit()
    finally:
        c.close()


def init_db():
    with conn() as c:
        c.execute(
            """CREATE TABLE IF NOT EXISTS queries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT DEFAULT CURRENT_TIMESTAMP,
                question TEXT NOT NULL,
                answer TEXT,
                sources TEXT,
                latency_ms INTEGER,
                input_tokens INTEGER,
                output_tokens INTEGER,
                feedback INTEGER
            )"""
        )


def log_query(question, answer, sources, latency_ms, input_tokens, output_tokens) -> int:
    with conn() as c:
        cur = c.execute(
            "INSERT INTO queries (question, answer, sources, latency_ms, input_tokens, output_tokens) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (question, answer, ",".join(sources), latency_ms, input_tokens, output_tokens),
        )
        return cur.lastrowid


def set_feedback(query_id: int, value: int) -> bool:
    with conn() as c:
        cur = c.execute("UPDATE queries SET feedback = ? WHERE id = ?", (value, query_id))
        return cur.rowcount > 0


def stats() -> dict:
    with conn() as c:
        total = c.execute("SELECT COUNT(*) FROM queries").fetchone()[0]
        avg_latency = c.execute("SELECT AVG(latency_ms) FROM queries").fetchone()[0]
        up = c.execute("SELECT COUNT(*) FROM queries WHERE feedback = 1").fetchone()[0]
        down = c.execute("SELECT COUNT(*) FROM queries WHERE feedback = -1").fetchone()[0]
        per_day = c.execute(
            "SELECT date(ts) AS day, COUNT(*) AS n FROM queries GROUP BY day ORDER BY day DESC LIMIT 7"
        ).fetchall()
        failed = c.execute(
            "SELECT id, question FROM queries WHERE feedback = -1 ORDER BY id DESC LIMIT 5"
        ).fetchall()
    rated = up + down
    return {
        "total_queries": total,
        "avg_latency_ms": round(avg_latency) if avg_latency else 0,
        "thumbs_up": up,
        "thumbs_down": down,
        "thumbs_down_rate": round(down / rated, 3) if rated else 0,
        "queries_per_day": [dict(r) for r in per_day],
        "recent_thumbs_down": [dict(r) for r in failed],
    }
