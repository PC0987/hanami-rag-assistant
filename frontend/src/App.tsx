import { useEffect, useState } from "react";

const API = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

type Source = { source: string; text: string };
type Reply = { id: number; answer: string; sources: Source[]; latency_ms: number };
type Stats = {
  total_queries: number;
  avg_latency_ms: number;
  thumbs_up: number;
  thumbs_down: number;
  thumbs_down_rate: number;
  queries_per_day: { day: string; n: number }[];
  recent_thumbs_down: { id: number; question: string }[];
};

export default function App() {
  const [question, setQuestion] = useState("");
  const [reply, setReply] = useState<Reply | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [rated, setRated] = useState<number | null>(null);
  const [stats, setStats] = useState<Stats | null>(null);

  async function loadStats() {
    try {
      const res = await fetch(`${API}/stats`);
      if (res.ok) setStats(await res.json());
    } catch {
      /* the stats panel is optional, so ignore failures */
    }
  }

  useEffect(() => {
    loadStats();
  }, []);

  async function ask() {
    if (!question.trim() || loading) return;
    setLoading(true);
    setError("");
    setReply(null);
    setRated(null);
    try {
      const res = await fetch(`${API}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });
      if (!res.ok) throw new Error((await res.json()).detail ?? "Request failed");
      setReply(await res.json());
      loadStats();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Something went wrong. Try again.");
    } finally {
      setLoading(false);
    }
  }

  async function rate(value: 1 | -1) {
    if (!reply) return;
    await fetch(`${API}/feedback`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id: reply.id, value }),
    });
    setRated(value);
    loadStats();
  }

  return (
    <main className="page">
      <header>
        <h1>Hanami Cloud support</h1>
        <p className="muted">Ask about plans, billing, sharing or errors. English and Japanese both work.</p>
      </header>

      <div className="ask">
        <input
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && ask()}
          placeholder="How long are deleted files kept?"
          aria-label="Your question"
        />
        <button onClick={ask} disabled={loading || !question.trim()}>
          {loading ? "Searching docs" : "Ask"}
        </button>
      </div>

      {error && <p className="error" role="alert">{error}</p>}

      {reply && (
        <section className="answer">
          <p className="answer-text">{reply.answer}</p>
          <div className="rate">
            <button className={rated === 1 ? "on" : ""} onClick={() => rate(1)} aria-label="Helpful">Helpful</button>
            <button className={rated === -1 ? "on bad" : ""} onClick={() => rate(-1)} aria-label="Not helpful">Not helpful</button>
            <span className="muted">{reply.latency_ms} ms</span>
          </div>
          <h2>Sources</h2>
          <ol className="sources">
            {reply.sources.map((s, i) => (
              <li key={i}>
                <strong>{s.source}</strong>
                <p>{s.text}</p>
              </li>
            ))}
          </ol>
        </section>
      )}

      {stats && (
        <section className="stats">
          <h2>Usage so far</h2>
          <dl>
            <div><dt>Questions</dt><dd>{stats.total_queries}</dd></div>
            <div><dt>Avg latency</dt><dd>{stats.avg_latency_ms} ms</dd></div>
            <div><dt>Not helpful rate</dt><dd>{Math.round(stats.thumbs_down_rate * 100)}%</dd></div>
          </dl>
          {stats.queries_per_day.length > 0 && (
            <div className="bars" aria-label="Questions per day">
              {[...stats.queries_per_day].reverse().map((d) => (
                <div key={d.day} className="bar-row">
                  <span>{d.day}</span>
                  <div className="bar" style={{ width: `${Math.max(6, d.n * 24)}px` }} />
                  <span>{d.n}</span>
                </div>
              ))}
            </div>
          )}
          {stats.recent_thumbs_down.length > 0 && (
            <>
              <h3>Recent questions marked not helpful</h3>
              <ul>
                {stats.recent_thumbs_down.map((q) => (
                  <li key={q.id}>{q.question}</li>
                ))}
              </ul>
            </>
          )}
        </section>
      )}
    </main>
  );
}
