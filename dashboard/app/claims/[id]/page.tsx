"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useEffect, useState } from "react";
import { api, type Action, type Claim } from "@/lib/api";

export default function ClaimPage() {
  const { id } = useParams<{ id: string }>();
  const [claim, setClaim] = useState<Claim | null>(null);
  const [note, setNote] = useState("");
  const [summary, setSummary] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const load = () => api.claim(id).then((c) => {
      setClaim(c);
      setSummary((s) => s || c.result?.recommendation?.summary || "");
    }).catch((e) => setError(String(e)));
    load();
    const t = setInterval(load, 3000);
    return () => clearInterval(t);
  }, [id]);

  async function decide(action: Action) {
    setBusy(true); setError(null);
    try {
      await api.decide(id, { action, adjuster: "adjuster-1", note, edited_summary: summary });
      setClaim(await api.claim(id));
    } catch (e) { setError(String(e)); } finally { setBusy(false); }
  }

  if (!claim) return <p className="muted">Loading…</p>;
  const r = claim.result;
  const rec = r?.recommendation;

  return (
    <>
      <p><Link href="/">← All claims</Link></p>
      <h1 style={{ marginTop: 0 }}>{claim.claim_id} <span className="muted" style={{ fontSize: 16 }}>{claim.status}</span></h1>
      {error && <div className="panel warn">{error}</div>}
      {claim.error && <div className="panel warn">Processing failed: {claim.error}</div>}
      {r?.safety.injection_suspected && (
        <div className="panel warn">
          <strong>Possible prompt injection in the claim content.</strong> Approval is blocked until you review it.
          <ul className="evidence">{r.safety.injection_snippets.map((s, i) => <li key={i}><code>{s}</code></li>)}</ul>
        </div>
      )}

      {rec && (
        <div className="panel">
          <div className="row"><h2 style={{ margin: 0 }}>AI recommendation</h2><span className={`badge ${rec.action}`}>{rec.action}</span>
            {rec.payout_estimate != null && <span>estimated payout <strong>${rec.payout_estimate.toLocaleString()}</strong></span>}</div>
          <p className="muted" style={{ marginBottom: 4 }}>Summary (edit it if it's wrong or unclear; your edits improve the model)</p>
          <textarea rows={4} value={summary} onChange={(e) => setSummary(e.target.value)} disabled={claim.status !== "ready_for_review"} />
          <h3>Evidence</h3>
          <ul className="evidence">{rec.reasons.map((x, i) => <li key={i}><strong>[{x.kind}] {x.ref}</strong>: {x.detail}</li>)}</ul>
          {claim.status === "ready_for_review" ? (
            <>
              <input placeholder="Note for the audit log" value={note} onChange={(e) => setNote(e.target.value)} />
              <div className="row" style={{ marginTop: 10 }}>
                <button className="primary" disabled={busy || r?.safety.injection_suspected} onClick={() => decide("approve")}>Approve</button>
                <button disabled={busy} onClick={() => decide("deny")}>Deny</button>
                <button disabled={busy} onClick={() => decide("investigate")}>Send to investigation</button>
                <button disabled={busy} onClick={() => decide("request_info")}>Request info</button>
              </div>
            </>
          ) : claim.decision && (
            <p>Decided by {claim.decision.adjuster}: <span className={`badge ${claim.decision.action}`}>{claim.decision.action}</span> {claim.decision.note}</p>
          )}
        </div>
      )}

      <div className="grid">
        <div className="panel">
          <h3>Coverage</h3>
          {r?.coverage ? <p>{r.coverage.covered === null ? "Unclear" : r.coverage.covered ? "Covered" : "Not covered"} · {r.coverage.reason}<br />
            Deductible: {r.coverage.deductible ?? "n/a"} · Sections: {r.coverage.sections.join(", ") || "none"}</p> : <p className="muted">…</p>}
          <h3>Fraud</h3>
          {r?.fraud ? <><p>Score <strong>{r.fraud.score.toFixed(2)}</strong> <span className="muted">({r.fraud.model})</span></p>
            <ul className="evidence">{r.fraud.reasons.map((x) => <li key={x.feature}>{x.explanation}: {String(x.value)}</li>)}</ul></> : <p className="muted">…</p>}
          {r && r.consistency_issues.length > 0 && <><h3>Consistency issues</h3>
            <ul className="evidence">{r.consistency_issues.map((x, i) => <li key={i}>{x}</li>)}</ul></>}
        </div>
        <div className="panel">
          <h3>Voice note <span className="muted">({r?.transcript?.source})</span></h3>
          <p style={{ whiteSpace: "pre-wrap" }}>{r?.transcript?.text}</p>
          <h3>Damage <span className="muted">({r?.damage?.source})</span></h3>
          <ul className="evidence">{r?.damage?.items.map((d, i) => <li key={i}>{d.damage_type} on {d.part.replaceAll("_", " ")} ({d.severity})</li>)}</ul>
          <h3>Documents</h3>
          <ul className="evidence">{claim.submission.documents.map((p) => <li key={p}><a href={api.fileUrl(p)} target="_blank">{p.split("/").pop()}</a></li>)}</ul>
        </div>
      </div>

      {claim.audit && (
        <div className="panel">
          <h3>Audit trail</h3>
          <table><tbody>{claim.audit.map((a, i) => <tr key={i}><td>{new Date(a.at).toLocaleString()}</td><td>{a.actor}</td><td>{a.event}</td></tr>)}</tbody></table>
        </div>
      )}
    </>
  );
}
