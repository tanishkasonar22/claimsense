"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";
import { api, type Claim } from "@/lib/api";

const STATUSES = ["", "ready_for_review", "processing", "decided", "failed"];

export default function ClaimsPage() {
  const [claims, setClaims] = useState<Claim[]>([]);
  const [status, setStatus] = useState("");
  const [goldId, setGoldId] = useState("G000005");
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(() => {
    api.claims(status || undefined).then(setClaims).catch((e) => setError(String(e)));
  }, [status]);

  useEffect(() => {
    load();
    const t = setInterval(load, 3000); // simple polling; claims move through statuses in the background
    return () => clearInterval(t);
  }, [load]);

  async function submitSample() {
    setError(null);
    try { await api.submitGold(goldId); load(); } catch (e) { setError(String(e)); }
  }

  return (
    <>
      <div className="panel row">
        <strong>Load a sample claim</strong>
        <input style={{ width: 140 }} value={goldId} onChange={(e) => setGoldId(e.target.value)} />
        <button className="primary" onClick={submitSample}>Submit</button>
        <span className="muted">Gold IDs run from G000000 to G000199.</span>
      </div>
      {error && <div className="panel warn">{error}</div>}
      <div className="panel">
        <div className="row" style={{ justifyContent: "space-between", marginBottom: 8 }}>
          <h2 style={{ margin: 0 }}>Claims</h2>
          <select style={{ width: 200 }} value={status} onChange={(e) => setStatus(e.target.value)}>
            {STATUSES.map((s) => <option key={s} value={s}>{s || "all statuses"}</option>)}
          </select>
        </div>
        <table>
          <thead><tr><th>Claim</th><th>Status</th><th>Policy</th><th>AI recommendation</th><th>Fraud score</th><th>Decision</th></tr></thead>
          <tbody>
            {claims.map((c) => {
              const rec = c.result?.recommendation;
              return (
                <tr key={c.claim_id}>
                  <td><Link href={`/claims/${c.claim_id}`}>{c.claim_id}</Link></td>
                  <td>{c.status}</td>
                  <td>{c.product_code}</td>
                  <td>{rec ? <span className={`badge ${rec.action}`}>{rec.action}</span> : "…"}</td>
                  <td>{c.result?.fraud ? c.result.fraud.score.toFixed(2) : "…"}</td>
                  <td>{c.decision ? <span className={`badge ${c.decision.action}`}>{c.decision.action}</span> : ""}</td>
                </tr>
              );
            })}
            {claims.length === 0 && <tr><td colSpan={6} className="muted">No claims yet. Submit a sample above.</td></tr>}
          </tbody>
        </table>
      </div>
    </>
  );
}
