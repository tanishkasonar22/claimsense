// Typed client for the ClaimSense API. Types mirror claimsense/schemas.py.
// Tip (Phase 1.2): generate these automatically with `npx openapi-typescript http://localhost:8000/openapi.json -o lib/openapi.ts`.

export const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export type Action = "approve" | "deny" | "investigate" | "request_info";

export interface EvidenceRef { kind: string; ref: string; detail: string }
export interface Recommendation {
  action: Action; payout_estimate: number | null; summary: string; reasons: EvidenceRef[]; needs_human: boolean;
}
export interface FraudReason { feature: string; value: unknown; contribution: number; explanation: string }
export interface ClaimResult {
  claim_id: string;
  transcript?: { text: string; source: string } | null;
  facts?: Record<string, unknown> | null;
  damage?: { items: { part: string; damage_type: string; severity: string; confidence: number }[]; source: string } | null;
  coverage?: { covered: boolean | null; deductible: number | null; sections: string[]; reason: string } | null;
  fraud?: { score: number; model: string; reasons: FraudReason[] } | null;
  safety: { injection_suspected: boolean; injection_snippets: string[]; pii_redacted_count: number };
  consistency_issues: string[];
  recommendation?: Recommendation | null;
  documents: { path: string; doc_type: string; method: string }[];
}
export interface Claim {
  claim_id: string; status: string; customer_id: string; product_code: string;
  submission: { claim_amount?: number | null; documents: string[]; photos: string[] };
  result: ClaimResult | null; decision: { action: Action; adjuster: string; note: string } | null;
  error: string | null; created_at: string; updated_at: string;
  audit?: { event: string; actor: string; payload: unknown; at: string }[];
}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(`${API}${path}`, { cache: "no-store", ...init });
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json() as Promise<T>;
}

export const api = {
  claims: (status?: string) => req<Claim[]>(`/claims${status ? `?status=${status}` : ""}`),
  claim: (id: string) => req<Claim>(`/claims/${id}`),
  submitGold: (goldId: string) => req<{ claim_id: string }>(`/demo/gold/${goldId}`, { method: "POST" }),
  decide: (id: string, body: { action: Action; adjuster: string; note: string; edited_summary?: string }) =>
    req(`/claims/${id}/decision`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }),
  stats: () => req<{ claims: number; by_status: Record<string, number>; by_recommendation: Record<string, number>; adjuster_agreement: number | null }>("/stats"),
  fileUrl: (path: string) => `${API}/files?path=${encodeURIComponent(path)}`,
};
