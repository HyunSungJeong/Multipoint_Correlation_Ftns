"""Reclassify the stored Step-1 summary of notebooks/vertex_krylov.ipynb under the
task-sheet amendment recorded in vertex_krylov_STATUS.md (A1-A3), without recomputation.

A1: C4 of the unstabilized particle chain (projector 1 - vv†) is "report"; for exact
    data the guaranteed C4 is that of the stabilized chain ([VD] Prop. 13), tol 1e-9
    (V2a; 1e-11 for the atom, V1).
A3: C2 (Σ^S, spectra, first-block weights, S = N̄², start block) is guaranteed at 1e-10.

Usage: python3 notebooks/reclassify_step1_summary.py [notebook]   (prints counts and changed rows)
"""
import json, re, sys
path = sys.argv[1] if len(sys.argv) > 1 else "notebooks/vertex_krylov.ipynb"
nb = json.load(open(path))
text = ""
for c in nb["cells"]:
    for o in c.get("outputs", []):
        if o.get("output_type") == "stream":
            text += "".join(o["text"])
rows = []
pat = re.compile(r"^(V1|V2a|V2b|V3a|V3b)\s+(.{26})\s(.{7})\s(.{30})\s+(\S+)\s+(\S+)\s+(\S+)$")
for line in text.splitlines():
    m = pat.match(line)
    if m:
        ph, case, chan, check, val, tol, st = (g.strip() for g in m.groups())
        rows.append(dict(phase=ph, case=case, chan=chan, check=check, value=float(val), tol=tol, status=st))
changed = []
for r in rows:
    old = r["status"]
    if r["check"] == "C4" and r["phase"] in ("V1", "V2a"):
        r["check"] = "C4 (unstabilized, diagnostic)"; r["tol"] = "—"; r["status"] = "report"
    elif r["check"] == "C4 (BigFloat chain)":          # Step 1 ran the unstabilized chain in BigFloat
        r["check"] = "C4 (BigFloat, unstabilized)"; r["tol"] = "—"; r["status"] = "report"
    elif r["check"] == "C4 (stabilized)":
        pass                                          # guaranteed (A1), unchanged
    elif r["check"] == "C2":
        r["tol"] = "1e-10"
        r["status"] = "pass" if r["value"] <= 1e-10 else "FAIL"
    if r["status"] != old:
        changed.append((r, old))
n = len(rows)
cnt = {s: sum(r["status"] == s for r in rows) for s in ("pass", "FAIL", "report")}
print(f"{n} rows: {cnt['pass']} pass, {cnt['FAIL']} FAIL, {cnt['report']} report")
print(f"max C2: {max(r['value'] for r in rows if r['check'] == 'C2'):.2e}; "
      f"max C4 (stabilized): {max(r['value'] for r in rows if r['check'] == 'C4 (stabilized)'):.2e}")
for r, old in changed:
    print(f"  {r['phase']:4s} {r['case']:26s} {r['chan']:7s} {r['check']:30s} {r['value']:.2e}  {old} -> {r['status']}")
