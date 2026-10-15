#!/usr/bin/env python3
"""
Build a mini glycan/protein biomarker knowledge graph.

Inputs
  --biomarkers   biomarkers_llm_glycan.tsv
  --masterlist   human_protein_masterlist_matched.csv
  --doid-obo     doid.obo (optional; downloaded from the Disease Ontology GitHub if omitted)

Outputs (in --outdir)
  nodes.csv          one row per node   (node_id, node_label, name, synonyms, <typed properties>)
  relationships.csv  one row per edge   (source_id, target_id, relationship_type, <edge properties>)
  schema.yaml        node labels, relationship types and their properties
  build_report.txt   counts, removed duplicates and other QC notes

Multi-valued properties are pipe-delimited ("|").
"""
import argparse
import re
import sys
import urllib.request
from collections import defaultdict, Counter
from pathlib import Path

import pandas as pd
import yaml

DOID_URL = ("https://raw.githubusercontent.com/DiseaseOntology/"
            "HumanDiseaseOntology/main/src/ontology/doid.obo")
SEP = "|"

# --------------------------------------------------------------------------- #
# Schema: the single source of truth for node/edge labels and properties.
# schema.yaml is written from this, and nodes/relationships are checked against it.
# --------------------------------------------------------------------------- #
SCHEMA = {
    "nodes": {
        "Biomarker": {
            "id": "biomarker_id (e.g. EM6323-2)",
            "description": "A measurable change in one glycan or a glycan-protein pair, independent of context.",
            "properties": {
                "name": "string  - most frequent biomarker text",
                "synonyms": "list    - other biomarker texts used for this biomarker_id",
                "controlled_vocab": "list    - biomarker_controlled_vocab values",
                "change_type": "string  - increased | decreased | altered | presence of | absence of",
                "aspect_type": "string  - level | not_specified",
                "composition": "string  - glycan | glycan+protein",
            },
        },
        "BiomarkerAssertion": {
            "id": "BMA:<biomarker_index> (e.g. BMA:21)",
            "description": "One reported use of a biomarker in a condition/specimen/publication context (one per biomarker_index).",
            "properties": {
                "name": "string  - biomarker text as written for this assertion",
                "biomarker_index": "int",
                "best_biomarker_role": "string  - diagnostic | prognostic | predictive | response | monitoring | susceptibility",
            },
        },
        "Glycan": {
            "id": "bGSL:<uuid>",
            "description": "Glycan identified by the in-house bGSL identifier.",
            "properties": {"name": "string", "synonyms": "list"},
        },
        "Protein": {
            "id": "bare UniProtKB accession (e.g. P06731)",
            "description": "Protein; UPKB: prefix and isoform suffix removed.",
            "properties": {
                "name": "string",
                "synonyms": "list",
                "uniprot_canonical_isoform": "string  - canonical isoform from masterlist, if present",
                "uniprot_status": "string  - reviewed | (blank if not in masterlist)",
                "in_masterlist": "bool",
            },
        },
        "Gene": {
            "id": "GENE:<symbol> (e.g. GENE:CEACAM5)",
            "description": "Gene symbol from the protein masterlist.",
            "properties": {"name": "string  - gene symbol"},
        },
        "Isoform": {
            "id": "UniProtKB isoform accession (e.g. P06731-2, M0QX98-1)",
            "description": "Reviewed or unreviewed isoform listed in the masterlist.",
            "properties": {"name": "string", "isoform_status": "string  - reviewed | unreviewed"},
        },
        "Condition": {
            "id": "DOID:<n>",
            "description": "Disease Ontology term; includes DOID ancestors so related conditions can be grouped.",
            "properties": {
                "name": "string  - official DOID name (falls back to dataset label)",
                "synonyms": "list    - labels used in the dataset",
                "in_dataset": "bool    - true if directly referenced by a biomarker, false if added as an ancestor",
                "obsolete": "bool",
            },
        },
        "TissueType": {
            "id": "UBERON:<n>",
            "description": "Anatomical specimen (fluid or tissue).",
            "properties": {"name": "string", "synonyms": "list"},
        },
        "CellType": {
            "id": "CL:<n>",
            "description": "Cell-type specimen.",
            "properties": {"name": "string", "synonyms": "list"},
        },
        "Publication": {
            "id": "PubMed:<pmid>",
            "description": "Supporting publication.",
            "properties": {"name": "string  - PMID"},
        },
    },
    "relationships": {
        "HAS_COMPONENT": {
            "source": "Biomarker", "target": ["Glycan", "Protein"],
            "properties": {"entity_index": "int", "entity_type": "string"},
        },
        "GLYCAN_ON_PROTEIN": {
            "source": "Glycan", "target": "Protein",
            "description": "Derived from glycan+protein biomarker pairs (glycoform). One edge per glycan-protein pair.",
            "properties": {
                "biomarker_ids": "list",
                "condition_ids": "list",
                "n_assertions": "int",
            },
        },
        "ASSERTS": {"source": "BiomarkerAssertion", "target": "Biomarker", "properties": {}},
        "ASSOCIATED_WITH": {
            "source": "BiomarkerAssertion", "target": "Condition",
            "properties": {
                "best_biomarker_role": "string",
                "change_type": "string",
                "condition_label": "string  - label as written in the source row",
            },
        },
        "MEASURED_IN": {
            "source": "BiomarkerAssertion", "target": ["TissueType", "CellType"],
            "properties": {"specimen_label": "string  - label as written in the source row"},
        },
        "SUPPORTED_BY": {
            "source": "BiomarkerAssertion", "target": "Publication",
            "properties": {"evidence": "string  - evidence sentences from the source row"},
        },
        "ENCODED_BY": {"source": "Protein", "target": "Gene", "properties": {}},
        "HAS_ISOFORM": {
            "source": "Protein", "target": "Isoform",
            "properties": {"isoform_status": "string", "is_canonical": "bool"},
        },
        "IS_A": {
            "source": "Condition", "target": "Condition",
            "description": "Disease Ontology is_a hierarchy (child -> parent).",
            "properties": {},
        },
    },
    "conventions": {
        "multi_value_delimiter": SEP,
        "node_key": "node_id (ontology/database identifier; labels are never used as keys)",
    },
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def bare_uniprot(acc: str) -> str:
    acc = re.sub(r"^(UPKB|UniProtKB|uniprot):", "", str(acc).strip(), flags=re.I)
    return acc.split("-")[0]


def most_common(values):
    vals = [v for v in values if isinstance(v, str) and v.strip()]
    return Counter(vals).most_common(1)[0][0] if vals else ""


def join(values):
    seen = []
    for v in values:
        if isinstance(v, str) and v.strip() and v not in seen:
            seen.append(v)
    return SEP.join(seen)


def parse_obo(path: Path):
    """Minimal OBO parser: returns {id: {name, is_a[], obsolete, replaced_by}}."""
    terms, cur, in_term = {}, None, False
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("["):
            if cur and in_term:
                terms[cur["id"]] = cur
            in_term = line == "[Term]"
            cur = {"id": None, "name": "", "is_a": [], "obsolete": False, "replaced_by": None} if in_term else None
            continue
        if not in_term or not line or ":" not in line:
            continue
        tag, val = line.split(":", 1)
        val = val.strip()
        if tag == "id":
            cur["id"] = val
        elif tag == "name":
            cur["name"] = val
        elif tag == "is_a":
            cur["is_a"].append(val.split("!")[0].strip())
        elif tag == "is_obsolete" and val == "true":
            cur["obsolete"] = True
        elif tag == "replaced_by":
            cur["replaced_by"] = val
    if cur and in_term:
        terms[cur["id"]] = cur
    return terms


def load_doid(path_arg, outdir, report):
    if path_arg:
        p = Path(path_arg)
    else:
        p = outdir / "doid.obo"
        if not p.exists():
            try:
                print(f"Downloading Disease Ontology from {DOID_URL} ...")
                urllib.request.urlretrieve(DOID_URL, p)
            except Exception as e:  # noqa: BLE001
                report.append(f"WARNING: could not download doid.obo ({e}); condition hierarchy skipped.")
                return {}
    terms = parse_obo(p)
    version = next((l.split(":", 1)[1].strip() for l in p.open(encoding="utf-8")
                    if l.startswith("data-version")), "unknown")
    report.append(f"Disease Ontology: {len(terms)} terms loaded from {p} (data-version {version})")
    return terms


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #
def build(args):
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    report = []

    bm = pd.read_csv(args.biomarkers, sep="\t", dtype=str).fillna("")
    ml = pd.read_csv(args.masterlist, dtype=str).fillna("")
    for df in (bm, ml):
        df.columns = [c.strip() for c in df.columns]
    bm["biomarker_index"] = bm["biomarker_index"].astype(int)
    bm["entity_index"] = bm["entity_index"].astype(int)
    report.append(f"Input rows: {len(bm)} biomarker rows, {len(ml)} masterlist rows")

    # ---- Remove duplicated assertions (whole biomarker_index blocks identical apart from the index)
    content_cols = [c for c in bm.columns if c != "biomarker_index"]
    seen, drop = {}, []
    for idx, g in bm.sort_values(["biomarker_index", "entity_index"]).groupby("biomarker_index"):
        sig = tuple(map(tuple, g[content_cols].values))
        if sig in seen:
            drop.append(idx)
            report.append(f"Removed duplicate biomarker_index {idx} (identical to {seen[sig]})")
        else:
            seen[sig] = idx
    bm = bm[~bm["biomarker_index"].isin(drop)].copy()

    nodes = {}                     # node_id -> dict
    rels = []                      # list of dicts

    def add_node(nid, label, **props):
        n = nodes.setdefault(nid, {"node_id": nid, "node_label": label})
        if n["node_label"] != label:
            raise ValueError(f"{nid} assigned two labels: {n['node_label']} / {label}")
        for k, v in props.items():
            if v not in ("", None):
                n[k] = v

    def add_rel(src, tgt, rtype, **props):
        rels.append({"source_id": src, "target_id": tgt, "relationship_type": rtype, **props})

    # ---- Entities (Glycan / Protein) ----------------------------------------
    bm["entity_node_id"] = bm.apply(
        lambda r: bare_uniprot(r["assessed_biomarker_entity_id"])
        if r["assessed_entity_type"] == "protein" else r["assessed_biomarker_entity_id"].strip(), axis=1)

    for (nid, etype), g in bm.groupby(["entity_node_id", "assessed_entity_type"]):
        names = g["assessed_biomarker_entity"].tolist()
        name = most_common(names)
        label = "Protein" if etype == "protein" else "Glycan"
        add_node(nid, label, name=name, synonyms=join(n for n in names if n != name))

    # ---- Biomarkers ----------------------------------------------------------
    for bid, g in bm.groupby("biomarker_id"):
        texts = g.drop_duplicates("biomarker_index")["biomarker"].tolist()
        name = most_common(texts)
        types = sorted(set(g["assessed_entity_type"]))
        add_node(bid, "Biomarker",
                 name=name,
                 synonyms=join(t for t in texts if t != name),
                 controlled_vocab=join(g["biomarker_controlled_vocab"]),
                 change_type=most_common(g["change_type_vocab"]),
                 aspect_type=most_common(g["aspect_type_vocab"]),
                 composition="+".join(types))
        for _, r in g.drop_duplicates(["entity_node_id", "entity_index"]).iterrows():
            add_rel(bid, r["entity_node_id"], "HAS_COMPONENT",
                    entity_index=r["entity_index"], entity_type=r["assessed_entity_type"])

    # ---- Assertions and their context ----------------------------------------
    cond_labels, spec_labels = defaultdict(list), defaultdict(list)
    for idx, g in bm.groupby("biomarker_index"):
        r = g.sort_values("entity_index").iloc[0]
        aid = f"BMA:{idx}"
        add_node(aid, "BiomarkerAssertion", name=r["biomarker"], biomarker_index=idx,
                 best_biomarker_role=r["best_biomarker_role"])
        add_rel(aid, r["biomarker_id"], "ASSERTS")

        cid = r["condition_id"].strip()
        cond_labels[cid].append(r["condition"])
        add_rel(aid, cid, "ASSOCIATED_WITH", best_biomarker_role=r["best_biomarker_role"],
                change_type=r["change_type_vocab"], condition_label=r["condition"])

        sid = r["specimen_id"].strip()
        if sid:
            spec_labels[sid].append(r["specimen"])
            add_rel(aid, sid, "MEASURED_IN", specimen_label=r["specimen"])

        pub = r["evidence_source"].strip()
        add_node(pub, "Publication", name=pub.split(":", 1)[-1])
        add_rel(aid, pub, "SUPPORTED_BY", evidence=r["evidence"])

    # ---- Glycan-on-protein edges (from paired biomarkers) ---------------------
    pairs = defaultdict(lambda: {"bids": [], "cids": [], "n": 0})
    for idx, g in bm.groupby("biomarker_index"):
        gl = g[g["assessed_entity_type"] == "glycan"]["entity_node_id"].tolist()
        pr = g[g["assessed_entity_type"] == "protein"]["entity_node_id"].tolist()
        for a in gl:
            for b in pr:
                p = pairs[(a, b)]
                p["bids"].append(g["biomarker_id"].iloc[0])
                p["cids"].append(g["condition_id"].iloc[0])
                p["n"] += 1
    for (a, b), p in pairs.items():
        add_rel(a, b, "GLYCAN_ON_PROTEIN", biomarker_ids=join(p["bids"]),
                condition_ids=join(p["cids"]), n_assertions=p["n"])

    # ---- Specimens: UBERON -> TissueType, CL -> CellType ---------------------
    for sid, labels in spec_labels.items():
        prefix = sid.split(":")[0].upper()
        label = {"UBERON": "TissueType", "CL": "CellType"}.get(prefix)
        if label is None:
            report.append(f"WARNING: specimen id {sid} is neither UBERON nor CL; skipped")
            rels[:] = [e for e in rels if e["target_id"] != sid]
            continue
        name = most_common(labels)
        add_node(sid, label, name=name, synonyms=join(l for l in labels if l != name))

    # ---- Conditions + Disease Ontology hierarchy ------------------------------
    doid = load_doid(args.doid_obo, outdir, report)
    for cid, labels in cond_labels.items():
        term = doid.get(cid)
        name = term["name"] if term and term["name"] else most_common(labels)
        add_node(cid, "Condition", name=name, synonyms=join(l for l in labels if l != name),
                 in_dataset=True, obsolete=bool(term and term["obsolete"]))
        if doid and not term:
            report.append(f"WARNING: {cid} not found in Disease Ontology")
        elif term and term["obsolete"]:
            report.append(f"WARNING: {cid} is obsolete in DOID (replaced_by={term['replaced_by']})")

    if doid:
        added_edges, stack, visited = set(), list(cond_labels), set()
        while stack:
            cid = stack.pop()
            if cid in visited:
                continue
            visited.add(cid)
            for parent in doid.get(cid, {}).get("is_a", []):
                if parent not in nodes:
                    pt = doid.get(parent, {})
                    add_node(parent, "Condition", name=pt.get("name", ""), in_dataset=False,
                             obsolete=bool(pt.get("obsolete")))
                if (cid, parent) not in added_edges:
                    add_rel(cid, parent, "IS_A")
                    added_edges.add((cid, parent))
                stack.append(parent)
        n_anc = sum(1 for n in nodes.values() if n["node_label"] == "Condition" and n.get("in_dataset") is False)
        report.append(f"Condition hierarchy: {len(added_edges)} IS_A edges, {n_anc} ancestor terms added")

    # ---- Proteins: masterlist genes and isoforms -----------------------------
    ml["bare"] = ml["uniprotkb_canonical_ac"].map(bare_uniprot)
    protein_ids = [k for k, v in nodes.items() if v["node_label"] == "Protein"]
    missing = []
    for pid in protein_ids:
        g = ml[ml["bare"] == pid]
        if g.empty:
            nodes[pid]["in_masterlist"] = False
            missing.append(pid)
            continue
        canon = g["uniprotkb_canonical_ac"].iloc[0]
        add_node(pid, "Protein", in_masterlist=True, uniprot_canonical_isoform=canon,
                 uniprot_status=most_common(g["status"]))
        for sym in sorted(set(g["gene_name"]) - {""}):
            gid = f"GENE:{sym}"
            add_node(gid, "Gene", name=sym)
            add_rel(pid, gid, "ENCODED_BY")
        for col, status in (("reviewed_isoforms", "reviewed"), ("unreviewed_isoforms", "unreviewed")):
            for iso in sorted(set(g[col]) - {""}):
                add_node(iso, "Isoform", name=iso, isoform_status=status)
                add_rel(pid, iso, "HAS_ISOFORM", isoform_status=status, is_canonical=(iso == canon))
    report.append(f"Proteins without masterlist entry ({len(missing)}): {', '.join(sorted(missing))}")

    # ---- Validate against schema and write -----------------------------------
    node_df = pd.DataFrame(nodes.values())
    rel_df = pd.DataFrame(rels).drop_duplicates()
    validate(node_df, rel_df, nodes)

    node_cols = ["node_id", "node_label", "name", "synonyms"]
    node_cols += sorted({p for spec in SCHEMA["nodes"].values() for p in spec["properties"]} - set(node_cols))
    node_df = node_df.reindex(columns=node_cols).sort_values(["node_label", "node_id"])
    rel_cols = ["source_id", "target_id", "relationship_type"]
    rel_cols += sorted({p for spec in SCHEMA["relationships"].values() for p in spec["properties"]})
    rel_df = rel_df.reindex(columns=rel_cols).sort_values(["relationship_type", "source_id", "target_id"])

    node_df.to_csv(outdir / "nodes.csv", index=False)
    rel_df.to_csv(outdir / "relationships.csv", index=False)
    with open(outdir / "schema.yaml", "w", encoding="utf-8") as fh:
        yaml.safe_dump(SCHEMA, fh, sort_keys=False, allow_unicode=True, width=120)

    report.append("")
    report.append("Node counts:\n" + node_df["node_label"].value_counts().to_string())
    report.append("Relationship counts:\n" + rel_df["relationship_type"].value_counts().to_string())
    (outdir / "build_report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))


def validate(node_df, rel_df, nodes):
    known_labels = set(SCHEMA["nodes"])
    bad = set(node_df["node_label"]) - known_labels
    if bad:
        raise ValueError(f"Node labels missing from schema: {bad}")
    for rtype, g in rel_df.groupby("relationship_type"):
        spec = SCHEMA["relationships"].get(rtype)
        if spec is None:
            raise ValueError(f"Relationship {rtype} missing from schema")
        srcs = {spec["source"]} if isinstance(spec["source"], str) else set(spec["source"])
        tgts = {spec["target"]} if isinstance(spec["target"], str) else set(spec["target"])
        for s, t in zip(g["source_id"], g["target_id"]):
            if s not in nodes or t not in nodes:
                raise ValueError(f"{rtype}: dangling edge {s} -> {t}")
            if nodes[s]["node_label"] not in srcs or nodes[t]["node_label"] not in tgts:
                raise ValueError(f"{rtype}: {s}({nodes[s]['node_label']}) -> {t}({nodes[t]['node_label']}) "
                                 f"violates schema")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--biomarkers", default="biomarkers_llm_glycan.tsv")
    ap.add_argument("--masterlist", default="human_protein_masterlist_matched.csv")
    ap.add_argument("--doid-obo", default=None, help="Path to doid.obo; downloaded if omitted")
    ap.add_argument("--outdir", default="kg_output")
    sys.exit(build(ap.parse_args()))
