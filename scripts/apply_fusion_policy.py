import argparse
import json
import math
import os
import re
from collections import Counter
from typing import Any, Dict, List, Set
import torch
import torch.nn as nn


DEFAULT_ACTIONS = ["pure_ppr", "pure_llm", "keep1", "keep2", "keep3", "keep4"]



def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows: List[Dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(obj: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default


def safe_int(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return default
        return int(x)
    except Exception:
        return default


def norm_len(x: Any, denom: float, cap: float = 1.0) -> float:
    return min(cap, safe_float(x) / max(1e-8, denom))


def tokenize_question(q: str) -> List[str]:
    q = str(q or "").lower()
    return re.findall(r"[a-z0-9][a-z0-9\-']+", q)


def infer_hop_count(row: Dict[str, Any]) -> int:
    qid = str(row.get("id") or row.get("question_id") or "")
    m = re.search(r"(\d+)hop", qid)
    if m:
        return int(m.group(1))

    qd = row.get("question_decomposition") or []
    if isinstance(qd, list) and qd:
        return len(qd)

    return 0


def get_passages(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    return (
        row.get("evidence_passages")
        or row.get("retrieved_passages")
        or row.get("passages")
        or []
    )


def set_passages(row: Dict[str, Any], passages: List[Dict[str, Any]]) -> None:
    if "evidence_passages" in row:
        row["evidence_passages"] = passages
    elif "retrieved_passages" in row:
        row["retrieved_passages"] = passages
    elif "passages" in row:
        row["passages"] = passages
    else:
        row["evidence_passages"] = passages


def passage_aliases(p: Dict[str, Any]) -> Set[str]:
    ids = set()

    for k in ["id", "passage_id", "node_id", "idx", "fallback_idx"]:
        if p.get(k) is not None:
            ids.add(str(p.get(k)))

    meta = p.get("metadata") or {}
    for k in [
        "fallback_idx",
        "idx",
        "passage_idx",
        "source_idx",
        "corpus_idx",
        "passage_id",
        "node_id",
    ]:
        if meta.get(k) is not None:
            ids.add(str(meta.get(k)))

    return ids


def alias_key(p: Dict[str, Any]) -> str:
    ids = sorted(passage_aliases(p))
    if ids:
        return "||".join(ids)
    return json.dumps(p, sort_keys=True, ensure_ascii=False)


def add_unique(out: List[Dict[str, Any]], p: Dict[str, Any], seen: Set[str]) -> None:
    key = alias_key(p)
    if key in seen:
        return
    seen.add(key)
    out.append(p)


def base_rank_map(base_passages: List[Dict[str, Any]]) -> Dict[str, int]:
    mp = {}
    for i, p in enumerate(base_passages, start=1):
        mp[alias_key(p)] = i
    return mp


def topk_old_ranks(
    fused: List[Dict[str, Any]],
    base_passages: List[Dict[str, Any]],
    k: int,
) -> List[int]:
    mp = base_rank_map(base_passages)
    ranks = []
    for p in fused[:k]:
        ranks.append(mp.get(alias_key(p), -1))
    return ranks


def overlap_count(a: List[Dict[str, Any]], b: List[Dict[str, Any]], k: int) -> int:
    aset = {alias_key(p) for p in a[:k]}
    bset = {alias_key(p) for p in b[:k]}
    return len(aset & bset)


def get_filtered_triples(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    triples = list(row.get("filtered_triples") or [])

    if not triples:
        for step in row.get("steps") or []:
            obs = step.get("observation") or {}
            sr = obs.get("search_result") or {}
            triples.extend(sr.get("filtered_triples") or [])

    seen = set()
    out = []
    for t in triples:
        key = json.dumps(t, sort_keys=True, ensure_ascii=False)
        if key not in seen:
            seen.add(key)
            out.append(t)
    return out


def llm_selected_indices(row: Dict[str, Any]) -> List[int]:
    meta = row.get("metadata") or {}
    sel = (meta.get("llm_evidence_selector") or {}).get("selected_indices") or []
    out = []
    for x in sel:
        try:
            out.append(int(x))
        except Exception:
            pass
    return out


def fuse_passages(
    action: str,
    base_passages: List[Dict[str, Any]],
    llm_passages: List[Dict[str, Any]],
    target_k: int,
) -> List[Dict[str, Any]]:

    if action == "pure_ppr":
        return list(base_passages)

    if action == "pure_llm":
        out = []
        seen = set()

        for p in llm_passages:
            add_unique(out, p, seen)

        for p in base_passages:
            add_unique(out, p, seen)

        return out

    if action.startswith("keep"):
        try:
            keep_n = int(action.replace("keep", ""))
        except Exception:
            keep_n = 2

        out = []
        seen = set()

        for p in base_passages[:keep_n]:
            add_unique(out, p, seen)

        for p in llm_passages:
            if len(out) >= target_k:
                break
            add_unique(out, p, seen)

        for p in base_passages:
            add_unique(out, p, seen)

        return out

    raise ValueError(f"Unknown fusion action: {action}")


def action_keep_value(action: str) -> int:
    if action.startswith("keep"):
        return safe_int(action.replace("keep", ""), 0)
    return 0


def row_base_features(
    base_row: Dict[str, Any],
    llm_row: Dict[str, Any],
    base_passages: List[Dict[str, Any]],
    llm_passages: List[Dict[str, Any]],
) -> List[float]:

    q = base_row.get("question") or base_row.get("query") or ""
    toks = tokenize_question(q)

    selected = llm_selected_indices(llm_row)
    selected = [safe_int(x) for x in selected if safe_int(x) > 0]

    if selected:
        sel_min = min(selected)
        sel_max = max(selected)
        sel_mean = sum(selected) / len(selected)
    else:
        sel_min = 0
        sel_max = 0
        sel_mean = 0.0

    count_1_5 = sum(1 for x in selected if 1 <= x <= 5)
    count_6_10 = sum(1 for x in selected if 6 <= x <= 10)

    feats = [
        norm_len(infer_hop_count(base_row), 4.0),
        norm_len(len(base_passages), 20.0),
        norm_len(len(llm_passages), 20.0),
        norm_len(len(get_filtered_triples(base_row)), 50.0),
        norm_len(overlap_count(base_passages, llm_passages, 5), 5.0),
        norm_len(len(selected), 5.0),
        norm_len(sel_min, 10.0),
        norm_len(sel_max, 10.0),
        norm_len(sel_mean, 10.0),
        norm_len(count_1_5, 5.0),
        norm_len(count_6_10, 5.0),
        norm_len(len(toks), 40.0),
        norm_len(len(str(q)), 300.0),
    ]

    return feats


def action_features(
    action: str,
    fused: List[Dict[str, Any]],
    base_passages: List[Dict[str, Any]],
    actions: List[str],
    target_k: int,
) -> List[float]:

    old_ranks = topk_old_ranks(fused, base_passages, target_k)

    old_rank_sum = sum(r for r in old_ranks if r > 0)
    old_rank_max = max([r for r in old_ranks if r > 0] or [0])
    num_promoted_from_6_10 = sum(1 for r in old_ranks if 6 <= r <= 10)
    num_from_top5 = sum(1 for r in old_ranks if 1 <= r <= 5)

    action_feat = [1.0 if action == a else 0.0 for a in actions]

    feats = []
    feats.extend(action_feat)

    feats.extend([
        1.0 if action == "pure_ppr" else 0.0,
        1.0 if action == "pure_llm" else 0.0,
        1.0 if action.startswith("keep") else 0.0,
        norm_len(action_keep_value(action), 4.0),

        norm_len(old_rank_sum, 40.0),
        norm_len(old_rank_max, 10.0),
        norm_len(num_promoted_from_6_10, 5.0),
        norm_len(num_from_top5, 5.0),
    ])

    return feats


def build_action_matrix(
    base_row: Dict[str, Any],
    llm_row: Dict[str, Any],
    actions: List[str],
    target_k: int,
) -> Dict[str, Any]:

    base_passages = get_passages(base_row)
    llm_passages = get_passages(llm_row)

    row_feats = row_base_features(
        base_row=base_row,
        llm_row=llm_row,
        base_passages=base_passages,
        llm_passages=llm_passages,
    )

    feat_rows = []
    fused_by_action = {}

    for action in actions:
        fused = fuse_passages(
            action=action,
            base_passages=base_passages,
            llm_passages=llm_passages,
            target_k=target_k,
        )

        fused_by_action[action] = fused
        feat_rows.append(
            row_feats + action_features(
                action=action,
                fused=fused,
                base_passages=base_passages,
                actions=actions,
                target_k=target_k,
            )
        )

    return {
        "features": torch.tensor(feat_rows, dtype=torch.float32),
        "fused_by_action": fused_by_action,
    }



class FusionPolicyNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        layers = []
        d = input_dim

        for _ in range(num_layers):
            layers.append(nn.Linear(d, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            d = hidden_dim

        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            return self.net(x).squeeze(-1)

        b, a, f = x.shape
        return self.net(x.reshape(b * a, f)).reshape(b, a)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_path", required=True)
    ap.add_argument("--llmselect_path", required=True)
    ap.add_argument("--checkpoint_path", required=True)
    ap.add_argument("--output_path", required=True)
    ap.add_argument("--summary_path", default=None)

    ap.add_argument("--target_k", type=int, default=5)
    ap.add_argument("--fallback_action", default="keep2")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if os.path.exists(args.output_path) and not args.force:
        raise SystemExit(f"Output exists. Use --force: {args.output_path}")

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")

    ckpt = torch.load(args.checkpoint_path, map_location=device)
    config = ckpt.get("config") or {}
    actions = ckpt.get("actions") or config.get("actions") or DEFAULT_ACTIONS

    input_dim = int(config.get("input_dim", 27))
    hidden_dim = int(config.get("hidden_dim", 128))
    num_layers = int(config.get("num_layers", 2))
    dropout = float(config.get("dropout", 0.10))

    model = FusionPolicyNet(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    base_rows = load_jsonl(args.base_path)
    llm_rows = load_jsonl(args.llmselect_path)

    n = min(len(base_rows), len(llm_rows))
    if args.limit is not None:
        n = min(n, args.limit)

    out_rows = []
    action_counts = Counter()
    errors = 0

    with torch.no_grad():
        for i in range(n):
            base_row = base_rows[i]
            llm_row = llm_rows[i]

            new_row = dict(base_row)

            try:
                item = build_action_matrix(
                    base_row=base_row,
                    llm_row=llm_row,
                    actions=actions,
                    target_k=args.target_k,
                )

                feats = item["features"].to(device)
                logits = model(feats)
                probs = torch.softmax(logits, dim=-1)

                pred_idx = int(torch.argmax(logits).item())
                pred_action = actions[pred_idx]

                fused = item["fused_by_action"][pred_action]

                set_passages(new_row, fused)

                new_row.setdefault("metadata", {})
                new_row["metadata"]["fusion_policy"] = {
                    "checkpoint_path": args.checkpoint_path,
                    "selected_action": pred_action,
                    "selected_action_index": pred_idx,
                    "actions": actions,
                    "logits": [float(x) for x in logits.detach().cpu().tolist()],
                    "probs": [float(x) for x in probs.detach().cpu().tolist()],
                    "target_k": args.target_k,
                }

                action_counts[pred_action] += 1

            except Exception as e:
                errors += 1

                # Safe fallback.
                base_passages = get_passages(base_row)
                llm_passages = get_passages(llm_row)
                fused = fuse_passages(
                    action=args.fallback_action,
                    base_passages=base_passages,
                    llm_passages=llm_passages,
                    target_k=args.target_k,
                )
                set_passages(new_row, fused)

                new_row.setdefault("metadata", {})
                new_row["metadata"]["fusion_policy_error"] = str(e)
                new_row["metadata"]["fusion_policy_fallback_action"] = args.fallback_action

                action_counts[args.fallback_action] += 1

            out_rows.append(new_row)

            if (i + 1) % 100 == 0:
                print(f"processed {i + 1}/{n}", flush=True)

    write_jsonl(out_rows, args.output_path)

    summary = {
        "base_path": args.base_path,
        "llmselect_path": args.llmselect_path,
        "checkpoint_path": args.checkpoint_path,
        "output_path": args.output_path,
        "num_rows": len(out_rows),
        "target_k": args.target_k,
        "actions": actions,
        "action_counts": dict(action_counts),
        "num_errors": errors,
        "device": str(device),
    }

    print(json.dumps(summary, indent=2))

    if args.summary_path:
        write_json(summary, args.summary_path)
        print("summary_path:", args.summary_path)

    print("saved:", args.output_path)


if __name__ == "__main__":
    main()
