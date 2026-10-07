"""Probe macro policies on the human decisions of a BC dataset.

    uv run python scripts/policy_probe.py data/bc/v7 n102=runs/step2n-train/checkpoints/update_000102.pt p060=...

For every checkpoint and every macro state type it reports the critic value, the
policy entropy, the KL from the first checkpoint, the share of screens where the
argmax agrees with the first checkpoint, and the mean probability put on each
kind of option (rest HEAL/SMITH, card take/skip, shop item category, map node
type, event option index).  The ``human`` row is the frequency of the recorded
choice.  Both are comparable where every screen offers the same options (rest
sites, card rewards); for shops and maps the human row is an unconditional
frequency while the model rows are means over the screens offering the option.
"""

from __future__ import annotations

import gzip
import json
import math
import sys

import torch

from sts2rl.actions import GameAction
from sts2rl.encoder import GameEncoder, GameTokenizer, GameVocabulary
from sts2rl.env import GameObservation
from sts2rl.training.surgery import resolve_source

MACRO = ("rest_site", "card_reward", "shop", "fake_merchant", "map", "event")
HP_BUCKETS = ("<50%", "50-80%", ">80%")


def node_type(game_map: dict, candidate: dict) -> str:
    options = game_map.get("next_options") or []
    index = candidate.get("index")
    if index is None or index >= len(options):
        return "?"
    option = options[index]
    if isinstance(option, dict):
        if option.get("type"):
            return str(option["type"])
        key = (option.get("col"), option.get("row"))
    elif isinstance(option, (list, tuple)):
        key = tuple(option[:2])
    else:
        return "?"
    for node in game_map.get("nodes", []):
        if (node.get("col"), node.get("row")) == key:
            return str(node.get("type"))
    return "?"


def labels(decision: dict) -> list[str]:
    """One aggregation label per candidate."""
    state_type, state, out = decision["state_type"], decision["raw_state"], []
    for candidate in decision["candidates"]:
        if state_type == "rest_site":
            options = (state.get("rest_site") or {}).get("options") or []
            if candidate["type"] == "choose_rest_option" and candidate["index"] < len(options):
                out.append(options[candidate["index"]]["id"])
            else:
                out.append(candidate["type"])
        elif state_type == "card_reward":
            out.append("skip" if candidate["type"] == "skip_card_reward" else "take")
        elif state_type in ("shop", "fake_merchant"):
            items = (state.get(state_type) or {}).get("items") or []
            if candidate["type"] == "shop_purchase" and candidate["index"] < len(items):
                out.append("buy:" + str(items[candidate["index"]].get("category")))
            else:
                out.append(candidate["type"])
        elif state_type == "map":
            out.append(node_type(state.get("map") or {}, candidate))
        else:
            out.append(str(candidate.get("index", candidate["type"])))
    return out


def hp_bucket(ratio: float) -> str:
    if ratio < 0.5:
        return HP_BUCKETS[0]
    return HP_BUCKETS[1] if ratio < 0.8 else HP_BUCKETS[2]


def mass(probabilities: list[float], candidate_labels: list[str], wanted: str) -> float:
    return sum(p for p, label in zip(probabilities, candidate_labels) if label == wanted)


def load_rows(dataset: str, tokenizer: GameTokenizer) -> list[tuple]:
    rows = []
    with gzip.open(f"{dataset}/decisions.jsonl.gz", "rt", encoding="utf-8") as handle:
        for line in handle:
            decision = json.loads(line)
            if decision["state_type"] not in MACRO or len(decision["candidates"]) < 2:
                continue
            candidates = [GameAction.from_dict(c) for c in decision["candidates"]]
            try:
                tokenized = tokenizer.tokenize_decision(
                    GameObservation(decision["raw_state"], decision.get("player_detail")), candidates
                )
            except Exception:  # noqa: BLE001 - a screen the tokenizer rejects is simply skipped
                continue
            player = decision["raw_state"].get("player") or {}
            hp = (player.get("hp") or 0) / max(player.get("max_hp") or 1, 1)
            rows.append((decision["state_type"], labels(decision), decision["expert_index"], hp, tokenized))
    return rows


def score(path: str, vocabulary: GameVocabulary, rows: list[tuple]) -> list[tuple[list[float], float]]:
    manager, resolved = resolve_source(path, vocabulary)
    loaded = manager.load(resolved, map_location="cpu")
    encoder = GameEncoder(vocabulary, loaded.plan.encoder)
    encoder.load_state_dict(dict(loaded.agent_state["encoder"]))
    encoder.eval()
    out = []
    with torch.no_grad():
        for row in rows:
            output = encoder.policy_value(row[4])
            out.append((torch.softmax(output.logits.flatten(), 0).tolist(), float(output.value)))
    return out


def main() -> None:
    dataset, checkpoints = sys.argv[1], [arg.split("=", 1) for arg in sys.argv[2:]]
    vocabulary = GameVocabulary.from_bundled_data()
    rows = load_rows(dataset, GameTokenizer(vocabulary))
    print(f"{len(rows)} macro decisions:", {t: sum(r[0] == t for r in rows) for t in MACRO}, flush=True)
    results = {}
    for label, path in checkpoints:
        results[label] = score(path, vocabulary, rows)
        print("scored", label, flush=True)
    reference = checkpoints[0][0]
    for state_type in MACRO:
        idx = [i for i, row in enumerate(rows) if row[0] == state_type]
        if not idx:
            continue
        print(f"\n== {state_type}: {len(idx)} decisions")
        all_labels = sorted({lab for i in idx for lab in rows[i][1]}, key=lambda x: -sum(x in rows[i][1] for i in idx))[:8]
        header = ["ckpt", "value", "entropy", f"KL({reference}||·)", f"argmax={reference}"]
        header += [f"P({lab})" for lab in all_labels]
        if state_type == "rest_site":
            header += [f"P(HEAL){bucket}" for bucket in HP_BUCKETS]
        print(" | ".join(header))
        human = ["human", "", "", "", ""]
        human += [f"{sum(rows[i][1][rows[i][2]] == lab for i in idx) / len(idx):.2f}" for lab in all_labels]
        if state_type == "rest_site":
            for bucket in HP_BUCKETS:
                ii = [i for i in idx if hp_bucket(rows[i][3]) == bucket]
                human.append(f"{sum(rows[i][1][rows[i][2]] == 'HEAL' for i in ii) / len(ii):.2f}(n={len(ii)})" if ii else "-")
        print(" | ".join(human))
        for label, _ in checkpoints:
            scored, base = results[label], results[reference]
            entropy = sum(-sum(p * math.log(p + 1e-12) for p in scored[i][0]) for i in idx) / len(idx)
            kl = sum(
                sum(q * math.log((q + 1e-12) / (p + 1e-12)) for q, p in zip(base[i][0], scored[i][0])) for i in idx
            ) / len(idx)
            agree = sum(
                max(range(len(scored[i][0])), key=scored[i][0].__getitem__)
                == max(range(len(base[i][0])), key=base[i][0].__getitem__)
                for i in idx
            ) / len(idx)
            value = sum(scored[i][1] for i in idx) / len(idx)
            line = [label, f"{value:.2f}", f"{entropy:.2f}", f"{kl:.3f}", f"{agree:.2f}"]
            for lab in all_labels:
                ii = [i for i in idx if lab in rows[i][1]]
                line.append(f"{sum(mass(scored[i][0], rows[i][1], lab) for i in ii) / len(ii):.2f}" if ii else "-")
            if state_type == "rest_site":
                for bucket in HP_BUCKETS:
                    ii = [i for i in idx if hp_bucket(rows[i][3]) == bucket and "HEAL" in rows[i][1]]
                    line.append(f"{sum(mass(scored[i][0], rows[i][1], 'HEAL') for i in ii) / len(ii):.2f}" if ii else "-")
            print(" | ".join(line))


if __name__ == "__main__":
    main()
