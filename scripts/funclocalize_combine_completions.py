"""Reconstruct {messages, tools} from OpenHands events when log_completions=false.

Output mirrors what evaluation/combine_final_completions.py would have produced
(an output.with_completions.jsonl.gz next to the input), but built from the
events recorded in output.jsonl's `history` field instead of llm_completions/*.
"""

import argparse
import gzip
import json
from typing import Any

from tqdm import tqdm


def get_openai_tools() -> list[dict[str, Any]]:
    """Resolve the agent's tool set via the registry (default preset, no browser,
    plus the built-in Finish/Think tools that the agent always carries)."""
    from openhands.sdk.tool import resolve_tool
    from openhands.sdk.tool.builtins.finish import FinishTool
    from openhands.sdk.tool.builtins.think import ThinkTool
    from openhands.tools.preset.default import get_default_tools

    specs = get_default_tools(enable_browser=False)

    # Each tool's create() pokes at different conv_state fields. Build a
    # nested stub that satisfies all the access patterns we hit. We're after
    # schemas, not real execution, so concrete values don't matter.
    import tempfile
    from types import SimpleNamespace

    work = tempfile.mkdtemp(prefix="funclocalize_tool_schema_")
    stub = SimpleNamespace(
        workspace=SimpleNamespace(working_dir=work),
        env_observation_persistence_dir=work,
        persistence_dir=work,
        secrets={},
        agent=SimpleNamespace(
            llm=SimpleNamespace(vision_is_active=lambda: False),
        ),
    )
    resolved: list = []
    for spec in specs:
        defs = resolve_tool(spec, stub)  # pyright: ignore[reportArgumentType]
        if isinstance(defs, (list, tuple)):
            resolved.extend(defs)
        else:
            resolved.append(defs)
    # Built-in tools (Finish, Think) aren't in the spec registry; instantiate
    # them directly via their classmethod, mirroring how Agent does it.
    for cls in (FinishTool, ThinkTool):
        instances = cls.create(conv_state=None)
        resolved.extend(list(instances))

    out: list[dict[str, Any]] = []
    for tool in resolved:
        oai = tool.to_openai_tool()
        if hasattr(oai, "model_dump"):
            out.append(oai.model_dump(mode="json", exclude_none=True))
        else:
            out.append(oai)
    return out


def _flatten_content(parts):
    """Collapse a list of {type:'text', text:...} parts into a single text string."""
    if isinstance(parts, str):
        return parts
    if parts is None:
        return ""
    chunks = []
    for p in parts:
        if isinstance(p, dict):
            if p.get("type") == "text" and "text" in p:
                chunks.append(p["text"])
            else:
                chunks.append(json.dumps(p))
        else:
            chunks.append(str(p))
    return "".join(chunks)


def events_to_messages(history: list[dict]) -> list[dict]:
    """Map OpenHands events to OpenAI chat messages."""
    messages: list[dict] = []
    for ev in history:
        kind = ev.get("kind")
        if kind == "SystemPromptEvent":
            sp = ev.get("system_prompt") or {}
            text = sp.get("text") if isinstance(sp, dict) else str(sp)
            messages.append({"role": "system", "content": text or ""})
        elif kind == "MessageEvent":
            llm_msg = ev.get("llm_message") or {}
            role = llm_msg.get("role", ev.get("source", "user"))
            content = _flatten_content(llm_msg.get("content"))
            msg = {"role": role, "content": content}
            tcs = llm_msg.get("tool_calls")
            if tcs:
                msg["tool_calls"] = tcs
            tci = llm_msg.get("tool_call_id")
            if tci:
                msg["tool_call_id"] = tci
            messages.append(msg)
        elif kind == "ActionEvent":
            thought = _flatten_content(ev.get("thought"))
            tc = ev.get("tool_call") or {}
            tool_call = {
                "id": tc.get("id") or ev.get("tool_call_id"),
                "type": "function",
                "function": {
                    "name": tc.get("name") or ev.get("tool_name"),
                    "arguments": tc.get("arguments", "{}"),
                },
            }
            messages.append(
                {
                    "role": "assistant",
                    "content": thought,
                    "tool_calls": [tool_call],
                }
            )
        elif kind == "ObservationEvent":
            obs = ev.get("observation") or {}
            content = _flatten_content(obs.get("content"))
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": ev.get("tool_call_id"),
                    "content": content,
                }
            )
        # ConversationStateUpdateEvent and others are not part of the message stream.
    return messages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl_path", type=str)
    parser.add_argument(
        "--limit", type=int, default=0, help="Process only N rows (for testing)"
    )
    args = parser.parse_args()

    tools = get_openai_tools()
    print(
        f"Resolved {len(tools)} OpenAI tool schemas: {[t.get('function', {}).get('name') for t in tools]}"
    )

    output_path = args.jsonl_path.replace(".jsonl", ".with_completions.jsonl.gz")
    print(f"Writing to {output_path}")

    written = 0
    with open(args.jsonl_path, "r") as f_in, gzip.open(output_path, "wt") as f_out:
        for line in tqdm(f_in):
            data = json.loads(line)
            history = data.get("history") or []
            messages = events_to_messages(history)
            data["raw_completions"] = {"messages": messages, "tools": tools}
            f_out.write(json.dumps(data) + "\n")
            written += 1
            if args.limit and written >= args.limit:
                break

    print(f"Wrote {written} rows to {output_path}")


if __name__ == "__main__":
    main()
