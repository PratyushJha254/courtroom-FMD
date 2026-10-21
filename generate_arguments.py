"""Generate prosecution/defense arguments for every claim in an FMD json file.

Example:
    export DEEPSEEK_API_KEY=...
    python generate_arguments.py --input data/test.json --output data/test_aug_0.5.json
"""
import argparse
import asyncio
import json
import os

from fmd.data import load_json
from fmd.generation import FAILED, ChatClient, generate


def parse_ids(value):
    if os.path.isfile(value):
        with open(value) as f:
            return [int(line) for line in f if line.strip()]
    return [int(x) for x in value.split(",")]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--round", type=int, choices=[1, 2], default=1,
                   help="2 = add a second debate round to a file that already has round-one arguments")
    p.add_argument("--model", default="deepseek-chat")
    p.add_argument("--base-url", default="https://api.deepseek.com")
    p.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--max-tokens", type=int, default=350, help="0 = no limit")
    p.add_argument("--concurrency", type=int, default=50)
    p.add_argument("--limit", type=int, help="only process the first N items")
    p.add_argument("--ids", help="comma-separated item positions, or a file with one per line")
    args = p.parse_args()

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"set the {args.api_key_env} environment variable")

    items = load_json(args.input)
    if args.ids:
        ids = parse_ids(args.ids)
        items = [{**items[i], "id": i} for i in ids]
    elif args.limit:
        items = items[: args.limit]

    llm = ChatClient(api_key, args.base_url, args.model, args.temperature, args.max_tokens or None)
    out = asyncio.run(generate(llm, items, args.round, args.concurrency))

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=4)

    failed = sum(FAILED in item.get("Open-ended Verifiable Question", "") for item in out)
    print(f"wrote {len(out)} items to {args.output} ({failed} with a failed generation)")


if __name__ == "__main__":
    main()
