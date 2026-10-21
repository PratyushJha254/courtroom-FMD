"""Prosecution and defense argument generation with an OpenAI-compatible chat API (DeepSeek by default).

Arguments are appended to the "Open-ended Verifiable Question" field as

    <original question>\n\n[PROSECUTION]\n...\n\n[DEFENSE]\n...

which is the format the training and inference code expects.
"""
import asyncio

from tqdm.asyncio import tqdm

from . import prompts
from .data import QUESTION_KEY

FAILED = "GENERATION_FAILED"


class ChatClient:
    def __init__(self, api_key, base_url, model, temperature, max_tokens=None, retries=3):
        from openai import AsyncOpenAI

        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retries = retries

    async def complete(self, system, user):
        kwargs = {"max_tokens": self.max_tokens} if self.max_tokens else {}
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        for _ in range(self.retries):
            try:
                resp = await self.client.chat.completions.create(
                    model=self.model, messages=messages, temperature=self.temperature, **kwargs
                )
                if resp.choices[0].message.content:
                    return resp.choices[0].message.content
            except Exception as e:
                print(f"API error: {e}")
            await asyncio.sleep(2)
        return FAILED


def claim_from_question(question):
    if "[true/false]" in question:
        return question.split("[true/false]")[-1].strip().lstrip(".\n\r ")
    return question.strip()


def split_round_one(text):
    """Return (claim, prosecution, defense) from an augmented question, or None if untagged."""
    if "[PROSECUTION]" not in text or "[DEFENSE]" not in text:
        return None
    base, rest = text.split("[PROSECUTION]", 1)
    prosecution, defense = rest.split("[DEFENSE]", 1)
    return claim_from_question(base), prosecution.strip(), defense.strip()


async def _round_one(llm, item, sem):
    question = item.get(QUESTION_KEY, item.get("instruction", item.get("claim", "")))
    claim = claim_from_question(question)
    async with sem:
        prosecution = await llm.complete(prompts.PROSECUTOR, claim)
    async with sem:
        defense = await llm.complete(
            prompts.DEFENDER, prompts.DEFENDER_USER.format(claim=claim, prosecution=prosecution)
        )
    return {**item, QUESTION_KEY: f"{question}\n\n[PROSECUTION]\n{prosecution}\n\n[DEFENSE]\n{defense}"}


async def _round_two(llm, item, sem):
    text = item.get(QUESTION_KEY, "")
    parts = split_round_one(text)
    if parts is None:
        return item
    claim, prosecution, defense = parts
    async with sem:
        prosecution_2 = await llm.complete(
            prompts.PROSECUTOR_R2,
            prompts.PROSECUTOR_R2_USER.format(claim=claim, prosecution=prosecution, defense=defense),
        )
    async with sem:
        defense_2 = await llm.complete(
            prompts.DEFENDER_R2,
            prompts.DEFENDER_R2_USER.format(claim=claim, defense=defense, prosecution_2=prosecution_2),
        )
    return {**item, QUESTION_KEY: f"{text}\n\n[PROSECUTION 2]\n{prosecution_2}\n\n[DEFENSE 2]\n{defense_2}"}


async def generate(llm, items, round_=1, concurrency=50):
    step = _round_one if round_ == 1 else _round_two
    sem = asyncio.Semaphore(concurrency)

    async def run(i, item):
        return i, await step(llm, item, sem)

    tasks = [run(i, item) for i, item in enumerate(items)]
    results = [await t for t in tqdm.as_completed(tasks, total=len(tasks), desc=f"round {round_}")]
    return [item for _, item in sorted(results, key=lambda r: r[0])]
