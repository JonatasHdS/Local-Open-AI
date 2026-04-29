"""
Optimized LLM-as-judge classification pipeline for health misinformation.

Optimizations vs original:
- Fully static system prompt (maximizes vLLM prefix cache hits)
- Async concurrent requests (saturates vLLM server throughput)
- Structured JSON output with guided decoding via response_format
- Multi-label output: "0" is exclusive; all others can be combined
- Strict label validation with retry logic
- temperature=0 for deterministic output
- Checkpoint saves every N rows (crash recovery)
"""

import asyncio
import json
import re
import logging
from pathlib import Path

import pandas as pd
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

# ── Configuration ────────────────────────────────────────────────────────────

VLLM_BASE_URL = "http://localhost:8000/v1"
VLLM_API_KEY  = "fake-key"
MODEL         = "Qwen/Qwen3-14B"

INPUT_CSV     = "dataset_950.csv"
MESSAGE_COL   = "Mensagem"
OUTPUT_CSV    = "resultado.csv"
CHECKPOINT    = "checkpoint.csv"       # partial results, survives crashes

MAX_CONCURRENT = 64                    # tune to your vLLM server capacity
MAX_RETRIES    = 3
CHECKPOINT_EVERY = 100                 # save partial results every N rows

VALID_LABELS = {
    "0",
    "1.1", "1.2", "1.3", "1.4",
    "2.1", "2.2", "2.3", "2.4",
    "3.1", "3.2",
    "4.1", "4.2", "4.3", "4.4", "4.5",
    "5.1", "5.2", "5.3", "5.4",
    "6.1", "6.2", "6.3", "6.4",
    "7.1", "7.2",
}

# ── System prompt (STATIC — do NOT modify at runtime) ────────────────────────
# Kept fully static so vLLM can cache this prefix across all 950 requests.
# Any per-request mutation (f-strings, dynamic insertions) would break caching.

SYSTEM_PROMPT = """\
You are a health misinformation classifier for Brazilian WhatsApp messages (2020–2024).

TASK: Assign one or more labels from the taxonomy below.
- If no misinformation is present → return ONLY ["0"]. "0" cannot be combined with other labels.
- Otherwise → return ALL subcategories that apply. Use the most specific subcategory always (e.g. "4.2", never just "4").
- Do NOT explain your reasoning.

TAXONOMY:
0   Not misinformation
1.1 Anecdotal Evidence — personal anecdotes replacing scientific data
1.2 Outdated Research — superseded or debunked studies cited as valid
1.3 Poor Research Quality — complaints about study insufficiency/quality
1.4 Fallibility of Science — claim that science can never be certain
2.1 Fabricated Studies — invented scientific studies
2.2 Selective Reporting — cherry-picked data or out-of-context results
2.3 Conspiracy — hidden malicious intent behind health actions
2.4 Censorship — claims that contrary information is suppressed
3.1 Alternative Products/Therapies — unvalidated treatments or false health claims
3.2 Alternatives to Vaccination — treatments claimed to replace vaccines
4.1 Alarmism — exaggerated fears about treatments/diseases (e.g. toxic ingredients)
4.2 Pseudoscientific Claims — science-sounding but baseless assertions
4.3 Correlation vs Causation — confusing correlation with causal relationship
4.4 Financial Misconduct — financial greed driving health decisions
4.5 Autonomy Rights — defending individual right to refuse treatment
5.1 Imperfect Protection — vaccines/treatments not 100% effective argument
5.2 Herd Immunity — others' immunity makes personal vaccination unnecessary
5.3 Natural Immunity — natural immunity preferred over vaccination
5.4 Low Disease Risk — disease minimised; vaccination portrayed as unnecessary
6.1 Direct Transmission — fear of contracting disease through vaccination
6.2 Specific Side Effects — reports of specific adverse effects
6.3 Unsafe Administration — dangerous application or health authority negligence
6.4 High-Risk Groups — exaggerated vulnerability of specific groups (children, elderly)
7.1 Viral Myths — widely circulated online health myths
7.2 Religious/Ethical Beliefs — vaccination conflicts with personal beliefs

OUTPUT FORMAT — return ONLY valid JSON, no other text:
{"labels": ["<code>", ...]}

EXAMPLES:
{"labels": ["0"]}
{"labels": ["4.2"]}
{"labels": ["2.3", "4.1", "6.2"]}\
"""

# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ── Core classification function ─────────────────────────────────────────────

client = AsyncOpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)

async def classify_message(message: str, semaphore: asyncio.Semaphore) -> str:
    """
    Classifies a single message. Returns a comma-separated label string or "ERROR".

    "0" is exclusive and cannot appear alongside other labels.
    All other labels can be combined freely.
    The semaphore limits concurrency to avoid overwhelming the vLLM server.
    """
    async with semaphore:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = await client.chat.completions.create(
                    model=MODEL,
                    temperature=0,          # fully deterministic
                    max_tokens=60,          # enough for ~5 labels as a JSON array
                    extra_body={
                        "enable_thinking": False,
                    },
                    # Guided decoding: force the model to emit valid JSON.
                    # vLLM supports OpenAI-compatible response_format.
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": message},
                    ],
                )

                raw = response.choices[0].message.content.strip()

                # Strip residual <think>…</think> blocks (Qwen thinking mode)
                raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

                parsed = json.loads(raw)
                labels = parsed.get("labels", [])

                # Normalize: accept both list and accidental bare string
                if isinstance(labels, str):
                    labels = [labels]
                labels = [str(l).strip() for l in labels]

                # Validate: all codes must exist in taxonomy
                if not labels or not all(l in VALID_LABELS for l in labels):
                    log.warning(
                        "Attempt %d/%d — invalid labels %r for message: %.80s",
                        attempt, MAX_RETRIES, labels, message,
                    )
                    continue

                # Enforce exclusivity of "0"
                if "0" in labels and len(labels) > 1:
                    log.warning(
                        "Attempt %d/%d — '0' mixed with other labels %r, retrying: %.80s",
                        attempt, MAX_RETRIES, labels, message,
                    )
                    continue

                # Store as comma-separated string for easy CSV handling
                return ",".join(sorted(labels))

            except json.JSONDecodeError as exc:
                log.warning("Attempt %d/%d — JSON parse error: %s", attempt, MAX_RETRIES, exc)
            except Exception as exc:
                log.warning("Attempt %d/%d — API error: %s", attempt, MAX_RETRIES, exc)
                await asyncio.sleep(2 ** attempt)   # exponential back-off

        log.error("All retries exhausted for message: %.80s", message)
        return "ERROR"


# ── Batch runner ──────────────────────────────────────────────────────────────

async def run_batch(messages: list[str]) -> list[str]:
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    tasks     = [classify_message(msg, semaphore) for msg in messages]
    results   = await tqdm_asyncio.gather(*tasks, desc="Classifying")
    return results


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def load_checkpoint() -> dict[int, str]:
    """Returns {original_index: label} for already-processed rows."""
    if not Path(CHECKPOINT).exists():
        return {}
    ck = pd.read_csv(CHECKPOINT, index_col="index")
    return ck["label"].to_dict()


def save_checkpoint(results: dict[int, str]) -> None:
    pd.DataFrame(
        [{"index": k, "label": v} for k, v in results.items()]
    ).to_csv(CHECKPOINT, index=False)


# ── Entry point ───────────────────────────────────────────────────────────────

async def main() -> None:
    df = pd.read_csv(INPUT_CSV)
    assert MESSAGE_COL in df.columns, f"Column '{MESSAGE_COL}' not found in {INPUT_CSV}"

    done = load_checkpoint()
    log.info("Loaded %d cached results from checkpoint.", len(done))

    # Identify rows that still need processing
    todo_idx = [i for i in df.index if i not in done]
    log.info("%d messages to classify (%d skipped).", len(todo_idx), len(done))

    # Process in chunks so we checkpoint periodically
    for chunk_start in range(0, len(todo_idx), CHECKPOINT_EVERY):
        chunk = todo_idx[chunk_start : chunk_start + CHECKPOINT_EVERY]
        messages = df.loc[chunk, MESSAGE_COL].tolist()

        labels = await run_batch(messages)

        for idx, label in zip(chunk, labels):
            done[idx] = label

        save_checkpoint(done)
        log.info("Checkpoint saved (%d/%d done).", len(done), len(df))

    # Assemble final output
    df["label"] = df.index.map(done)
    df.to_csv(OUTPUT_CSV, index=False)
    log.info("Done. Results saved to %s", OUTPUT_CSV)

    # Quick quality report
    n_error = (df["label"] == "ERROR").sum()
    if n_error:
        log.warning("%d rows could not be classified (label=ERROR).", n_error)

    # Explode multi-label column for per-code distribution
    exploded = df["label"].dropna().str.split(",").explode()
    dist = exploded.value_counts()
    log.info("Label distribution (per code):\n%s", dist.to_string())


if __name__ == "__main__":
    asyncio.run(main())
