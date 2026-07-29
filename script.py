#!/usr/bin/env python3
"""
kv_cache_ordering_test.py

Empirically tests the central caching claim from the note
"Prompt Ordering and Repetition in Decoder-Only LLMs":

    A KV entry is reusable EXACTLY WHEN it does not depend on the query.
      - query-LAST  -> context prefix is byte-stable across queries -> cache READ hit
      - query-FIRST -> query is part of the prefix, changes it       -> NO read, re-write

The Anthropic prompt cache is a *prefix* cache: it stores the encoded state of
everything up to a cache_control breakpoint, and reuses it only when a later
request begins with the exact same bytes. So the experiment is:

  Build one large, byte-stable CONTEXT block (must exceed the model's minimum
  cacheable length -- 2048 tokens for Sonnet 4.x).

  QUERY-LAST arrangement:  [ system: CONTEXT + <breakpoint> ] [ user: QUERY ]
     -> prefix = CONTEXT, identical no matter what QUERY is.
     -> Call with query A (cold, writes cache), then query B (warm).
        The warm call reuses the CONTEXT prefix even though the query changed:
        expect cache_read_input_tokens > 0.

  QUERY-FIRST arrangement: [ system: QUERY + CONTEXT + <breakpoint> ]
     -> prefix now includes QUERY, so changing the query changes the prefix.
     -> Call with query A (cold), then query B: the prefix differs, so nothing
        can be read; expect cache_read_input_tokens == 0 and a fresh write.

Two different queries per arrangement is the honest test: it isolates
"is the *context* reusable across queries?", which is the property the note
is actually about. (Sending the identical request twice would show a hit in
BOTH arrangements and prove nothing about ordering.)

Uses Sonnet 5 with thinking OFF (the repetition/ordering findings concern
non-reasoning behaviour; and thinking tokens would muddy usage accounting).

--------------------------------------------------------------------------
RUN:
    export ANTHROPIC_API_KEY=sk-ant-...
    rtk python kv_cache_ordering_test.py          # rtk per your shell setup
    # (plain `python kv_cache_ordering_test.py` also works)

COST: ~4 short calls, max_tokens=1 each. A few cents at most.
--------------------------------------------------------------------------
"""

import os
import sys
import time

try:
    from anthropic import Anthropic
except ImportError:
    sys.exit("pip install anthropic  (or: rtk pip install anthropic)")

MODEL = "claude-sonnet-5"  # non-reasoning path; thinking is off by default (no `thinking` param sent)
TTL_LABEL = "5-minute ephemeral cache"

# --- Build a byte-stable context comfortably above the 2048-token minimum ----
# Filler must be DETERMINISTIC and identical across calls -- no timestamps,
# no UUIDs, no randomness -- or the prefix bytes differ and nothing caches.
# ~200 numbered sentences lands well over 2048 tokens for Sonnet.
CONTEXT = "REFERENCE DOCUMENT (fictional company handbook).\n\n" + "\n".join(
    f"Section {i}. Policy clause {i}: employees in department {i % 12} must file "
    f"form F-{1000 + i} before the {((i % 28) + 1)}th of each month, routed to "
    f"reviewer R{i % 40}, retained for {3 + (i % 7)} years per schedule S{i % 9}."
    for i in range(1, 221)
)

QUERY_A = "Which reviewer handles department 5's filings?"
QUERY_B = "How many years are department 8's forms retained?"

client = Anthropic()  # reads ANTHROPIC_API_KEY from env


def usage_dict(resp):
    u = resp.usage
    return {
        "input": u.input_tokens,
        "cache_write": getattr(u, "cache_creation_input_tokens", 0) or 0,
        "cache_read": getattr(u, "cache_read_input_tokens", 0) or 0,
    }


def call_query_last(query):
    """[system: CONTEXT + breakpoint] [user: QUERY]  -> context is the stable prefix."""
    return client.messages.create(
        model=MODEL,
        max_tokens=1,
        system=[
            {
                "type": "text",
                "text": CONTEXT,
                "cache_control": {"type": "ephemeral"},
            },  # breakpoint AFTER context, BEFORE query
        ],
        messages=[{"role": "user", "content": query}],
    )


def call_query_first(query):
    """[system: QUERY + CONTEXT + breakpoint] -> query is inside the prefix."""
    return client.messages.create(
        model=MODEL,
        max_tokens=1,
        system=[
            {
                "type": "text",
                "text": f"{query}\n\n{CONTEXT}",
                "cache_control": {"type": "ephemeral"},
            },  # breakpoint after query+context
        ],
        messages=[{"role": "user", "content": "Answer the question above."}],
    )


def run(label, fn):
    print(f"\n{'='*66}\n{label}\n{'='*66}")

    cold = usage_dict(fn(QUERY_A))
    print(
        f"  cold (query A): write={cold['cache_write']:>6}  "
        f"read={cold['cache_read']:>6}  fresh_input={cold['input']}"
    )

    time.sleep(2)  # stay well inside the 5-min TTL

    warm = usage_dict(fn(QUERY_B))  # DIFFERENT query on purpose
    print(
        f"  warm (query B): write={warm['cache_write']:>6}  "
        f"read={warm['cache_read']:>6}  fresh_input={warm['input']}"
    )

    return warm["cache_read"]


def main():
    from dotenv import load_dotenv

    load_dotenv()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Set ANTHROPIC_API_KEY first.")

    print(f"Model: {MODEL}  |  {TTL_LABEL}")
    print("Prediction: query-LAST warm call reads the cache (context prefix stable);")
    print("            query-FIRST warm call does NOT (query changed the prefix).")

    last_read = run("QUERY-LAST   [context | breakpoint] + query", call_query_last)
    time.sleep(2)
    first_read = run("QUERY-FIRST  [query + context | breakpoint]", call_query_first)

    print(f"\n{'#'*66}\nRESULT\n{'#'*66}")
    print(f"  query-LAST  warm cache_read : {last_read}")
    print(f"  query-FIRST warm cache_read : {first_read}")

    ok = last_read > 0 and first_read == 0
    if ok:
        print(
            "\n  ✔ CLAIM HELD: the context prefix is reusable only when the query\n"
            "    sits after the breakpoint. Query-first destroys the reusable prefix."
        )
    elif last_read > 0 and first_read > 0:
        print(
            "\n  ~ PARTIAL: both read from cache. Most likely the two query-first\n"
            "    queries happened to share enough leading bytes, or a breakpoint\n"
            "    landed unexpectedly. Make QUERY_A and QUERY_B differ in their\n"
            "    FIRST characters and re-run."
        )
    elif last_read == 0:
        print(
            "\n  ✘ query-LAST did not cache. Check: (1) CONTEXT exceeds 2048 tokens,\n"
            "    (2) calls ran within 5 min, (3) CONTEXT is byte-identical across calls."
        )
    else:
        print("\n  ? Unexpected pattern -- inspect the per-call numbers above.")


if __name__ == "__main__":
    main()
