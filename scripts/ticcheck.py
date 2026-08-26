#!/usr/bin/env python3
"""Count the prose habits that have hardened into templates across the blog.

A 2026-07-28 read of all 78 posts found the same handful of moves everywhere.
This is the pre-ship check for them: run it bare for a corpus-wide ranking, or
name posts to see only those.

    python3 scripts/ticcheck.py
    python3 scripts/ticcheck.py src/content/blog/lazy-training.mdx
    python3 scripts/ticcheck.py --per-post          # every post, every tic

The counts are a smell test, not a gate. Being top of a list means the habit is
yours to break in this draft; it does not mean every instance is wrong.
"""
import glob
import os
import re
import sys
from collections import Counter, defaultdict

PATTERNS = {
    "inverted epigram (X is not Y. It is Z.)":
        r"(?:is|was|are|were)\s+not\s+[^.!?]{3,70}[.;]\s*It\s+(?:is|was)\b",
    "rhetorical-question heading": r"^##+ .*\?\s*$",
    "'live in your browser'": r"live in your browser",
    "certification adverb ('exactly the/what/where')":
        r"\bexactly (?:the|what|where|why|how)\b",
    "'not a metaphor / not decoration'":
        r"not (?:a )?(?:metaphor|decoration|decorative|merely|just a trick)",
    "'Every number ... from a real run'":
        r"[Ee]very (?:number|figure)[^.]{0,60}from a real run",
    "'is the finding/point/argument'":
        r"\bis the (?:finding|point|argument|reading|tell)\b",
    "'the price of'": r"\bthe price of\b",
    "money metaphor (saturation)":
        r"\b(?:price|cost|bill|ledger|budget|tax|currency|wage)\b",
    "'earns its'": r"earns? its|earned its",
    "'load-bearing'": r"load-bearing",
    "announced honesty":
        r"[Tt]he honest (?:so what|counterweight|version|answer)"
        r"|[Oo]ne honest (?:complication|caveat)|to be honest",
    "series self-counting":
        r"(?:spent|for the)\s+(?:\w+teen|five|six|seven|eight|nine|ten|eleven|twelve)"
        r"\s+posts?|for the \w+teenth time",
    "debt-recitation opener":
        r"^(?:\*)?(?:The )?(?:previous |last )post[^.]{0,80}"
        r"(?:left|ended|owed|closed on)\b",
    "failure pre-announced as valuable":
        r"fail(?:ed|s|ure)[^.]{0,60}(?:says more|tells you more|matters more|"
        r"is the valuable|worth more)",
}

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def scan(paths):
    totals = Counter()
    byfile = defaultdict(Counter)
    for f in paths:
        with open(f, encoding="utf-8") as fh:
            body = fh.read()
        body = re.sub(r"^---.*?^---", "", body, count=1, flags=re.S | re.M)
        body = re.sub(r"^```.*?^```", "", body, flags=re.S | re.M)
        slug = os.path.basename(f)[:-4]
        for name, pat in PATTERNS.items():
            n = len(re.findall(pat, body, re.M))
            if n:
                totals[name] += n
                byfile[name][slug] = n
    return totals, byfile


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    per_post = "--per-post" in sys.argv
    paths = args or sorted(glob.glob(os.path.join(ROOT, "src/content/blog/*.mdx")))
    totals, byfile = scan(paths)

    if not totals:
        print("clean: no tracked habits found")
        return

    if per_post:
        worst = defaultdict(list)
        for name, counts in byfile.items():
            for slug, n in counts.items():
                worst[slug].append((n, name))
        for slug in sorted(worst, key=lambda s: -sum(n for n, _ in worst[slug])):
            items = sorted(worst[slug], reverse=True)
            total = sum(n for n, _ in items)
            print(f"{total:4}  {slug}")
            for n, name in items:
                print(f"        {n:3}  {name}")
        return

    for name, n in totals.most_common():
        top = ", ".join(f"{k}:{v}" for k, v in byfile[name].most_common(4))
        print(f"{n:5}  in {len(byfile[name]):2} posts   {name}")
        print(f"         worst: {top}")


if __name__ == "__main__":
    main()
