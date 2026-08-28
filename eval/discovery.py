"""
discovery.py
------------
Maps a generation run directory onto the Phase 0 prompt list.

Anchors prompt_idx to position in the prompts yaml -- NOT to folder names --
because the CLIPSeg noun and the variable/canonical split are keyed by index.
A silent mismatch here mislabels prompts instead of raising.

Expected layout (what sd3/run.py produces):

    <run_dir>/
        000_a_photo_of_a_convertible_sports_car/
            seed1.png ... seed64.png
        001_a_photo_of_a_pickup_truck/
            seed1.png ... seed64.png
        ...

Tolerates naming drift: missing index prefix, different image extensions,
seed numbers embedded anywhere in the filename.
"""
import re
from pathlib import Path

import yaml

IMG_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def slugify(text: str, max_len: int = 60) -> str:
    """Identical to sd3/run.py's slugify -- keep the two in sync."""
    text = re.sub(r"[^a-z0-9]+", "_", text.lower().strip()).strip("_")
    return text[:max_len] if text else "prompt"


def load_prompts(path):
    """Accepts a flat yaml list or {"prompts": [...]}."""
    data = yaml.safe_load(open(path))
    if isinstance(data, dict):
        data = data.get("prompts", []) or []
    return data or []


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def _seed_of(path: Path, fallback: int):
    """Parse a seed from the filename; fall back to sorted position."""
    for pat in (r"seed[_\-]?(\d+)", r"(\d+)$", r"(\d+)"):
        m = re.search(pat, path.stem, re.IGNORECASE)
        if m:
            return int(m.group(1)), True
    return fallback, False


def resolve_run(run_dir, prompts_path, nouns, n_variable, n_seeds=64,
                verbose=True):
    """Resolve a run directory against the prompt list.

    Returns (records, problems) where each record is
        dict(idx, dir, path, how, noun, cls, seeds=[(seed, Path)], n_found)
    and `problems` is a list of human-readable strings. An empty `problems`
    means the run is clean and safe to featurize.
    """
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise SystemExit(f"not a directory: {run_dir}")

    prompts = load_prompts(prompts_path)
    if not prompts:
        raise SystemExit(f"no prompts loaded from {prompts_path}")
    slugs = [slugify(p) for p in prompts]
    by_slug = {s: i for i, s in enumerate(slugs)}

    dirs = sorted(p for p in run_dir.iterdir() if p.is_dir())
    if not dirs:
        raise SystemExit(
            f"no prompt subfolders under {run_dir}\n"
            f"expected one folder per prompt, seed images inside each")

    claimed, unmatched, problems = {}, [], []

    for d in dirs:
        name, idx, how = d.name, None, ""

        m = re.match(r"^(\d+)[_\-](.*)$", name)
        if m:
            cand, rest = int(m.group(1)), _norm(m.group(2))
            if cand < len(slugs) and rest == slugs[cand]:
                idx, how = cand, "prefix+slug"
            elif rest in by_slug:
                idx, how = by_slug[rest], "slug"
                problems.append(
                    f"{name}: folder index {cand} != prompt index {idx}; "
                    f"trusting the slug")
            elif cand < len(slugs):
                idx, how = cand, "prefix only"

        if idx is None:
            n = _norm(name)
            if n in by_slug:
                idx, how = by_slug[n], "slug"
            else:
                hits = [i for s, i in by_slug.items() if s in n or n in s]
                if len(hits) == 1:
                    idx, how = hits[0], "substring"
                elif len(hits) > 1:
                    problems.append(f"{name}: ambiguous, matches "
                                    f"{len(hits)} prompts")

        if idx is None:
            unmatched.append(name)
            continue
        if idx in claimed:
            raise SystemExit(f"two folders map to prompt {idx:03d}: "
                             f"{claimed[idx]['dir']} and {name}")

        files = sorted(f for f in d.iterdir()
                       if f.is_file() and f.suffix.lower() in IMG_EXT)
        if not files:
            problems.append(f"{name}: no image files")
            continue

        seeds, parsed_ok = [], True
        for k, f in enumerate(files):
            s, ok = _seed_of(f, k)
            parsed_ok &= ok
            seeds.append((s, f))
        if not parsed_ok:
            problems.append(
                f"{name}: seed not parseable from filenames; using sort order "
                f"(cross-config pairing then relies on identical naming in "
                f"both runs)")
        seeds.sort(key=lambda t: t[0])
        if len({s for s, _ in seeds}) != len(seeds):
            problems.append(f"{name}: duplicate seed numbers")

        claimed[idx] = dict(
            idx=idx, dir=name, path=d, how=how,
            noun=nouns[idx] if idx < len(nouns) else _norm(name),
            cls="variable" if idx < n_variable else "canonical",
            seeds=seeds[:n_seeds], n_found=len(seeds),
        )

    for name in unmatched:
        problems.append(f"{name}: no matching prompt in "
                        f"{Path(prompts_path).name}")
    for i in range(len(prompts)):
        if i not in claimed:
            problems.append(f"prompt {i:03d} ({slugs[i][:40]}): no folder found")
    for r in claimed.values():
        if r["n_found"] != n_seeds:
            problems.append(f"{r['dir']}: {r['n_found']} images, "
                            f"expected {n_seeds}")

    records = [claimed[i] for i in sorted(claimed)]

    if verbose:
        print(f"{run_dir}  ->  {len(records)}/{len(prompts)} prompts matched")
        for r in records:
            ss = [s for s, _ in r["seeds"]]
            span = f"{min(ss)}..{max(ss)}" if ss else "-"
            flag = "" if r["how"] == "prefix+slug" else "   <-- check"
            print(f"  {r['idx']:03d}  {r['noun']:<11s} {r['cls']:<9s} "
                  f"n={r['n_found']:<3d} seeds {span:<9s} "
                  f"[{r['how']}]  {r['dir'][:44]}{flag}")
        if problems:
            print(f"\n{len(problems)} problem(s):")
            for p in problems:
                print(f"  ! {p}")
        else:
            print("\nclean")

    return records, problems
