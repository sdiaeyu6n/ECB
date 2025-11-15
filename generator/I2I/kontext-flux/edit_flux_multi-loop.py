#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Edit existing T2I images with 5-step chained edits via ComfyUI REST.

- Scan input images from: outputs/t2i_flux/flux_*.png
- Expected filename patterns:
    1) flux_{country}_{category}_{subcategory}_{variant}.png
    2) flux_{country}_{category}_{variant}.png        # e.g., people, landscape (no subcategory)
- Build prompts from filename components and run 5 iterative edits:
    output: {original_stem}_{1..5}.png

Template requirements (from your ComfyUI JSON):
- Must contain CLIPTextEncode (positive), LoadImageOutput, and SaveImage nodes.
- This script sets:
    * CLIPTextEncode.inputs.text       ← built prompt
    * LoadImageOutput.inputs.image     ← "<name>.png [output]"
    * SaveImage.inputs.filename_prefix ← destination path without extension

Note: Because the template uses LoadImageOutput, each step copies the current source
image into ComfyUI's ./output folder before running the workflow.
"""

import os, time, json, shutil, requests, re
from pathlib import Path
from typing import Dict, Optional, Tuple, List

# ---------- User settings ----------
# ComfyUI server endpoint (change the port/host if your ComfyUI runs elsewhere)
COMFY_HOST        = os.environ.get("COMFY_HOST", "http://127.0.0.1:8125")

# Path to the ComfyUI workflow JSON template
TEMPLATE_PATH     = "workflow/flux_kontext_dev_basic.json"

# Where to scan for input images and where to save edited images
SCAN_DIR          = Path("outputs/t2i_flux")
OUT_DIR           = Path("outputs/i2i_flux")

# Countries allowed for processing (match tokens in the filename)
# e.g., {"korea", "united_states", "china"}
ALLOW_COUNTRIES   = {"korea"}

# Categories that do not use a subcategory in the filename
NO_SUBCAT_CATS    = {"people", "landscape"}

# REST wait/poll
TIMEOUT_HISTORY   = 180.0
POLL_HISTORY      = 0.8
PAUSE_BETWEEN_JOBS= 0.2

# Debug
PRINT_PROMPTS     = True
DRY_RUN           = False   # If True, only print prompts without calling ComfyUI
# -----------------------------------


# ---------- REST helpers ----------
def comfy_post(payload: Dict) -> Dict:
    r = requests.post(f"{COMFY_HOST}/prompt", json=payload)
    r.raise_for_status()
    return r.json()

def comfy_hist(prompt_id: str) -> Optional[Dict]:
    r = requests.get(f"{COMFY_HOST}/history/{prompt_id}")
    if r.status_code == 200:
        try:
            return r.json()
        except Exception:
            return None
    return None

def wait_history(prompt_id: str, timeout: float, poll: float) -> Optional[Dict]:
    t0 = time.time()
    while True:
        h = comfy_hist(prompt_id)
        if h:
            return h
        if time.time() - t0 > timeout:
            return None
        time.sleep(poll)

def extract_saved_files(history: Dict) -> List[Path]:
    out = []
    try:
        pid = next(iter(history.keys()))
        outputs = history[pid].get("outputs", {})
        for node_out in outputs.values():
            for im in node_out.get("images", []):
                filename = im.get("filename")
                subfolder = im.get("subfolder", "")
                if not filename:
                    continue
                # ComfyUI default output folder
                p = Path("output") / subfolder / filename
                if p.exists():
                    out.append(p)
                else:
                    q = Path(filename)
                    if q.exists():
                        out.append(q)
    except Exception:
        pass
    return out


# ---------- Filename parsing & prompt rules ----------
def _country_phrase(country_tok: str) -> str:
    if country_tok == "united_states":
        return "the United States"
    return country_tok.replace("_", " ").title()

def _nice(x: Optional[str]) -> Optional[str]:
    if x is None:
        return None
    return x.replace("_", " ").lower()

def _match_country_prefix(parts: List[str]) -> Tuple[str, int]:
    """
    Given a list of tokens (parts) from the filename, find the longest prefix
    (joined by "_") that matches one of ALLOW_COUNTRIES.

    - Returns (country_token, n) where country_token = "_".join(parts[:n])
    - n is at least 1 and at most len(parts) - 2 (safe upper bound).
    """
    max_n = max(1, min(len(parts) - 2, 4))  # safety upper bound 4
    for n in range(max_n, 0, -1):
        cand = "_".join(parts[:n])
        if cand in ALLOW_COUNTRIES:
            return cand, n
    # Fallback: treat the first token as country if nothing matches
    return parts[0], 1

def parse_stem(stem: str) -> Tuple[str, str, str, Optional[str], str]:
    """
    Parse filename stem into (model='flux', country, category, subcategory_or_None, variant).

    Rules:
      - Country: longest prefix of tokens (after "flux_") that matches ALLOW_COUNTRIES.
      - For categories in NO_SUBCAT_CATS (e.g., people, landscape):
          subcategory = None, variant = all remaining tokens joined by "_".
      - For other categories:
          * If only one token remains: subcategory = None, variant = that token.
          * If more than one token remains:
              - variant = last token
              - subcategory = all middle tokens joined by "_".
    """
    if not stem.startswith("flux_"):
        raise ValueError("Model prefix must be 'flux_'")
    parts = stem.split("_")[1:]
    if len(parts) < 3:
        raise ValueError(f"Not enough parts: {stem}")

    country, n = _match_country_prefix(parts)
    remain = parts[n:]
    if len(remain) < 2:
        raise ValueError(f"Not enough parts after country: {stem}")

    category = remain[0]
    tail = remain[1:]

    if _nice(category) in NO_SUBCAT_CATS:
        subcat = None
        variant = "_".join(tail)  # people/landscape: variant may contain multiple tokens
    else:
        if len(tail) == 1:
            subcat = None
            variant = tail[0]
        else:
            variant = tail[-1]
            subcat = "_".join(tail[:-1])

    return ("flux", country, category, subcat, variant)

def normalize_phrase(text: str) -> str:
    """
    Heuristic phrase normalization:
      - 'food modern staple' → 'modern staple food'
      - 'food staple' → 'staple food'
      - 'life daily' → 'daily life'
      - Various permutations of 'bride/groom' → 'bride and groom'
      - Collapse multiple spaces
    """
    t = text

    def swap(pattern: str, repl: str):
        nonlocal t
        t = re.sub(rf"\b{pattern}\b", repl, t)

    # Compound nouns / idiomatic expressions
    swap(r"food modern staple", "modern staple food")
    swap(r"food staple", "staple food")
    swap(r"life daily", "daily life")

    # bride/groom normalizations
    t = re.sub(r"\band\s+groom\s+bride\b", "bride and groom", t)
    t = re.sub(r"\bgroom\s+and\s+bride\b", "bride and groom", t)
    t = re.sub(r"\bgroom\s+bride\b", "bride and groom", t)
    t = re.sub(r"\bbride\s+groom\b", "bride and groom", t)

    # Remove multiple spaces
    t = re.sub(r"\s{2,}", " ", t).strip()
    return t

def build_prompt(country: str, category: str, subcat: Optional[str], variant: str) -> str:
    """
    Prompt construction rules:

      - Base template: "Change the image to represent {head} in {Country}."

      Where {head} is determined as:
        - If category == 'wildlife':
              head = "{variant} animal"
        - If category == 'people' and subcategory is None:
              head = "{variant}"
        - If category == 'landscape' and subcategory is None:
              head = "{variant} landscape"
        - Else if variant == 'general':
              head = subcategory if present, otherwise the category
        - Else:
              head = "{variant} {subcategory}" if subcategory present,
                     otherwise "{variant} {category}"

      - 'united_states' is rendered as 'the United States'
      - Phrases are further normalized by normalize_phrase.
    """
    C = _country_phrase(country)
    cat, sub, var = _nice(category), _nice(subcat), _nice(variant)

    if cat == "wildlife":
        head = f"{var} animal"
    elif cat == "people" and sub is None:
        head = f"{var}"
    elif cat == "landscape" and sub is None:
        head = f"{var} landscape"
    elif var == "general":
        head = sub if sub else cat
    else:
        head = f"{var} {sub}" if sub else f"{var} {cat}"

    head = normalize_phrase(head)
    return f"Change the image to represent {head} in {C}."


# ---------- Template mutation ----------
def build_edit_payload(template_path: Path, prompt_text: str, load_image_name_for_output_node: str, out_prefix: Path) -> Dict:
    """
    Prepare a ComfyUI prompt payload from the template workflow:

    - CLIPTextEncode.inputs.text       ← prompt_text
    - LoadImageOutput.inputs.image     ← "<name>.png [output]"
    - SaveImage.inputs.filename_prefix ← out_prefix (as string)
    """
    wf = json.loads(Path(template_path).read_text(encoding="utf-8"))

    for node in wf.values():
        ctype = node.get("class_type", "")
        inp = node.get("inputs", {})

        # Positive prompt
        if ctype == "CLIPTextEncode" and isinstance(inp.get("text"), str):
            if inp["text"].strip() != "":
                inp["text"] = prompt_text

        # Load from ComfyUI/output
        if ctype == "LoadImageOutput":
            inp["image"] = f"{load_image_name_for_output_node} [output]"

        # Save
        if ctype == "SaveImage":
            inp["filename_prefix"] = str(out_prefix)

    return {"prompt": wf}


# ---------- I/O helpers ----------
def stage_into_comfy_output(src_img: Path) -> str:
    """
    Since the template uses LoadImageOutput, copy the current input image into
    ComfyUI's ./output folder before running the workflow.

    Returns:
        The filename (with extension) that LoadImageOutput should reference.
    """
    outdir = Path("output")
    outdir.mkdir(parents=True, exist_ok=True)
    dst = outdir / src_img.name
    if str(src_img.resolve()) != str(dst.resolve()):
        shutil.copy2(src_img, dst)
    # LoadImageOutput will use "<name>.png [output]" with this name.
    return src_img.name

def run_one_step(prompt_text: str, source_img: Path, dest_path: Path) -> bool:
    # 1) Stage the current source image into ComfyUI/output
    staged_name = stage_into_comfy_output(source_img)

    # 2) SaveImage prefix (without extension)
    out_prefix = dest_path.with_suffix("")

    # 3) Build payload
    payload = build_edit_payload(Path(TEMPLATE_PATH), prompt_text, staged_name, out_prefix)

    # 4) POST & wait
    resp = comfy_post(payload)
    pid = resp.get("prompt_id", "")
    hist = wait_history(pid, TIMEOUT_HISTORY, POLL_HISTORY)
    files = extract_saved_files(hist) if hist else []
    if not files:
        print("⚠️ No saved file detected (check ComfyUI/output).")
        return False

    saved = sorted(files, key=lambda p: p.stat().st_mtime)[-1]
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.move(str(saved), str(dest_path))
    except Exception:
        shutil.copy2(str(saved), str(dest_path))
        try:
            saved.unlink()
        except Exception:
            pass
    print(f"✅ Saved: {dest_path}")
    return True


# ---------- Main ----------
def main():
    imgs = sorted(SCAN_DIR.glob("flux_*.png"))
    if not imgs:
        print(f"⚠️ No images in {SCAN_DIR}")
        return

    for img in imgs:
        stem = img.stem
        try:
            _, country, category, subcat, variant = parse_stem(stem)
        except Exception as e:
            print(f"⏭️  Skip (unparsable): {img.name}  ({e})")
            continue

        if country not in ALLOW_COUNTRIES:
            print(f"⏭️  Skip (country not allowed): {img.name}")
            continue

        prompt = build_prompt(country, category, subcat, variant)
        if PRINT_PROMPTS:
            print(f"\n[FILE] {img.name}\n→ PROMPT: {prompt}")

        if DRY_RUN:
            continue

        prev = img
        for k in range(1, 6):
            out_path = OUT_DIR / f"{stem}_{k}.png"
            if out_path.exists():
                print(f"⏭️  Skip (exists): {out_path.name}")
                prev = out_path
                continue
            try:
                ok = run_one_step(prompt, prev, out_path)
                if not ok:
                    break
                prev = out_path
            except Exception as e:
                print(f"❌ Error: {e}")
                break
            time.sleep(PAUSE_BETWEEN_JOBS)

    print("\n🎉 Done!")

if __name__ == "__main__":
    main()
