#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HiDream i2i chained edit for ONE specific image via ComfyUI REST.

- Target original:
    /home/sieunch0i/BIG/ComfyUI/outputs/t2i_hidream/hidream_india_architecture_landmark_general.png
- Output (5 chained steps):
    outputs/india_landmark/hidream_india_architecture_landmark_general_{1..5}.png

Template (HiDream) must have:
  - CLIPTextEncode(positive/negative), LoadImage, SaveImage, etc.
We adapt:
  * CLIPTextEncode(positive).text ← built prompt
  * LoadImage.image              ← <filename only> (staged to ComfyUI/input)
  * SaveImage.filename_prefix    ← OUT_DIR/<stem>_<k>

Resume:
  * If {stem}_{k}.png exists, skip and continue.
"""

import os, time, json, shutil, requests, re
from pathlib import Path
from typing import Dict, Optional, Tuple, List

# ---------- User settings ----------
COMFY_HOST        = os.environ.get("COMFY_HOST", "http://127.0.0.1:8288")
TEMPLATE_PATH     = "workflow/hidream_e1_1.json"
TARGET_IMG        = Path("path/to/your/target_image.png")
OUT_DIR           = Path("outputs/india_house")
ALLOW_COUNTRIES   = {"india"}
NO_SUBCAT_CATS    = {"people", "landscape"}

# REST wait/poll
TIMEOUT_HISTORY   = 180.0
POLL_HISTORY      = 0.8
PAUSE_BETWEEN_JOBS= 0.2

# Debug
PRINT_PROMPTS     = True
DRY_RUN           = False   # True → just print prompts, no REST call
# -----------------------------------

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- REST helpers ----------
def comfy_post(payload: Dict) -> Dict:
    r = requests.post(f"{COMFY_HOST}/prompt", json=payload)
    try:
        r.raise_for_status()
    except requests.HTTPError:
        print("---- ComfyUI response ----")
        try: print(r.text)
        except: pass
        print("-------------------------")
        raise
    return r.json()

def comfy_hist(prompt_id: str):
    r = requests.get(f"{COMFY_HOST}/history/{prompt_id}")
    if r.status_code == 200:
        try: return r.json()
        except: return None
    return None

def wait_history(prompt_id: str, timeout: float, poll: float):
    t0 = time.time()
    while True:
        h = comfy_hist(prompt_id)
        if h: return h
        if time.time() - t0 > timeout: return None
        time.sleep(poll)

def extract_saved_files(history: Dict) -> List[Path]:
    out = []
    try:
        pid = next(iter(history.keys()))
        outputs = history[pid].get("outputs", {})
        for node_out in outputs.values():
            for im in node_out.get("images", []):
                filename = im.get("filename"); subfolder = im.get("subfolder", "")
                if not filename: continue
                p = Path("output") / subfolder / filename
                if p.exists(): out.append(p)
                else:
                    q = Path(filename)
                    if q.exists(): out.append(q)
    except: pass
    return out

# ---------- Prompt rules ----------
def _country_phrase(country_tok: str) -> str:
    if country_tok == "united_states": return "the United States"
    return country_tok.replace("_", " ").title()

def _nice(x: Optional[str]) -> Optional[str]:
    if x is None: return None
    return x.replace("_", " ").lower()

def normalize_phrase(text: str) -> str:
    t = text
    def swap(pat, rep):
        nonlocal t; t = re.sub(rf"\b{pat}\b", rep, t)
    swap(r"food modern staple", "modern staple food")
    swap(r"food staple", "staple food")
    swap(r"life daily", "daily life")
    t = re.sub(r"\band\s+groom\s+bride\b", "bride and groom", t)
    t = re.sub(r"\bgroom\s+and\s+bride\b", "bride and groom", t)
    t = re.sub(r"\bgroom\s+bride\b", "bride and groom", t)
    t = re.sub(r"\bbride\s+groom\b", "bride and groom", t)
    return re.sub(r"\s{2,}", " ", t).strip()

def build_prompt(country: str, category: str, subcat: Optional[str], variant: str) -> str:
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

# ---------- Filename parsing ----------
def _match_country_prefix(parts: List[str], allow: set) -> Tuple[str, int]:
    max_n = max(1, min(len(parts) - 2, 4))
    for n in range(max_n, 0, -1):
        cand = "_".join(parts[:n])
        if cand in allow: return cand, n
    return parts[0], 1

def parse_stem(stem: str) -> Tuple[str, str, str, Optional[str], str]:
    """
    stem like: hidream_india_architecture_landmark_general
    return: ('hidream', country, category, subcategory_or_None, variant)
    """
    if not stem.startswith("hidream_"):
        raise ValueError("Model prefix must be 'hidream_'")
    parts = stem.split("_")[1:]
    if len(parts) < 3: raise ValueError(f"Not enough parts: {stem}")

    country, n = _match_country_prefix(parts, ALLOW_COUNTRIES)
    remain = parts[n:]
    if len(remain) < 2: raise ValueError(f"Not enough parts after country: {stem}")

    category = remain[0]
    tail = remain[1:]

    if _nice(category) in NO_SUBCAT_CATS:
        subcat = None
        variant = "_".join(tail)
    else:
        if len(tail) == 1:
            subcat = None; variant = tail[0]
        else:
            variant = tail[-1]; subcat = "_".join(tail[:-1])

    return ("hidream", country, category, subcat, variant)

# ---------- Template mutation (HiDream) ----------
def build_edit_payload(template_path: Path, prompt_text: str, load_image_name: str, out_prefix: Path) -> Dict:
    wf = json.loads(Path(template_path).read_text(encoding="utf-8"))
    for _, node in wf.items():
        ctype = node.get("class_type", ""); inp = node.get("inputs", {})
        if ctype == "CLIPTextEncode" and isinstance(inp.get("text"), str):
            if inp["text"].strip() != "low quality, blurry, distorted":
                inp["text"] = prompt_text
        if ctype == "LoadImage":
            inp["image"] = load_image_name  # filename only (in ComfyUI/input)
        if ctype == "SaveImage":
            inp["filename_prefix"] = str(out_prefix)
        if ctype == "InstructPixToPixConditioning":
            inp["positive"] = ["6", 0]; inp["negative"] = ["7", 0]
        if ctype == "DualCFGGuider":
            inp["negative"] = ["7", 0]
    return {"prompt": wf}

# ---------- I/O helpers ----------
def stage_into_comfy_input(src_img: Path) -> str:
    input_dir = Path("input"); input_dir.mkdir(parents=True, exist_ok=True)
    dst = input_dir / src_img.name
    if str(src_img.resolve()) != str(dst.resolve()):
        shutil.copy2(src_img, dst)
    return src_img.name  # filename only

def run_one_step(prompt_text: str, source_img: Path, dest_path: Path) -> bool:
    load_name = stage_into_comfy_input(source_img)
    out_prefix = dest_path.with_suffix("")
    payload = build_edit_payload(Path(TEMPLATE_PATH), prompt_text, load_name, out_prefix)
    resp = comfy_post(payload)
    pid = resp.get("prompt_id", "")
    hist = wait_history(pid, TIMEOUT_HISTORY, POLL_HISTORY)
    files = extract_saved_files(hist) if hist else []
    if not files:
        print("⚠️ No saved file detected (check ComfyUI/output)."); return False
    saved = sorted(files, key=lambda p: p.stat().st_mtime)[-1]
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    try: shutil.move(str(saved), str(dest_path))
    except Exception:
        shutil.copy2(str(saved), str(dest_path))
        try: saved.unlink()
        except: pass
    print(f"✅ Saved: {dest_path}")
    return True

# ---------- Main (single target) ----------
def main():
    img = TARGET_IMG
    if not img.exists():
        print(f"❌ Target not found: {img}"); return

    stem = img.stem
    try:
        _, country, category, subcat, variant = parse_stem(stem)
    except Exception as e:
        print(f"⏭️  Skip (unparsable): {img.name}  ({e})"); return

    if country not in ALLOW_COUNTRIES:
        print(f"⏭️  Skip (country not allowed): {img.name}"); return

    prompt = build_prompt(country, category, subcat, variant)
    if PRINT_PROMPTS:
        print(f"\n[FILE] {img.name}\n→ PROMPT: {prompt}")

    if DRY_RUN:
        print("DRY_RUN=True, skipping REST calls."); return

    prev = img
    for k in range(1, 6):
        out_path = OUT_DIR / f"{stem}_{k}.png"
        if out_path.exists():
            print(f"⏭️  Skip (exists): {out_path.name}")
            prev = out_path
            continue
        try:
            ok = run_one_step(prompt, prev, out_path)
            if not ok: break
            prev = out_path
        except Exception as e:
            print(f"❌ Error: {e}"); break
        time.sleep(PAUSE_BETWEEN_JOBS)

    print("\n🎉 Done!")

if __name__ == "__main__":
    main()
