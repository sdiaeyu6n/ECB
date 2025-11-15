#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HiDream i2i chained edit (Korea / United States / China) via ComfyUI REST.

- Scan originals from: outputs/t2i_hidream/hidream_*.png
- Output (5 chained steps) to: outputs/i2i_hidream/{stem}_{1..5}.png
- Template (HiDream): expects CLIPTextEncode(positive/negative), LoadImage, SaveImage, etc.

Template adaptation:
  * CLIPTextEncode(positive).inputs.text ← built prompt
  * LoadImage.inputs.image               ← <filename only> (we stage source into ComfyUI/input)
  * SaveImage.inputs.filename_prefix     ← OUT_DIR/<stem>_<k>  (extension added by Comfy)

Resume:
  * If outputs/i2i_hidream/{stem}_{k}.png exists, skip and continue from next step.
"""

import os, time, json, shutil, requests, re
from pathlib import Path
from typing import Dict, Optional, Tuple, List

# ---------- User settings ----------
COMFY_HOST        = os.environ.get("COMFY_HOST", "http://127.0.0.1:8124")
TEMPLATE_PATH     = "workflow/hidream_e1_1.json"      # HiDream template (JSON)
SCAN_DIR          = Path("outputs/t2i_hidream")     # Source images
OUT_DIR           = Path("outputs/i2i_hidream")     # Output directory
ALLOW_COUNTRIES   = {"india"}
NO_SUBCAT_CATS    = {"people", "landscape"}         # Categories without subcategory

# REST wait/poll
TIMEOUT_HISTORY   = 180.0
POLL_HISTORY      = 0.8
PAUSE_BETWEEN_JOBS= 0.2

# Debug
PRINT_PROMPTS     = True
DRY_RUN           = False   # If True, only print prompts without calling ComfyUI
# -----------------------------------

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- REST helpers ----------
def comfy_post(payload: Dict) -> Dict:
    r = requests.post(f"{COMFY_HOST}/prompt", json=payload)
    try:
        r.raise_for_status()
    except requests.HTTPError:
        # Printing raw server response is helpful for debugging 4xx errors
        print("---- ComfyUI response ----")
        try:
            print(r.text)
        except Exception:
            pass
        print("-------------------------")
        raise
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
                # ComfyUI default output directory
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
    max_n = max(1, min(len(parts) - 2, 4))
    for n in range(max_n, 0, -1):
        cand = "_".join(parts[:n])
        if cand in ALLOW_COUNTRIES:
            return cand, n
    return parts[0], 1

def parse_stem(stem: str) -> Tuple[str, str, str, Optional[str], str]:
    """
    Return: (model='hidream', country, category, subcategory_or_None, variant)
    Accepts:
      - hidream_{country}_{category}_{subcategory}_{variant}
      - hidream_{country}_{category}_{variant}  (people/landscape, etc.)
    """
    if not stem.startswith("hidream_"):
        raise ValueError("Model prefix must be 'hidream_'")
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
        variant = "_".join(tail)
    else:
        if len(tail) == 1:
            subcat = None
            variant = tail[0]
        else:
            variant = tail[-1]
            subcat = "_".join(tail[:-1])

    return ("hidream", country, category, subcat, variant)

def normalize_phrase(text: str) -> str:
    """
    Rule-based phrase normalization:
      - 'food modern staple' → 'modern staple food'
      - 'and groom bride' / 'groom and bride' / 'groom bride' / 'bride groom' → 'bride and groom'
      - 'life daily' → 'daily life'
      - Collapse multiple spaces
    """
    t = text

    def swap(pattern: str, repl: str):
        nonlocal t
        t = re.sub(rf"\b{pattern}\b", repl, t)

    swap(r"food modern staple", "modern staple food")
    swap(r"food staple", "staple food")
    swap(r"life daily", "daily life")
    t = re.sub(r"\band\s+groom\s+bride\b", "bride and groom", t)
    t = re.sub(r"\bgroom\s+and\s+bride\b", "bride and groom", t)
    t = re.sub(r"\bgroom\s+bride\b", "bride and groom", t)
    t = re.sub(r"\bbride\s+groom\b", "bride and groom", t)
    t = re.sub(r"\s{2,}", " ", t).strip()
    return t

def build_prompt(country: str, category: str, subcat: Optional[str], variant: str) -> str:
    """
    Rules:
      - default: "{variant} {subcategory} in {Country}"
      - variant == 'general' → "{subcategory} in {Country}."
      - people: no subcategory → "{variant} in {Country}."
      - wildlife: "{variant} animal in {Country}."
      - landscape: "{variant} landscape in {Country}."
      - united_states → "the United States"
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


# ---------- Template mutation (HiDream) ----------
def build_edit_payload(template_path: Path, prompt_text: str, load_image_name: str, out_prefix: Path) -> Dict:
    """
    - CLIPTextEncode(positive).inputs.text ← prompt_text
    - LoadImage.inputs.image               ← <filename only> (under ComfyUI/input)
    - SaveImage.inputs.filename_prefix     ← str(out_prefix)
    - DualCFGGuider.negative               ← force connection to Negative node (7)
    """
    wf = json.loads(Path(template_path).read_text(encoding="utf-8"))

    for node_id, node in wf.items():
        ctype = node.get("class_type", "")
        inp = node.get("inputs", {})

        # Positive prompt (id 6 in your JSON)
        if ctype == "CLIPTextEncode" and isinstance(inp.get("text"), str):
            # Keep template negative as-is (e.g., "low quality, blurry, distorted")
            if inp["text"].strip() != "low quality, blurry, distorted":
                inp["text"] = prompt_text

        # LoadImage: filename only (looked up under ComfyUI/input)
        if ctype == "LoadImage":
            inp["image"] = load_image_name

        # SaveImage: output prefix
        if ctype == "SaveImage":
            inp["filename_prefix"] = str(out_prefix)

        # Safety: fix graph connections
        if ctype == "InstructPixToPixConditioning":
            # 6: Positive, 7: Negative
            inp["positive"] = ["6", 0]
            inp["negative"] = ["7", 0]
        if ctype == "DualCFGGuider":
            # Some templates point negative to 6; force it to 7
            inp["negative"] = ["7", 0]

    return {"prompt": wf}


# ---------- I/O helpers ----------
def stage_into_comfy_input(src_img: Path) -> str:
    """
    The HiDream template uses LoadImage, so we copy the current input image
    into ComfyUI's ./input folder.

    Returns:
        The filename (string) to be passed to LoadImage.
    """
    input_dir = Path("input")
    input_dir.mkdir(parents=True, exist_ok=True)
    dst = input_dir / src_img.name
    if str(src_img.resolve()) != str(dst.resolve()):
        shutil.copy2(src_img, dst)
    return src_img.name  # Pass only filename to LoadImage

def run_one_step(prompt_text: str, source_img: Path, dest_path: Path) -> bool:
    # 1) Stage current input
    load_name = stage_into_comfy_input(source_img)

    # 2) SaveImage prefix (without extension)
    out_prefix = dest_path.with_suffix("")

    # 3) Build payload
    payload = build_edit_payload(Path(TEMPLATE_PATH), prompt_text, load_name, out_prefix)

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
    imgs = sorted(SCAN_DIR.glob("hidream_*.png"))
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
