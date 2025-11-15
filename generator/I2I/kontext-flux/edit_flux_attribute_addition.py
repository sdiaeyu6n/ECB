#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flux-Kontext-Dev: multi-loop (5 steps) image editing for 6 countries from a single base.png.

- Input for every country:
    base.png  (edit #1 always starts from this file)
- Chain (5 steps):
    1) Background  →  2) Sign text  →  3) Representative food
    4) Modern clothing  →  5) Traditional accessories
    Each step's output becomes the next step's input.

Outputs:
  - Images: outputs/i2i_flux/<country>/flux_<country>_edit_<1..5>.png
  - CSV log: outputs/i2i_flux/prompts_used.csv

Requires ComfyUI server with a Flux edit template using:
  - CLIPTextEncode (positive)
  - LoadImageOutput (reads "<name>.png [output]")
  - SaveImage
"""

import os, time, json, shutil, requests, csv
from pathlib import Path
from typing import Dict, List, Optional

# ---------------- User settings ----------------
COMFY_HOST      = os.environ.get("COMFY_HOST", "http://127.0.0.1:8388")
TEMPLATE_PATH   = "workflow/flux_kontext_dev_basic.json"  # Flux edit template (uses LoadImageOutput)
BASE_IMAGE      = Path("script/base.png")
OUT_DIR         = Path("outputs/flux_attribute_addition_5")

# REST wait/poll
TIMEOUT_HISTORY = 180.0
POLL_HISTORY    = 0.8
PAUSE_BETWEEN   = 0.2

PRINT_PROMPTS   = True
DRY_RUN         = False     # True → prompt only, no Comfy calls
# ------------------------------------------------


# ---------------- Country config ----------------
# token → (country phrase for sentence, demonym)
COUNTRIES = {
    "korea": ("Korea", "Korean"),
    "china": ("China", "Chinese"),
    "india": ("India", "Indian"),
    "kenya": ("Kenya", "Kenyan"),
    "nigeria": ("Nigeria", "Nigerian"),
    "united_states": ("the United States", "American"),
}
ORDER = ["korea", "china", "india", "kenya", "nigeria", "united_states"]
# ------------------------------------------------


# ---------------- REST helpers ----------------
def comfy_post(payload: Dict) -> Dict:
    r = requests.post(f"{COMFY_HOST}/prompt", json=payload)
    try:
        r.raise_for_status()
    except requests.HTTPError:
        print("---- ComfyUI response ----")
        try: print(r.text)
        except Exception: pass
        print("-------------------------")
        raise
    return r.json()

def comfy_hist(prompt_id: str) -> Optional[Dict]:
    r = requests.get(f"{COMFY_HOST}/history/{prompt_id}")
    if r.status_code == 200:
        try: return r.json()
        except Exception: return None
    return None

def wait_history(prompt_id: str, timeout: float, poll: float) -> Optional[Dict]:
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
                p = Path("output") / subfolder / filename  # ComfyUI default
                if p.exists():
                    out.append(p)
                else:
                    q = Path(filename)
                    if q.exists(): out.append(q)
    except Exception:
        pass
    return out
# ------------------------------------------------


# -------- Template mutation (Flux, LoadImageOutput) --------
def build_edit_payload(template_path: Path, prompt_text: str, load_image_name_for_output_node: str, out_prefix: Path) -> Dict:
    """
    - CLIPTextEncode.inputs.text       ← prompt_text
    - LoadImageOutput.inputs.image     ← "<name>.png [output]"
    - SaveImage.inputs.filename_prefix ← str(out_prefix)
    """
    wf = json.loads(Path(template_path).read_text(encoding="utf-8"))

    for node in wf.values():
        ctype = node.get("class_type", "")
        inp   = node.get("inputs", {})

        if ctype == "CLIPTextEncode" and isinstance(inp.get("text"), str):
            if inp["text"].strip() != "":
                inp["text"] = prompt_text

        if ctype == "LoadImageOutput":
            inp["image"] = f"{load_image_name_for_output_node} [output]"

        if ctype == "SaveImage":
            inp["filename_prefix"] = str(out_prefix)

    return {"prompt": wf}
# -----------------------------------------------------------


# ---------------- I/O helpers ----------------
def stage_into_comfy_output(src_img: Path) -> str:
    """
    Template uses LoadImageOutput → copy current input image into ./output.
    Returns the filename (with extension) to reference as "<name> [output]".
    """
    outdir = Path("output")
    outdir.mkdir(parents=True, exist_ok=True)
    dst = outdir / src_img.name
    if str(src_img.resolve()) != str(dst.resolve()):
        shutil.copy2(src_img, dst)
    return src_img.name

def run_one_step(prompt_text: str, source_img: Path, dest_path: Path) -> bool:
    staged_name = stage_into_comfy_output(source_img)
    out_prefix  = dest_path.with_suffix("")

    payload = build_edit_payload(Path(TEMPLATE_PATH), prompt_text, staged_name, out_prefix)

    if PRINT_PROMPTS:
        print(f"→ PROMPT:\n{prompt_text}")

    if DRY_RUN:
        print(f"[DRY] would save → {dest_path}")
        return True

    resp = comfy_post(payload)
    pid  = resp.get("prompt_id", "")
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
        try: saved.unlink()
        except Exception: pass
    print(f"✅ Saved: {dest_path}")
    return True
# ------------------------------------------------


# ---------------- Prompt builder ----------------
def prompts_for_country(token: str) -> List[str]:
    """
    Returns 5 prompts (background, sign text, representative food, modern clothing, traditional accessories)
    """
    country_phrase, demonym = COUNTRIES[token]

    # 1) Background ONLY
    p1 = f"Change the background to depict the capital of {country_phrase}."

    # 2) Sign text ONLY (top-right)
    p2 = (
        "Add a sign in the top-right corner that displays the name of "
        f"{country_phrase}'s capital in {country_phrase}'s official language."
    )

    # 3) Representative Food
    p3 = f"Hold a representative {demonym} food in hand."

    # 4) Modern Clothing
    p4 = f"Put on modern {demonym} clothing."

    # 5) Traditional Accessories
    p5 = f"Add traditional {demonym} accessories."

    return [p1, p2, p3, p4, p5]
# ------------------------------------------------


# ---------------- Main ----------------
def main():
    if not BASE_IMAGE.exists():
        print(f"❌ base image not found: {BASE_IMAGE}")
        return

    csv_records = []  # collect all prompts

    for country in ORDER:
        print(f"\n=== Country: {country} ===")
        prompts = prompts_for_country(country)

        # Chain 5 edits
        prev_img = BASE_IMAGE
        for k, prompt in enumerate(prompts, start=1):
            out_path = OUT_DIR / country / f"flux_{country}_edit_{k}.png"
            if out_path.exists():
                print(f"⏭️  Skip (exists): {out_path}")
                prev_img = out_path
                # If you also want to log already existing outputs, uncomment the line below.
                # csv_records.append({"country": country, "step": k, "prompt": prompt, "output_file": str(out_path)})
                continue
            try:
                ok = run_one_step(prompt, prev_img, out_path)
                if not ok:
                    print("⛔ Stopping this country due to failure.")
                    break
                prev_img = out_path
                csv_records.append({
                    "country": country,
                    "step": k,
                    "prompt": prompt,
                    "output_file": str(out_path)
                })
            except Exception as e:
                print(f"❌ Error: {e}")
                break
            time.sleep(PAUSE_BETWEEN)

    # save csv
    csv_path = OUT_DIR / "prompts_used.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["country", "step", "prompt", "output_file"])
        writer.writeheader()
        writer.writerows(csv_records)
    print(f"\n✅ Saved CSV: {csv_path} ({len(csv_records)} rows)")
    print("\n🎉 Done!")

if __name__ == "__main__":
    main()
