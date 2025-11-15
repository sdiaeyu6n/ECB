#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSV → ComfyUI REST API (batch generation based on FLUX template)
- Use the FLUX workflow JSON (API format) exported from the GUI as the template
- Wildlife: National/Common/General, others: Traditional/Modern/General
- Output filename: {model_name}_{country}_{category}_{subcategory}_{variant}.png
"""

import os, re, time, json, random, shutil, requests, pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

# ========= USER SETTINGS =========
COMFY_HOST    = "http://127.0.0.1:8123"
TEMPLATE_PATH = "workflow/flux_schnell_workflow.json"   # Path to FLUX JSON exported from the GUI
CSV_PATH      = "script/your_prompt_csv.csv"  # Path to your prompt CSV file (change this to your own CSV)
OUT_DIR       = Path("outputs/t2i_flux")              # Output directory for final images
MODEL_NAME    = "flux"                                # Used as filename prefix

# Recommended FLUX hyperparameters (forced override)
FLUX_WIDTH, FLUX_HEIGHT = 1024, 1024
FLUX_STEPS   = 24
FLUX_CFG     = 3.5
FLUX_SAMPLER = "dpmpp_2m"
FLUX_SCHEDULER = "karras"
NEGATIVE = "cartoon, illustration, painting, blurry, low quality"  # If empty string, negative prompt is not used

# REST polling settings
TIMEOUT_HISTORY = 120.0
POLL_HISTORY    = 0.5
PAUSE_BETWEEN_JOBS = 0.2
# =================================

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Utilities ----------
def norm_token(x: str) -> str:
    s = str(x).strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_]+", "", s)
    if s in {"no_country", "no-country", "no__country", "no", "none"}:
        s = "nocountry"
    return s

def is_blank_prompt(v) -> bool:
    if not isinstance(v, str): return True
    t = v.strip()
    return t == "" or t in {"-", "—"}

def comfy_post(payload: Dict) -> Dict:
    r = requests.post(f"{COMFY_HOST}/prompt", json=payload)
    r.raise_for_status()
    return r.json()

def comfy_hist(prompt_id: str) -> Optional[Dict]:
    r = requests.get(f"{COMFY_HOST}/history/{prompt_id}")
    if r.status_code == 200:
        try: return r.json()
        except: return None
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
                filename = im.get("filename")
                subfolder = im.get("subfolder", "")
                if not filename: continue
                p = Path("output") / subfolder / filename  # ComfyUI default output directory
                if p.exists(): out.append(p)
                else:
                    q = Path(filename)
                    if q.exists(): out.append(q)
    except: pass
    return out

# ---------- Template mutation ----------
def build_from_template(prompt_text: str, out_prefix: Path, debug_once: List[bool]) -> Dict:
    wf = json.loads(Path(TEMPLATE_PATH).read_text())

    for node in wf.values():
        ctype = node.get("class_type", "")
        inp   = node.get("inputs", {})

        # 1) Replace only positive prompt: CLIPTextEncode nodes whose text is not empty
        if ctype == "CLIPTextEncode" and isinstance(inp.get("text"), str):
            if inp["text"].strip() != "":          # keep template's negative prompt (empty string) unchanged
                inp["text"] = prompt_text

        # 2) Replace only SaveImage filename_prefix
        if ctype == "SaveImage":
            inp["filename_prefix"] = str(out_prefix)

    payload = {"prompt": wf}

    # One-time debug: sampler/steps/cfg submitted must match template values
    if not debug_once[0]:
        try:
            ks = [n for n in wf.values() if n.get("class_type") == "KSampler"][0]
            sv = [n for n in wf.values() if n.get("class_type") == "SaveImage"][0]
            pos = [n for n in wf.values() if n.get("class_type") == "CLIPTextEncode" and n.get("inputs",{}).get("text","").strip()!=""][0]
            print("\n[DEBUG] Using template params as-is:")
            print("  KSampler (template):", {k: ks["inputs"][k] for k in ("steps","cfg","sampler_name","scheduler")})
            print("  SaveImage filename_prefix:", sv["inputs"]["filename_prefix"])
            print("  Positive text (sample):", pos["inputs"]["text"][:120])
        except Exception:
            pass
        debug_once[0] = True

    return payload

# ---------- Column policy ----------
DEFAULT_COLS  = ["Traditional Prompt", "Modern Prompt", "General Prompt"]
WILDLIFE_COLS = ["National Prompt", "Common Prompt", "General Prompt"]

def pick_cols(category: str, headers: List[str]) -> List[str]:
    c = (category or "").strip().lower()
    cols = WILDLIFE_COLS if c == "wildlife" else DEFAULT_COLS
    return [x for x in cols if x in headers]

def col_to_variant(category: str, col_name: str) -> str:
    base = col_name.replace(" Prompt","").strip().lower()
    if (category or "").strip().lower() == "wildlife":
        return {"national":"national","common":"common","general":"general"}.get(base, base)
    return {"traditional":"traditional","modern":"modern","general":"general","generic":"general"}.get(base, base)

# ---------- Main ----------
def main():
    df = pd.read_csv(CSV_PATH)
    print(f"✅ Loaded CSV: {CSV_PATH}  ({len(df)} rows)")

    current_category = ""
    current_subcat   = ""
    debug_once = [False]

    for i, row in df.iterrows():
        if pd.notna(row.get("Category")):    current_category = str(row["Category"]).strip()
        if pd.notna(row.get("Subcategory")): current_subcat   = str(row["Subcategory"]).strip()
        country = row.get("Country")
        if pd.isna(country): continue

        country_tok   = norm_token(country)
        category_tok  = norm_token(current_category)
        subcat_tok    = norm_token(current_subcat)

        cols = pick_cols(current_category, list(df.columns))
        for col in cols:
            prompt = row.get(col, "")
            if is_blank_prompt(prompt): continue

            variant = col_to_variant(current_category, col)

            # === Build file paths ===
            final_name = f"{MODEL_NAME}_{country_tok}_{category_tok}_{subcat_tok}_{variant}.png"
            final_path = OUT_DIR / final_name
            prefix     = OUT_DIR / f"tmp_{MODEL_NAME}_{country_tok}_{category_tok}_{subcat_tok}_{variant}"

            # === Skip if file already exists ===
            if final_path.exists():
                print(f"⏭️  Skip (exists): {final_path}")
                continue

            print(f"\n[{i:04d}] {country} | {current_category} | {current_subcat} | {variant}")
            print(f"→ {prompt}")

            payload = build_from_template(str(prompt), prefix, debug_once)
            try:
                resp = comfy_post(payload)
                pid  = resp.get("prompt_id","")
                hist = wait_history(pid, TIMEOUT_HISTORY, POLL_HISTORY)
                files = extract_saved_files(hist) if hist else []
                if files:
                    saved = sorted(files, key=lambda p: p.stat().st_mtime)[-1]
                    final_path.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        shutil.move(str(saved), str(final_path))
                    except Exception:
                        shutil.copy2(str(saved), str(final_path))
                        try: saved.unlink()
                        except: pass
                    print(f"✅ Saved: {final_path}")
                else:
                    print("⚠️ No saved file detected (check ComfyUI/output).")
            except Exception as e:
                print(f"❌ Error: {e}")

            time.sleep(PAUSE_BETWEEN_JOBS)

    print("\n🎉 Done!")


if __name__ == "__main__":
    main()