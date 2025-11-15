#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Direct Prompts → ComfyUI REST (HiDREAM template-preserving)
- Use a HiDREAM workflow JSON exported from the ComfyUI GUI as the TEMPLATE.
- Never modify template parameters (steps/cfg/sampler/scheduler/negative/loader/size).
- Only replace the positive prompt text and SaveImage.filename_prefix.
- Output filename pattern: {model_name}_{country}_{category}_{subcategory}_{variant}.png
"""

import re, time, json, shutil, requests
from pathlib import Path
from typing import Dict, List, Optional

# ========= USER SETTINGS =========
COMFY_HOST    = "http://127.0.0.1:8188"
TEMPLATE_PATH = "workflow/hidream_i1_dev.json"
OUT_DIR       = Path("outputs/t2i_hidream_base")
MODEL_NAME    = "hidream"

# Add test prompts here
PROMPTS = [
    {
        "text": "Architecture, photorealistic.",
        "category": "Architecture", "subcategory": "Base"
    },
    {
        "text": "Art, photorealistic.",
        "category": "Art", "subcategory": "Base"
    },
    {
        "text": "Event, photorealistic.",
        "category": "Event", "subcategory": "Base"
    },
    {
        "text": "Fashion, photorealistic.",
        "category": "Fashion", "subcategory": "Base"
    },
    {
        "text": "Food, photorealistic.",
        "category": "Food", "subcategory": "Base"
    },
    {
        "text": "Landscape, photorealistic.",
        "category": "Landscape", "subcategory": "Base"
    },
    {
        "text": "A person, photorealistic.",
        "category": "People", "subcategory": "Base"
    },
    {
        "text": "Wildlife, photorealistic.",
        "category": "Wildlife", "subcategory": "Base"
    },
]
# =================================

OUT_DIR.mkdir(parents=True, exist_ok=True)

def norm_token(x: str) -> str:
    s = str(x).strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_]+", "", s)
    if s in {"no_country","no-country","no__country","no","none"}:
        s = "nocountry"
    return s

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

def wait_history(prompt_id: str, timeout: float = 120.0, poll: float = 0.5) -> Optional[Dict]:
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
                # ComfyUI default output directory
                p = Path("output") / subfolder / filename
                if p.exists(): out.append(p)
                else:
                    q = Path(filename)
                    if q.exists(): out.append(q)
    except: pass
    return out

def build_from_template(prompt_text: str, out_prefix: Path, debug_once: List[bool]) -> Dict:
    """
    Keep template parameters as-is.
    - Replace text only for the CLIPTextEncode node whose _meta.title contains 'Positive Prompt'.
    - Replace only SaveImage.filename_prefix.
    """
    wf = json.loads(Path(TEMPLATE_PATH).read_text())

    for node in wf.values():
        ctype = node.get("class_type", "")
        inp   = node.get("inputs", {})
        meta  = node.get("_meta", {})
        title = (meta.get("title") or "").lower()

        # Replace only the positive prompt (keep negative prompt from the template)
        if ctype == "CLIPTextEncode" and "text" in inp:
            if "positive" in title:
                inp["text"] = prompt_text

        if ctype == "SaveImage":
            inp["filename_prefix"] = str(out_prefix)

    payload = {"prompt": wf}

    # One-time debug: confirm template values are passed through as-is
    if not debug_once[0]:
        try:
            ks = [n for n in wf.values() if n.get("class_type") == "KSampler"][0]
            sv = [n for n in wf.values() if n.get("class_type") == "SaveImage"][0]
            pos = [n for n in wf.values()
                   if n.get("class_type") == "CLIPTextEncode"
                   and (n.get("_meta", {}).get("title","").lower().find("positive")!=-1)][0]
            print("\n[DEBUG] Using HiDREAM template params as-is:")
            print("  KSampler (template):", {k: ks["inputs"][k] for k in ("steps","cfg","sampler_name","scheduler")})
            print("  SaveImage filename_prefix:", sv["inputs"]["filename_prefix"])
            print("  Positive text (sample):", pos["inputs"]["text"][:120])
        except Exception:
            pass
        debug_once[0] = True

    return payload

def main():
    debug_once = [False]

    for idx, item in enumerate(PROMPTS):
        text = item["text"]
        country = item.get("country","No Country")
        category = item.get("category","misc")
        subcat = item.get("subcategory","misc")
        variant = item.get("variant","general")

        country_tok  = norm_token(country)
        category_tok = norm_token(category)
        subcat_tok   = norm_token(subcat)
        variant_tok  = norm_token(variant)

        final_name = f"{MODEL_NAME}_{country_tok}_{category_tok}_{subcat_tok}_{variant_tok}.png"
        final_path = OUT_DIR / final_name
        prefix     = OUT_DIR / f"tmp_{MODEL_NAME}_{country_tok}_{category_tok}_{subcat_tok}_{variant_tok}"

        print(f"\n[{idx:04d}] {country} | {category} | {subcat} | {variant_tok}")
        print(f"→ {text}")

        payload = build_from_template(text, prefix, debug_once)

        try:
            resp = comfy_post(payload)
            pid  = resp.get("prompt_id","")
            hist = wait_history(pid)
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

        time.sleep(0.2)

    print("\n🎉 Done!")

if __name__ == "__main__":
    main()
