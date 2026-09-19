#!/usr/bin/env python3
"""
apply_patch.py
──────────────────────────────────────────────────────────────────────────────
Patches app.py in-place to replace DeepFace with face_recognition.

Run from the project root:
    python apply_patch.py

What it does
────────────
1. Replaces:
       from deepface import DeepFace
   with:
       import face_recognition

2. Replaces the DeepFace.verify() verification block with an equivalent
   face_recognition.compare_faces() block (same logic, no AVX dependency).

3. Creates app.py.bak before modifying anything.
──────────────────────────────────────────────────────────────────────────────
"""

import re
import shutil
import sys
from pathlib import Path

APP = Path("app.py")

if not APP.exists():
    print(f"[ERROR] {APP} not found. Run this script from the project root.")
    sys.exit(1)

# ── backup ────────────────────────────────────────────────────────────────────
bak = APP.with_suffix(".py.bak")
shutil.copy2(APP, bak)
print(f"[OK] Backup written → {bak}")

src = APP.read_text(encoding="utf-8")

# ── patch 1: import ───────────────────────────────────────────────────────────
OLD_IMPORT = "from deepface import DeepFace"
NEW_IMPORT = "import face_recognition  # replaces DeepFace (AVX-free)"

if OLD_IMPORT not in src:
    print("[WARN] 'from deepface import DeepFace' not found — import patch skipped.")
else:
    src = src.replace(OLD_IMPORT, NEW_IMPORT, 1)
    print("[OK] Import patched.")

# ── patch 2: DeepFace.verify() block ─────────────────────────────────────────
# Match the grayscale conversion + DeepFace.verify() call + if result['verified']:
# We use a regex so minor surrounding whitespace variations don't matter.

OLD_VERIFY = r"""            # Convert to grayscale for DeepFace
            captured_cv_gray = cv2\.cvtColor\(captured_cv_color, cv2\.COLOR_BGR2GRAY\)
            stored_cv_gray = cv2\.cvtColor\(stored_cv, cv2\.COLOR_BGR2GRAY\)

            # Perform face verification
            result = DeepFace\.verify\(captured_cv_gray, stored_cv_gray, enforce_detection=True\)
            print\(result\)
            
            if result\['verified'\]:"""

NEW_VERIFY = """            # face_recognition uses RGB (dlib backend, SSE4.1 — works on N4020)
            captured_rgb = cv2.cvtColor(captured_cv_color, cv2.COLOR_BGR2RGB)
            stored_rgb   = cv2.cvtColor(stored_cv,         cv2.COLOR_BGR2RGB)

            captured_encs = face_recognition.face_encodings(captured_rgb)
            stored_encs   = face_recognition.face_encodings(stored_rgb)

            if not captured_encs or not stored_encs:
                raise ValueError("No face detected in one or both images.")

            match  = face_recognition.compare_faces(
                [stored_encs[0]], captured_encs[0], tolerance=0.55
            )
            result = {"verified": match[0]}
            print(result)

            if result['verified']:"""

patched, n = re.subn(OLD_VERIFY, NEW_VERIFY, src, flags=re.MULTILINE)

if n == 0:
    # Fallback: try a simpler literal replacement for the verify line only
    SIMPLE_OLD = "result = DeepFace.verify(captured_cv_gray, stored_cv_gray, enforce_detection=True)"
    SIMPLE_NEW = (
        "# --- AVX-free replacement: face_recognition (dlib/SSE4.1) ---\n"
        "            captured_rgb  = cv2.cvtColor(captured_cv_color, cv2.COLOR_BGR2RGB)\n"
        "            stored_rgb    = cv2.cvtColor(stored_cv,          cv2.COLOR_BGR2RGB)\n"
        "            captured_encs = face_recognition.face_encodings(captured_rgb)\n"
        "            stored_encs   = face_recognition.face_encodings(stored_rgb)\n"
        "            if not captured_encs or not stored_encs:\n"
        "                raise ValueError('No face detected in one or both images.')\n"
        "            match  = face_recognition.compare_faces([stored_encs[0]], captured_encs[0], tolerance=0.55)\n"
        "            result = {'verified': match[0]}"
    )
    if SIMPLE_OLD in src:
        patched = src.replace(SIMPLE_OLD, SIMPLE_NEW, 1)
        # Also remove now-stale grayscale conversion lines
        patched = patched.replace(
            "            # Convert to grayscale for DeepFace\n"
            "            captured_cv_gray = cv2.cvtColor(captured_cv_color, cv2.COLOR_BGR2GRAY)\n"
            "            stored_cv_gray = cv2.cvtColor(stored_cv, cv2.COLOR_BGR2GRAY)\n\n"
            "            # Perform face verification\n",
            "            # Perform face verification (face_recognition / dlib)\n",
        )
        print("[OK] Verify block patched (fallback literal match).")
    else:
        print(
            "[WARN] DeepFace.verify() block not matched automatically.\n"
            "       Please apply the verify patch from PATCH_NOTES.md manually."
        )
        patched = src
else:
    print("[OK] Verify block patched (regex match).")

APP.write_text(patched, encoding="utf-8")
print(f"[DONE] {APP} updated. Original saved as {bak}")
print()
print("Next steps:")
print("  1.  pip uninstall -y tensorflow intel-tensorflow keras mediapipe deepface")
print("  2.  pip install -r requirements.txt")
print("  3.  python apply_patch.py   (already done)")
print("  4.  python app.py")
