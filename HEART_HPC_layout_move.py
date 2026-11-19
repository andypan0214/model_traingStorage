#!/usr/bin/env python3
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from datetime import datetime

NOTEBOOK_CATEGORIES = {
    "general": [
        r"General_Pretraining_.*\.ipynb$",
        r"HEART_H200_.*\.ipynb$",
        r".*Pretraining.*\.ipynb$",
    ],
    "ugi": [
        r"HEART_Ugi_.*\.ipynb$",
        r".*Priority1.*\.ipynb$",
    ],
    "lnpdb": [
        r"HEART_LNPDB_.*\.ipynb$",
        r".*Priority2.*\.ipynb$",
    ],
    "chemistry": [
        r"HEART_Chemistry_.*\.ipynb$",
        r"chemistry.*\.py$",
    ],
    "duplicates": [
        r"HEART_Duplicate_Impact_.*\.ipynb$",
        r".*duplicate.*\.ipynb$",
    ],
}

EVIDENCE_MAPPINGS = {
    "ugi_priority1_interactive_evidence": "ugi/ugi_priority1_interactive_evidence",
    "ugi_lnpdb_priority2_evidence": "lnpdb/ugi_lnpdb_priority2_evidence",
    "general_pretraining_figures_v4": "general/general_pretraining_figures_v4",
    "duplicate_impact_expanded_evidence": "duplicate_impact/duplicate_impact_expanded_evidence",
}

FIGURE_EXTS = {".png", ".svg", ".pdf", ".jpg", ".jpeg"}

PRIORITY1_FILES = {
    "HEART_Ugi_Priority1_AutoFull_EN.ipynb",
    "HEART_Ugi_Priority1_AutoFull_Executed.ipynb",
    "HEART_Ugi_Priority1_AutoFull_Executed(1).ipynb",
    "HEART_Ugi_Priority1_AutoFull_Executed_AppendFigure.ipynb",
}
PRIORITY2_FILES = {
    "HEART_LNPDB_Priority2_AllInOne.ipynb",
    "HEART_LNPDB_Priority2_Executed.ipynb",
}


def sha256sum(p: Path) -> str:
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def categorized_notebook_dest(name: str, nb_root: Path) -> Path | None:
    for cat, patterns in NOTEBOOK_CATEGORIES.items():
        for pat in patterns:
            if re.match(pat, name):
                return nb_root / cat / name
    return None


def rewrite_notebook_paths(text: str, root: Path) -> str:
    # Only update newly organized validation paths. Keep /project paths unchanged.
    text = text.replace(
        'NOTEBOOK_DIR / "heart-molecular-pretraining"',
        'HEART_ROOT / "heart-molecular-pretraining"'
    )
    text = text.replace(
        'NOTEBOOK_DIR / "ugi_priority1_interactive_evidence"',
        'HEART_ROOT / "validation/evidence/ugi/ugi_priority1_interactive_evidence"'
    )
    text = text.replace(
        'MODEL_TRAIN / "ugi_priority1_interactive_evidence"',
        'MODEL_TRAIN / "validation/evidence/ugi/ugi_priority1_interactive_evidence"'
    )
    text = text.replace(
        'MODEL_TRAIN / "ugi_lnpdb_priority2_evidence"',
        'MODEL_TRAIN / "validation/evidence/lnpdb/ugi_lnpdb_priority2_evidence"'
    )
    text = text.replace(
        'MODEL_TRAIN / "general_pretraining_figures_v4"',
        'MODEL_TRAIN / "validation/evidence/general/general_pretraining_figures_v4"'
    )
    return text


def safe_move(src: Path, dst: Path, report: list[dict]):
    ensure_dir(dst.parent)
    if dst.exists():
        src_hash = sha256sum(src)
        dst_hash = sha256sum(dst)
        if src_hash == dst_hash:
            src.unlink()
            report.append({"action": "delete_duplicate_source", "src": str(src), "dst": str(dst)})
            return
        backup = dst.with_name(dst.stem + "_CONFLICT_" + datetime.now().strftime("%Y%m%d_%H%M%S") + dst.suffix)
        dst.rename(backup)
        report.append({"action": "backup_conflicting_dest", "src": str(src), "dst": str(dst), "backup": str(backup)})
    shutil.move(str(src), str(dst))
    report.append({"action": "move", "src": str(src), "dst": str(dst)})


def collect_figures(src_root: Path, dst_root: Path, report: list[dict]):
    mapping = {
        src_root / 'validation/evidence/general': dst_root / '01_general_pretraining',
        src_root / 'validation/evidence/ugi': dst_root / '03_ugi_validation',
        src_root / 'validation/evidence/lnpdb': dst_root / '04_lnpdb',
        src_root / 'validation/evidence/duplicate_impact': dst_root / '05_duplicate_impact',
        src_root / 'validation/notebooks/general': dst_root / '01_general_pretraining',
    }
    rows = []
    ensure_dir(dst_root)
    for source_base, target_base in mapping.items():
        if not source_base.exists():
            continue
        for p in source_base.rglob('*'):
            if p.is_file() and p.suffix.lower() in FIGURE_EXTS:
                target = target_base / p.name
                ensure_dir(target.parent)
                if target.exists():
                    # avoid overwrite collisions
                    stem, suff = target.stem, target.suffix
                    target = target.with_name(f"{stem}_{hashlib.md5(str(p).encode()).hexdigest()[:8]}{suff}")
                shutil.copy2(p, target)
                rows.append({"figure_name": target.name, "copied_from": str(p), "stored_at": str(target)})
                report.append({"action": "copy_figure", "src": str(p), "dst": str(target)})
    idx = dst_root / 'FIGURE_INDEX.csv'
    with idx.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['figure_name', 'copied_from', 'stored_at'])
        w.writeheader()
        w.writerows(rows)


def create_priority1_slurm(root: Path, slurm_dir: Path, report: list[dict]):
    ensure_dir(slurm_dir)
    script = slurm_dir / 'run_priority1_from_validation.sbatch'
    content = f'''#!/bin/bash
#SBATCH --job-name=heart-ugi-p1
#SBATCH --partition=medium
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output={root}/validation/logs/heart-ugi-p1-%j.out
#SBATCH --error={root}/validation/logs/heart-ugi-p1-%j.err

cd {root}
python -m jupyter nbconvert --to notebook --execute \
  validation/notebooks/ugi/HEART_Ugi_Priority1_AutoFull_EN.ipynb \
  --output validation/notebooks/ugi/HEART_Ugi_Priority1_AutoFull_Executed.ipynb \
  --ExecutePreprocessor.timeout=-1
'''
    script.write_text(content)
    report.append({"action": "write_slurm", "dst": str(script)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, help='model_train root path')
    ap.add_argument('--apply', action='store_true')
    args = ap.parse_args()

    root = Path(args.root).resolve()
    validation = root / 'validation'
    nb_root = validation / 'notebooks'
    ev_root = validation / 'evidence'
    fig_root = validation / 'figures_for_slides'
    slurm_root = validation / 'slurm'
    logs_root = validation / 'logs'
    maint_root = validation / 'maintenance'
    for d in [validation, nb_root, ev_root, fig_root, slurm_root, logs_root, maint_root]:
        ensure_dir(d)
    report = []

    if not args.apply:
        print('Dry run only. Re-run with --apply to move files.')
        print(f'Root: {root}')
        return

    # Move loose notebooks/scripts from root into validation/notebooks/*
    for p in root.iterdir():
        if not p.is_file():
            continue
        if p.name == Path(__file__).name:
            continue
        dest = categorized_notebook_dest(p.name, nb_root)
        if dest is None:
            # standalone slurm
            if p.suffix == '.sbatch' and 'Priority1' in p.name:
                dest = slurm_root / p.name
            elif p.suffix == '.sbatch' and 'Priority2' in p.name:
                dest = slurm_root / p.name
            elif p.name.endswith('.ipynb'):
                dest = nb_root / 'general' / p.name
            elif p.name.endswith('.py') and 'chemistry' in p.name.lower():
                dest = nb_root / 'chemistry' / p.name
            else:
                continue
        # Rewrite notebooks before moving if needed
        if p.suffix == '.ipynb' and p.name in PRIORITY1_FILES.union(PRIORITY2_FILES):
            txt = p.read_text(encoding='utf-8', errors='ignore')
            new_txt = rewrite_notebook_paths(txt, root)
            if new_txt != txt:
                p.write_text(new_txt, encoding='utf-8')
                report.append({"action": "rewrite_notebook_paths", "src": str(p)})
        safe_move(p, dest, report)

    # Move known evidence directories from root to validation/evidence/*
    for src_name, rel_dest in EVIDENCE_MAPPINGS.items():
        src = root / src_name
        dst = ev_root / rel_dest
        if src.exists():
            if dst.exists():
                # move children only
                ensure_dir(dst)
                for child in sorted(src.iterdir()):
                    safe_move(child, dst / child.name, report)
                try:
                    src.rmdir()
                    report.append({"action": "remove_empty_dir", "src": str(src)})
                except OSError:
                    pass
            else:
                ensure_dir(dst.parent)
                shutil.move(str(src), str(dst))
                report.append({"action": "move_dir", "src": str(src), "dst": str(dst)})

    # Copy slide figures from validation evidence into a central place
    collect_figures(root, fig_root, report)

    # Create a new slurm runner for priority1 under validation/slurm
    create_priority1_slurm(root, slurm_root, report)

    # Write report
    rep_path = maint_root / 'HEART_MIGRATION_REPORT.md'
    lines = [
        '# HEART migration report', '',
        f'- Generated: {datetime.now().isoformat(timespec="seconds")}',
        f'- Root: `{root}`', '',
        '## Actions', ''
    ]
    for item in report:
        lines.append(f"- `{item['action']}`: " + ', '.join(f"{k}=`{v}`" for k, v in item.items() if k != 'action'))
    rep_path.write_text('\n'.join(lines))

    json_path = maint_root / 'migration_actions.json'
    json_path.write_text(json.dumps(report, indent=2))
    print(f'Done. See {rep_path}')

if __name__ == '__main__':
    main()
