import os
import re
import pandas as pd
from nilearn.datasets import fetch_abide_pcp

# =========================
# CONFIGURATION
# =========================
DATA_DIR = './abide_cc200_data'
PIPELINE = 'cpac'
DERIVATIVES = 'rois_cc200'
BAND_PASS_FILTERING = True
GLOBAL_SIGNAL_REGRESSION = False
QUALITY_CHECKED = True
VERBOSE = 1

# =========================
# DOWNLOAD DATA
# =========================
def download_abide_cc200():
    print("=" * 70)
    print("DOWNLOADING ABIDE I - CPAC - CC200")
    print("=" * 70)

    dataset = fetch_abide_pcp(
        data_dir=DATA_DIR,
        pipeline=PIPELINE,
        derivatives=[DERIVATIVES],
        band_pass_filtering=BAND_PASS_FILTERING,
        global_signal_regression=GLOBAL_SIGNAL_REGRESSION,
        quality_checked=QUALITY_CHECKED,
        verbose=VERBOSE
    )

    return dataset


# =========================
# SAVE PHENOTYPIC CSV
# =========================
def save_phenotypic_data(dataset):
    phenotypic = dataset.phenotypic.copy()

    # Add CC200 file paths
    phenotypic['file_path'] = dataset.rois_cc200

    # Binary labels
    phenotypic['label'] = phenotypic['DX_GROUP'].apply(lambda x: 1 if x == 1 else 0)

    out_dir = os.path.join(DATA_DIR, 'phenotypic')
    os.makedirs(out_dir, exist_ok=True)

    out_csv = os.path.join(out_dir, 'phenotypic_data.csv')
    phenotypic.to_csv(out_csv, index=False)

    print(f"✓ Phenotypic CSV saved to: {out_csv}")
    return phenotypic


# =========================
# VERIFY FILES USING SUB_ID
# =========================
def verify_cc200_files(phenotypic_df):
    print("\n" + "=" * 70)
    print("VERIFYING CC200 FILES")
    print("=" * 70)

    # Normalize SUB_ID to 7 digits
    phenotypic_df['SUB_ID_STR'] = phenotypic_df['SUB_ID'].astype(str).str.zfill(7)
    csv_subjects = set(phenotypic_df['SUB_ID_STR'])

    # Locate CC200 directory
    cc200_dir = os.path.join(
        DATA_DIR, 'ABIDE_pcp', PIPELINE, 'filt_noglobal'
    )

    actual_files = [f for f in os.listdir(cc200_dir) if f.lower().endswith('.1d')]

    print(f"Subjects in CSV: {len(csv_subjects)}")
    print(f".1D files found: {len(actual_files)}")

    # Regex: extract 7-digit SUB_ID
    pattern = re.compile(
        r'_(\d{7})_rois_cc200\.1d',
        re.IGNORECASE
    )

    dir_subjects = {}
    for f in actual_files:
        m = pattern.search(f)
        if m:
            dir_subjects[m.group(1)] = f

    dir_subject_ids = set(dir_subjects.keys())

    # Comparisons
    in_both = csv_subjects & dir_subject_ids
    missing_from_dir = csv_subjects - dir_subject_ids
    extra_in_dir = dir_subject_ids - csv_subjects

    # =========================
    # REPORT
    # =========================
    print("\nRESULTS")
    print("-" * 70)
    print(f"✓ Matched subjects: {len(in_both)}")
    print(f"✗ Missing files: {len(missing_from_dir)}")
    print(f"✗ Extra files: {len(extra_in_dir)}")

    if missing_from_dir:
        print("\nFILES MISSING FROM DIRECTORY:")
        for sub in sorted(missing_from_dir):
            print(os.path.join(cc200_dir, f"*_{sub}_rois_cc200.1D"))

    if extra_in_dir:
        print("\nEXTRA FILES IN DIRECTORY:")
        for sub in sorted(extra_in_dir):
            print(os.path.join(cc200_dir, dir_subjects[sub]))

    if not missing_from_dir and not extra_in_dir:
        print("\n✓ PERFECT MATCH: All CC200 files accounted for.")

    print("=" * 70)


# =========================
# MAIN
# =========================
def main():
    dataset = download_abide_cc200()
    phenotypic_df = save_phenotypic_data(dataset)
    verify_cc200_files(phenotypic_df)

    print("\nDOWNLOAD + VERIFICATION COMPLETE")


if __name__ == "__main__":
    main()
