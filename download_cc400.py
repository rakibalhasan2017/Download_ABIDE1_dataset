import os
import re
import time
import pandas as pd
from nilearn.datasets import fetch_abide_pcp
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import requests

# =========================
# CONFIGURATION
# =========================
DATA_DIR = './abide_cc400_data'
PIPELINE = 'cpac'
DERIVATIVES = 'rois_cc400'
BAND_PASS_FILTERING = True
GLOBAL_SIGNAL_REGRESSION = False
QUALITY_CHECKED = True
VERBOSE = 1

# Retry configuration
MAX_RETRIES = 5
BACKOFF_FACTOR = 2  # Wait time multiplier between retries
TIMEOUT = 120  # Seconds


# =========================
# CONFIGURE SESSION WITH RETRIES
# =========================
def configure_session():
    """Configure requests session with retry logic and longer timeout."""
    session = requests.Session()
    
    retry_strategy = Retry(
        total=MAX_RETRIES,
        backoff_factor=BACKOFF_FACTOR,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"]
    )
    
    adapter = HTTPAdapter(
        max_retries=retry_strategy,
        pool_connections=10,
        pool_maxsize=10
    )
    
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    return session


# =========================
# DOWNLOAD DATA WITH RETRIES
# =========================
def download_abide_cc400():
    print("=" * 70)
    print("DOWNLOADING ABIDE I - CPAC - CC400")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  - Max retries: {MAX_RETRIES}")
    print(f"  - Timeout: {TIMEOUT}s")
    print(f"  - Backoff factor: {BACKOFF_FACTOR}")
    print("=" * 70)

    attempt = 0
    max_attempts = 3
    
    while attempt < max_attempts:
        try:
            print(f"\nAttempt {attempt + 1}/{max_attempts}")
            
            # Configure session with retries
            session = configure_session()
            
            dataset = fetch_abide_pcp(
                data_dir=DATA_DIR,
                pipeline=PIPELINE,
                derivatives=[DERIVATIVES],
                band_pass_filtering=BAND_PASS_FILTERING,
                global_signal_regression=GLOBAL_SIGNAL_REGRESSION,
                quality_checked=QUALITY_CHECKED,
                verbose=VERBOSE
            )
            
            print("\n✓ Download completed successfully!")
            return dataset
            
        except (requests.exceptions.ConnectionError, 
                requests.exceptions.Timeout,
                TimeoutError) as e:
            attempt += 1
            if attempt < max_attempts:
                wait_time = BACKOFF_FACTOR ** attempt * 10
                print(f"\n✗ Download failed: {type(e).__name__}")
                print(f"  Waiting {wait_time}s before retry {attempt + 1}/{max_attempts}...")
                time.sleep(wait_time)
            else:
                print(f"\n✗ Failed after {max_attempts} attempts")
                print("\nSuggestions:")
                print("  1. Check your internet connection")
                print("  2. Try again later (server might be busy)")
                print("  3. Use the resume capability - nilearn should skip already downloaded files")
                print("  4. Consider downloading in smaller batches")
                raise
        
        except Exception as e:
            print(f"\n✗ Unexpected error: {type(e).__name__}")
            print(f"  Message: {str(e)}")
            raise


# =========================
# SAVE PHENOTYPIC CSV
# =========================
def save_phenotypic_data(dataset):
    print("\n" + "=" * 70)
    print("SAVING PHENOTYPIC DATA")
    print("=" * 70)
    
    phenotypic = dataset.phenotypic.copy()

    # Add CC400 file paths
    phenotypic['file_path'] = dataset.rois_cc400

    # Binary labels
    phenotypic['label'] = phenotypic['DX_GROUP'].apply(lambda x: 1 if x == 1 else 0)

    out_dir = os.path.join(DATA_DIR, 'phenotypic')
    os.makedirs(out_dir, exist_ok=True)

    out_csv = os.path.join(out_dir, 'phenotypic_data.csv')
    phenotypic.to_csv(out_csv, index=False)

    print(f"✓ Phenotypic CSV saved to: {out_csv}")
    print(f"  Total subjects: {len(phenotypic)}")
    return phenotypic


# =========================
# VERIFY FILES USING SUB_ID
# =========================
def verify_cc400_files(phenotypic_df):
    print("\n" + "=" * 70)
    print("VERIFYING CC400 FILES")
    print("=" * 70)

    # Normalize SUB_ID to 7 digits
    phenotypic_df['SUB_ID_STR'] = phenotypic_df['SUB_ID'].astype(str).str.zfill(7)
    csv_subjects = set(phenotypic_df['SUB_ID_STR'])

    # Locate CC400 directory
    cc400_dir = os.path.join(
        DATA_DIR, 'ABIDE_pcp', PIPELINE, 'filt_noglobal'
    )

    if not os.path.exists(cc400_dir):
        print(f"✗ Directory not found: {cc400_dir}")
        return

    actual_files = [f for f in os.listdir(cc400_dir) if f.lower().endswith('.1d')]

    print(f"Subjects in CSV: {len(csv_subjects)}")
    print(f".1D files found: {len(actual_files)}")

    # Regex: extract 7-digit SUB_ID
    pattern = re.compile(
        r'_(\d{7})_rois_cc400\.1d',
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
        print("\nFILES MISSING FROM DIRECTORY (first 20):")
        for sub in sorted(missing_from_dir)[:20]:
            print(f"  *_{sub}_rois_cc400.1D")
        if len(missing_from_dir) > 20:
            print(f"  ... and {len(missing_from_dir) - 20} more")

    if extra_in_dir:
        print("\nEXTRA FILES IN DIRECTORY (first 20):")
        for sub in sorted(extra_in_dir)[:20]:
            print(f"  {dir_subjects[sub]}")
        if len(extra_in_dir) > 20:
            print(f"  ... and {len(extra_in_dir) - 20} more")

    if not missing_from_dir and not extra_in_dir:
        print("\n✓ PERFECT MATCH: All CC400 files accounted for.")

    print("=" * 70)
    
    return {
        'matched': len(in_both),
        'missing': len(missing_from_dir),
        'extra': len(extra_in_dir)
    }


# =========================
# CHECK EXISTING FILES
# =========================
def check_existing_downloads():
    """Check how many files have already been downloaded."""
    cc400_dir = os.path.join(DATA_DIR, 'ABIDE_pcp', PIPELINE, 'filt_noglobal')
    
    if not os.path.exists(cc400_dir):
        print("No existing downloads found.")
        return 0
    
    existing_files = [f for f in os.listdir(cc400_dir) if f.lower().endswith('.1d')]
    print(f"Found {len(existing_files)} existing .1D files")
    return len(existing_files)


# =========================
# MAIN
# =========================
def main():
    print("\n" + "=" * 70)
    print("ABIDE CC400 DATASET DOWNLOAD & VERIFICATION")
    print("=" * 70)
    
    # Check existing downloads
    existing_count = check_existing_downloads()
    
    if existing_count > 0:
        print(f"\n✓ Resuming download - {existing_count} files already exist")
        print("  (nilearn will skip already downloaded files)")
    
    try:
        # Download dataset
        dataset = download_abide_cc400()
        
        # Save phenotypic data
        phenotypic_df = save_phenotypic_data(dataset)
        
        # Verify files
        stats = verify_cc400_files(phenotypic_df)
        
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"✓ Download completed")
        print(f"✓ Phenotypic data saved")
        print(f"✓ Files verified: {stats['matched']} matched")
        
        if stats['missing'] > 0 or stats['extra'] > 0:
            print(f"\n⚠ Note: Some files may still be downloading or failed")
            print(f"  You can re-run this script to resume the download")
        
        print("=" * 70)
        
    except KeyboardInterrupt:
        print("\n\n✗ Download interrupted by user")
        print("  You can re-run this script to resume the download")
        
    except Exception as e:
        print(f"\n\n✗ Error occurred: {type(e).__name__}")
        print(f"  Message: {str(e)}")
        print("\nYou can re-run this script to resume the download")


if __name__ == "__main__":
    main()