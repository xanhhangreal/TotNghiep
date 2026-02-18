"""
Setup WESAD dataset - Multiple download methods
Run this script directly: python scripts/setup_wesad.py
"""
import os
import sys
import pickle
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "public"
WESAD_DIR = DATA_DIR / "WESAD"

def method_1_curl():
    """Download using curl.exe (built into Windows 10+)"""
    print("\n[Method 1] Downloading with curl.exe...")
    zip_path = DATA_DIR / "WESAD.zip"
    url = "https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx/download"
    
    # curl supports resume (-C -)
    cmd = f'curl.exe -L -C - -o "{zip_path}" "{url}"'
    print(f"Running: {cmd}")
    os.system(cmd)
    
    if zip_path.exists() and zip_path.stat().st_size > 2_000_000_000:  # >2GB
        print(f"Download complete! Size: {zip_path.stat().st_size / 1e9:.2f} GB")
        return True
    else:
        size = zip_path.stat().st_size / 1e6 if zip_path.exists() else 0
        print(f"Download may be incomplete. Current size: {size:.1f} MB")
        print("You can re-run this script to resume the download.")
        return False

def method_2_browser():
    """Open browser to download page"""
    import webbrowser
    print("\n[Method 2] Opening download page in browser...")
    print("URL: https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx")
    print(f"Save the file to: {DATA_DIR}")
    webbrowser.open("https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx")
    input("\nPress Enter after the download is complete...")

def extract_and_verify():
    """Extract zip and verify dataset"""
    zip_path = DATA_DIR / "WESAD.zip"
    
    if not zip_path.exists():
        print(f"Zip not found at: {zip_path}")
        return False
    
    print(f"\nExtracting {zip_path.name} ({zip_path.stat().st_size / 1e9:.2f} GB)...")
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(DATA_DIR)
        print("Extraction complete!")
    except zipfile.BadZipFile:
        print("ERROR: Zip file is corrupted. Please re-download.")
        return False
    
    return verify()

def verify():
    """Verify WESAD dataset structure"""
    print("\nVerifying WESAD dataset...")
    
    expected = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
    found = []
    
    for sid in expected:
        pkl = WESAD_DIR / f"S{sid}" / f"S{sid}.pkl"
        if pkl.exists():
            found.append(sid)
            print(f"  [OK] S{sid} ({pkl.stat().st_size / 1e6:.0f} MB)")
        else:
            print(f"  [--] S{sid} missing")
    
    print(f"\nFound {len(found)}/{len(expected)} subjects")
    
    if found:
        # Quick test: load one subject
        sid = found[0]
        pkl = WESAD_DIR / f"S{sid}" / f"S{sid}.pkl"
        print(f"\nLoading S{sid} for verification...")
        
        with open(pkl, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        
        print(f"  Keys: {list(data.keys())}")
        
        if 'signal' in data:
            sig = data['signal']
            if 'wrist' in sig:
                wrist = sig['wrist']
                print(f"  Wrist signals: {list(wrist.keys())}")
                for k, v in wrist.items():
                    if hasattr(v, 'shape'):
                        print(f"    {k}: {v.shape}")
            if 'chest' in sig:
                chest = sig['chest']
                print(f"  Chest signals: {list(chest.keys())}")
        
        if 'label' in data:
            import numpy as np
            labels = np.array(data['label']).flatten()
            unique, counts = np.unique(labels, return_counts=True)
            print(f"  Labels: {dict(zip(unique.astype(int), counts))}")
            # WESAD label mapping: 0=not defined, 1=baseline, 2=stress, 
            #                     3=amusement, 4=meditation
        
        print("\n✓ Dataset verified successfully!")
        return True
    
    return False

def main():
    print("=" * 60)
    print("  WESAD Dataset Setup")
    print("=" * 60)
    
    # Check if already exists
    if WESAD_DIR.exists():
        pkls = list(WESAD_DIR.glob("S*/S*.pkl"))
        if pkls:
            print(f"\nWESAD already set up with {len(pkls)} subjects!")
            verify()
            return
    
    # Check if zip already exists
    zip_path = DATA_DIR / "WESAD.zip"
    if zip_path.exists() and zip_path.stat().st_size > 2_000_000_000:
        print(f"\nFound complete zip ({zip_path.stat().st_size / 1e9:.2f} GB)")
        extract_and_verify()
        return
    
    print(f"\nWESAD not found. Choose download method:")
    print(f"  1. curl.exe (command-line, supports resume)")
    print(f"  2. Open in browser (manual download)")
    print(f"  3. Just verify (if you already placed files)")
    
    choice = input("\nChoice [1/2/3]: ").strip()
    
    if choice == "1":
        success = method_1_curl()
        if success:
            extract_and_verify()
    elif choice == "2":
        method_2_browser()
        if zip_path.exists():
            extract_and_verify()
    elif choice == "3":
        if not verify():
            print(f"\nDataset not found. Please download WESAD to:")
            print(f"  {WESAD_DIR}")
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()
