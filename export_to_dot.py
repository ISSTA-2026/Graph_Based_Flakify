import os
import subprocess
import argparse
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Export CPG binaries to dot files using joern-export.")
    parser.add_argument("dataset", type=str, choices=["FlakeFlagger", "IDoFT"], 
                        help="Specify the dataset name (FlakeFlagger or IDoFT)")
    
    args = parser.parse_args()
    dataset_name = args.dataset

    input_dir = f"./cpg_bins_{dataset_name}"
    output_dir = f"./dot_outputs_{dataset_name}"

    os.makedirs(output_dir, exist_ok=True)

    bin_files = [f for f in os.listdir(input_dir) if f.endswith(".bin")]
    
    print(f"Target Dataset: {dataset_name}")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Processing {len(bin_files)} files...")

    for filename in tqdm(bin_files):
        input_path = os.path.join(input_dir, filename)
        
        file_stem = filename.replace(".bin", "")
        output_subdir = os.path.join(output_dir, file_stem)

        cmd = ["joern-export", input_path, "--out", output_subdir]
        
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"\nFailed to export: {filename}")

if __name__ == "__main__":
    main()