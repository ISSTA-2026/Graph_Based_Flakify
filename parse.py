import os
import subprocess
import argparse
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Parse Java files to CPG binaries using joern-parse.")
    parser.add_argument("dataset", type=str, choices=["FlakeFlagger", "IDoFT"], 
                        help="Specify the dataset name (FlakeFlagger or IDoFT)")
    
    args = parser.parse_args()
    dataset_name = args.dataset

    input_dir = os.path.join("./dataset", dataset_name, f"{dataset_name}_class_files")
    output_dir = f"./cpg_bins_{dataset_name}"

    os.makedirs(output_dir, exist_ok=True)

    java_files = [f for f in os.listdir(input_dir) if f.endswith(".java")]
    
    print(f"Target Dataset: {dataset_name}")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Processing {len(java_files)} files...")

    for filename in tqdm(java_files):
        input_path = os.path.join(input_dir, filename)
        output_filename = filename.replace(".java", ".bin")
        output_path = os.path.join(output_dir, output_filename)

        cmd = ["joern-parse", input_path, "--output", output_path]
        
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"\nFailed to parse: {filename}")

if __name__ == "__main__":
    main()