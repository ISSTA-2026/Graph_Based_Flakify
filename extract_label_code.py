import os
import subprocess
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description="Make Node Label and Codes")
parser.add_argument("dataset", type=str, choices=["FlakeFlagger", "IDoFT"],
                    help="Specify the dataset name (FlakeFlagger or IDoFT)")
args = parser.parse_args()
dataset_name = args.dataset

cpg_dir = os.path.abspath(f"./cpg_bins_{dataset_name}")
working_dir = os.path.abspath(f"joern_scripts_{dataset_name}")
output_dir = os.path.abspath(f"node_token_outputs_{dataset_name}")

os.makedirs(working_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

bin_files = [f for f in os.listdir(cpg_dir) if f.endswith(".bin")]

for bin_file in tqdm(bin_files, desc="Processing bin files"):
    bin_path = os.path.join(cpg_dir, bin_file)
    base_name = os.path.splitext(bin_file)[0]
    out_txt_path = os.path.join(output_dir, f"{base_name}.txt")

    script_path = os.path.join(working_dir, f"{base_name}.sc")
    with open(script_path, "w") as f:
        f.write(f"""\
            val cpg = importCpg("{bin_path}").get
            val writer = new java.io.PrintWriter("{out_txt_path}")
            cpg.all.l.foreach {{ node =>
              val id = node.id
              val label = node.label
              val codeOpt = try {{
                val fields = node.productElementNames.zip(node.productIterator).toMap
                fields.getOrElse("code", "<no_code>")
              }} catch {{
                case _: Throwable => "<node_code>"
              }}
              writer.println(s"$id\\t$label\\t$codeOpt")
            }}
            writer.close()
            """)

    result = subprocess.run(["joern", "--script", script_path], capture_output=True, text=True)
