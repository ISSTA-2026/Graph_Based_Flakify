import os
import torch
import dgl
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import torch
import os
import re
import argparse

parser = argparse.ArgumentParser(description="Initial Node Embedding")
parser.add_argument("dataset", type=str, choices=["FlakeFlagger", "IDoFT"],
                    help="Specify the dataset name (FlakeFlagger or IDoFT)")
args = parser.parse_args()
dataset_name = args.dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

graphcodebert_tokenizer = AutoTokenizer.from_pretrained("microsoft/graphcodebert-base")
graphcodebert_model = AutoModel.from_pretrained("microsoft/graphcodebert-base")
graphcodebert_model.eval()
graphcodebert_model.to(device)

codebert_tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
codebert_model = AutoModel.from_pretrained("microsoft/codebert-base")
codebert_model.eval()
codebert_model.to(device)

unixcoder_tokenizer = AutoTokenizer.from_pretrained("microsoft/unixcoder-base")
unixcoder_model = AutoModel.from_pretrained("microsoft/unixcoder-base")
unixcoder_model.eval()
unixcoder_model.to(device)

ROOT_DIR = f"dot_outputs_{dataset_name}"
SAVE_DIR = f"saved_graphs_{dataset_name}"
NODE_DIR = f"node_token_outputs_{dataset_name}"
os.makedirs(SAVE_DIR, exist_ok=True)

global_node_types = set()
global_edge_types = set()

def graphcodebert_embed(text):
    inputs = graphcodebert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256).to(device)
    with torch.no_grad():
        outputs = graphcodebert_model(**inputs)
    cls_vector = outputs.last_hidden_state[:, 0, :]  # [1, 768]
    return cls_vector.squeeze(0).cpu()  # [768]

def codebert_embed(text):
    inputs = codebert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256).to(device)
    with torch.no_grad():
        outputs = codebert_model(**inputs)
    cls_vector = outputs.last_hidden_state[:, 0, :]  # [1, 768]
    return cls_vector.squeeze(0).cpu()  # [768]

def unixcoder_embed(text):
    inputs = unixcoder_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256).to(device)
    with torch.no_grad():
        outputs = unixcoder_model(**inputs)
    cls_vector = outputs.last_hidden_state[:, 0, :]  # [1, 768]
    return cls_vector.squeeze(0).cpu()  # [768]

HEADER_RE = re.compile(r'^(\d+)\t([^\t]+)\t(.*)$')  

def embed_safe(text, model_name):
    if model_name == "graphcodebert":
        if text == "<empty>" or text == "<no_code>":
            return torch.zeros(768) 
        else:
            return graphcodebert_embed(text)
    elif model_name == "codebert":
        if text == "<empty>" or text == "<no_code>":
            return torch.zeros(768)  
        else:
            return codebert_embed(text)
    elif model_name == "unixcoder":
        if text == "<empty>" or text == "<no_code>":
            return torch.zeros(768)  
        else:
            return unixcoder_embed(text)
    else:
        print(model_name + "is not defined")
        exit()

def extract_number(filename):
    return int(filename.split('-')[0])

def parse_node_file(filepath):
    nodes = []
    current = None
    code_buf = None

    with open(filepath, 'r', encoding='utf-8') as f:
        for raw in f:
            line = raw.rstrip('\n').rstrip('\r')

            m = HEADER_RE.match(line)
            if m:
                if current is not None:
                    current["code"] = "\n".join(code_buf)
                    nodes.append(current)

                node_id = int(m.group(1))
                node_label = m.group(2)
                code_start = m.group(3) if m.group(3) != "" else "<empty>"

                current = {"id": node_id, "label": node_label}
                code_buf = [code_start]
            else:
                if current is not None:
                    code_buf.append(line)
                else:
                    continue

    if current is not None:
        current["code"] = "\n".join(code_buf)
        nodes.append(current)

    return nodes


for java_dir in tqdm(os.listdir(ROOT_DIR)):
    dir_path = os.path.join(ROOT_DIR, java_dir)
    if not os.path.isdir(dir_path):
        continue

    node_path = os.path.join(NODE_DIR, java_dir)
    node_info = parse_node_file(node_path + ".txt")

    method_name = []

    for i in node_info:
        if i['label'] == "METHOD":
            if i['code'] != "<empty>" and i['code'] != "<lambda>":
                name = i['code'].split("(", 1)[0]
                if " " in name:
                    name = name.rsplit(" ", 1)[1]
                method_name.append(name)

    for dot_file in sorted(os.listdir(dir_path), key=extract_number):
        if len(method_name) == 0:
            break
        path = os.path.join(dir_path, dot_file)

        with open(path, 'r') as fi:
            first = fi.readlines()[0].split("\"", 1)[1].split("\"", 1)[0]
        if first not in method_name:
            continue

        method_name.remove(first)

        with open(path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            if " -> " in line:
                id, line = line.split(" -> ")
                src = id.split("\"", 1)[1].split("\"", 1)[0]
                dst, atr = line.split(" [ label = ")
                dst = dst.split("\"", 1)[1].split("\"", 1)[0]
                atr = atr.split("\"", 1)[1].split(":", 1)[0]
                global_edge_types.add(atr)
                continue

            if " [label = " not in line:
                continue

            id, line = line.split(" [label = ")
            id = id.split("\"", 1)[1].split("\"", 1)[0]

            for i in node_info:
                if int(id) == i['id']:
                    type = i['label']
                    break
            global_node_types.add(type)

node_type_list = sorted(global_node_types)
edge_type_list = sorted(global_edge_types)
node_type_to_idx = {v: i for i, v in enumerate(node_type_list)}
edge_type_to_idx = {v: i for i, v in enumerate(edge_type_list)}

print(global_node_types)
print(global_edge_types)

num_node_types = len(node_type_to_idx)
num_edge_types = len(edge_type_to_idx)
embed_dim = 16   


for java_dir in tqdm(os.listdir(ROOT_DIR)):
    dir_path = os.path.join(ROOT_DIR, java_dir)
    if not os.path.isdir(dir_path):
        continue

    merged_nodes_type = []
    merged_nodes_ids = []
    merged_nodes_code = []
    merged_edges = []
    merged_edges_type = []
    merged_edges_token = []


    node_path = os.path.join(NODE_DIR, java_dir)
    node_info = parse_node_file(node_path + ".txt")

    method_name = []

    for i in node_info:
        if i['label'] == "METHOD":
            if i['code'] != "<empty>" and i['code'] != "<lambda>":
                name = i['code'].split("(", 1)[0]
                if " " in name:
                    name = name.rsplit(" ", 1)[1]
                method_name.append(name)

    for dot_file in sorted(os.listdir(dir_path), key=extract_number):
        if len(method_name) == 0:
            break
        path = os.path.join(dir_path, dot_file)

        with open(path, 'r') as fi:
            first = fi.readlines()[0].split("\"", 1)[1].split("\"", 1)[0]
        if first not in method_name:
            continue

        method_name.remove(first)

        with open(path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            if " -> " in line:
                id, line = line.split(" -> ")
                src = id.split("\"", 1)[1].split("\"", 1)[0]
                dst, atr = line.split(" [ label = ")
                dst = dst.split("\"", 1)[1].split("\"", 1)[0]
                atr = atr.split("\"", 1)[1].split(":", 1)[0]
                src_id = str(src)
                dst_id = str(dst)

                merged_edges_type.append(edge_type_to_idx[atr])
                merged_edges.append({
                    "src": src_id,
                    "dst": dst_id
                })
                continue

            if " [label = " not in line:
                continue

            id, line = line.split(" [label = ")
            id = id.split("\"", 1)[1].split("\"", 1)[0]

            for i in node_info:
                if int(id) == i['id']:
                    type = i['label']
                    code = i['code']
                    break
            
            merged_nodes_ids.append(str(id))
            merged_nodes_type.append(node_type_to_idx[type])
            merged_nodes_code.append(code)
    
    
    id_to_idx = {nid: idx for idx, nid in enumerate(merged_nodes_ids)}

    src = [id_to_idx[e["src"]] for e in merged_edges]
    dst = [id_to_idx[e["dst"]] for e in merged_edges]
    dgl_g = dgl.graph((src, dst))

    node_type_embedding = nn.Embedding(num_node_types, embed_dim)

    node_type_ids = torch.tensor(merged_nodes_type, dtype=torch.long)

    dgl_g.ndata["type_vec"] = node_type_embedding(node_type_ids)

    node_embedding_graphcodebert = [embed_safe(token, "graphcodebert") for token in merged_nodes_code]
    node_embedding_codebert = [embed_safe(token, "codebert") for token in merged_nodes_code]
    node_embedding_unixcoder = [embed_safe(token, "unixcoder") for token in merged_nodes_code]

    node_codebert_tensor_graphcodebert = torch.stack(node_embedding_graphcodebert)
    node_codebert_tensor_codebert = torch.stack(node_embedding_codebert)
    node_codebert_tensor_unixcoder = torch.stack(node_embedding_unixcoder)

    dgl_g.ndata["graphcodebert_vec"] = node_codebert_tensor_graphcodebert
    dgl_g.ndata["codebert_vec"] = node_codebert_tensor_codebert
    dgl_g.ndata["unixcoder_vec"] = node_codebert_tensor_unixcoder

    dgl_g.edata["edge_type"] = torch.tensor(merged_edges_type)

    dgl_g.graph_id = java_dir

    torch.save(dgl_g, os.path.join(SAVE_DIR, f"{java_dir}.pt"))

print("completed")