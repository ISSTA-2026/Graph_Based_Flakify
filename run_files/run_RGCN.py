import time
import gc
import pandas as pd
import numpy as np
import dgl
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
import random
import argparse
from tqdm import tqdm
import os
from models.model_RGCN import GatedRGCNClassifier_CNN_Improved

# setting the seed for reproducibility
def set_deterministic(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)                    
    torch.cuda.manual_seed(seed)               
    torch.cuda.manual_seed_all(seed)           
    torch.backends.cudnn.deterministic = True 

# specify GPU
device = torch.device("cuda")

parser = argparse.ArgumentParser(description="Select Model Name")
parser.add_argument("dataset", type=str, choices=["FlakeFlagger", "IDoFT"],
                    help="Specify the dataset name (FlakeFlagger or IDoFT)")
parser.add_argument("model", type=str, choices=["CodeBERT", "GraphCodeBERT", "UniXcoder"],
help="Specify the model name (CodeBERT, GraphCodeBERT, or UniXcoder)")
args = parser.parse_args()

dataset_name = args.dataset
llm_name = args.model.lower()

dataset_path = f"../dataset/Flakify_{dataset_name}_dataset.csv"
results_file = "../results/Flakify_cross_validation_results.csv"

df = pd.read_csv(dataset_path)
input_data = df['full_code'] 
target_data = df['flaky']
case_names = df['test_case_name']
df.head()

GRAPH_DIR = f"../saved_graphs_{dataset_name}"
graphs = []

for fname in tqdm(os.listdir(GRAPH_DIR)):
    g = torch.load(os.path.join(GRAPH_DIR, fname))
    graphs.append(g)


def custom_collate_fn(batch):
    labels, graphs, type_vec, code_vec, edge_types = map(list, zip(*batch))
    return (
        torch.tensor(labels),
        dgl.batch(graphs),
        torch.cat(type_vec, dim=0),   
        torch.cat(code_vec, dim=0),    
        torch.cat(edge_types, dim=0)
    )

class MultiInputDataset(Dataset):
    def __init__(self, labels, test_case_names, graph_list):
        self.labels = labels
        self.test_case_names = test_case_names
        self.graph_list = graph_list
        self.graph_map = {g.graph_id: g for g in self.graph_list}

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        test_case_name = self.test_case_names.iloc[idx]  
        graph = self.graph_map[test_case_name]

        type_vec = graph.ndata['type_vec']    # shape: [num_nodes, 16]
        code_vec = graph.ndata[f'{llm_name}_vec']    # shape: [num_nodes, 768] 
        edge_types = graph.edata['edge_type']                # shape: [num_edges]

        return label, graph, type_vec, code_vec, edge_types


def sampling(train_df, valid_df):

    oversampling = RandomOverSampler(sampling_strategy='minority', random_state=seed)
    train_df = train_df.reset_index(drop=True)
    train_index = train_df.index.values.reshape(-1, 1)
    y_train = train_df['label'].values
    resampled_train_index, _ = oversampling.fit_resample(train_index, y_train)
    train_df_sampled = train_df.iloc[resampled_train_index.ravel()].reset_index(drop=True)
    return train_df_sampled, valid_df.reset_index(drop=True)


# sett seed for data_loaders for output reproducibility
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


g = torch.Generator()
seed = 42 # any number 
set_deterministic(seed)

def data_loaders(train_y, val_y, test_y,
                 train_case_names, val_case_names, test_case_names, graphs):
    batch_size = 4

    train_dataset = MultiInputDataset(train_y, train_case_names, graphs)
    val_dataset = MultiInputDataset(val_y, val_case_names, graphs)
    test_dataset = MultiInputDataset(test_y, test_case_names, graphs)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                  sampler=RandomSampler(train_dataset), worker_init_fn=seed_worker, collate_fn=custom_collate_fn)

    val_dataloader = DataLoader(val_dataset, batch_size=batch_size,
                                sampler=SequentialSampler(val_dataset), worker_init_fn=seed_worker, collate_fn=custom_collate_fn)
    
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                                sampler=SequentialSampler(test_dataset), worker_init_fn=seed_worker, collate_fn=custom_collate_fn)

    return train_dataloader, val_dataloader, test_dataloader


def train():

    model.train()

    total_loss = 0
    total_preds = []

    for batch in tqdm(train_dataloader):

        labels, graph, type_vec, code_vec, edge_types = batch

        labels = labels.to(device)
        graph =graph.to(device)
        type_vec = type_vec.to(device)
        code_vec = code_vec.to(device)
        edge_types = edge_types.to(device)

        optimizer.zero_grad()
        preds = model(graph, type_vec, code_vec, edge_types)
        loss = cross_entropy(preds, labels)
        total_loss = total_loss + loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        preds = preds.detach().cpu().numpy()
        total_preds.append(preds)

    avg_loss = total_loss / len(train_dataloader)
    total_preds = np.concatenate(total_preds, axis=0)

    return avg_loss, total_preds

def evaluate():

    print("\nEvaluating..")

    model.eval()

    total_loss = 0
    total_preds = []

    for batch in tqdm(val_dataloader):

        labels, graph, type_vec, code_vec, edge_types = batch

        labels = labels.to(device)
        graph =graph.to(device)
        type_vec = type_vec.to(device)
        code_vec = code_vec.to(device)
        edge_types = edge_types.to(device)

        with torch.no_grad():

            preds = model(graph, type_vec, code_vec, edge_types)
            loss = cross_entropy(preds, labels)
            total_loss = total_loss + loss.item()
            preds = preds.detach().cpu().numpy()
            total_preds.append(preds)

    avg_loss = total_loss / len(val_dataloader)
    total_preds = np.concatenate(total_preds, axis=0)

    return avg_loss, total_preds

def get_evaluation_scores(tn, fp, fn, tp):
    print("get_score method is defined")
    if(tp == 0):
        accuracy = (tp+tn)/(tn+fp+fn+tp)
        Precision = 0
        Recall = 0
        F1 = 0
    else:
        accuracy = (tp+tn)/(tn+fp+fn+tp)
        Precision = tp/(tp+fp)
        Recall = tp/(tp+fn)
        F1 = 2*((Precision*Recall)/(Precision+Recall))
    return accuracy, F1, Precision, Recall


def test(model, test_dataloader):
    print("\nPredicting on test data...")

    model.eval()
    all_preds = []

    for batch in tqdm(test_dataloader):

        labels, graph, type_vec, code_vec, edge_types = batch  

        labels = labels.to(device)
        graph =graph.to(device)
        type_vec = type_vec.to(device)
        code_vec = code_vec.to(device)
        edge_types = edge_types.to(device)

        with torch.no_grad():
            preds = model(graph, type_vec, code_vec, edge_types)
            preds = preds.detach().cpu().numpy()
            all_preds.append(preds)

    all_preds = np.concatenate(all_preds, axis=0)

    return np.argmax(all_preds, axis=1)


execution_time = time.time()
print("Start time of the experiment", execution_time)
skf = StratifiedKFold(n_splits=10,shuffle=True, random_state=seed)
TN = FP = FN = TP = 0
fold_number = 0
all_fold_test_results = []

for train_index, test_index in skf.split(input_data, target_data):

    print(" NOW IN FOLD NUMBER", fold_number)
    X_train, X_test = input_data.iloc[list(train_index)], input_data.iloc[list(test_index)]
    y_train, y_test = target_data.iloc[list(train_index)], target_data.iloc[list(test_index)]
    case_train, case_test = case_names.iloc[train_index], case_names.iloc[test_index]
    
    X_train, X_valid, y_train, y_valid, case_train, case_valid=train_test_split(X_train, y_train, case_train, random_state=seed, test_size=0.2, stratify=y_train)
    
    train_df = pd.DataFrame({
        'code': X_train,
        'label': y_train,
        'test_case_name': case_train
    })

    valid_df = pd.DataFrame({
        'code': X_valid,
        'label': y_valid,
        'test_case_name': case_valid
    })

    test_df = pd.DataFrame({
        'code': X_test,
        'label': y_test,
        'test_case_name': case_test
    })

    train_df_sampled, valid_df_sampled = sampling(train_df, valid_df)

    X_train_sampled = train_df_sampled['code']
    y_train_sampled = train_df_sampled['label']
    case_train_sampled = train_df_sampled['test_case_name']

    X_valid_sampled = valid_df_sampled['code']
    y_valid_sampled = valid_df_sampled['label']
    case_valid_sampled = valid_df_sampled['test_case_name']

    Y_train = pd.DataFrame(y_train_sampled)
    y_val = pd.DataFrame(y_valid_sampled)
    y_test = pd.DataFrame(y_test)

    Y_train.columns = ['which_tests']
    y_val.columns = ['which_tests']
    y_test.columns = ['which_tests']

    train_y = torch.tensor(Y_train['which_tests'].values)
    val_y = torch.tensor(y_val['which_tests'].values)
    test_y = torch.tensor(y_test['which_tests'].values)

    train_dataloader, val_dataloader, test_dataloader = data_loaders(
    train_y, val_y, test_y,
    train_df_sampled['test_case_name'],  
    valid_df_sampled['test_case_name'], 
    test_df['test_case_name'],
    graphs  
)

    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(Y_train.values), y=np.ravel(Y_train.values))
    weights = torch.tensor(class_weights, dtype=torch.float)
    weights = weights.to(device)
    cross_entropy = nn.NLLLoss(weight=weights)
    print("Class Weights:", class_weights)

    epochs = 30

    model = GatedRGCNClassifier_CNN_Improved(in_dim=768, num_rels=4, hidden_dim=768, out_dim=2, num_layers=4, dropout=0.1).to(device)

    optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    gc.collect()
    torch.cuda.empty_cache()
    # set initial loss to infinite
    best_valid_loss = float('inf')

    counter = 0
    patience = 5

    train_losses = []
    valid_losses = []

    for epoch in range(epochs):

        print('\n Epoch {:} / {:}'.format(epoch + 1, epochs))

        train_loss, _ = train()
        valid_loss, _ = evaluate()

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            counter = 0
            torch.save(model.state_dict(), f"../results/best_model_fold_{fold_number}.pt")
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        print(f'\nTraining Loss: {train_loss:.3f}')
        print(f'Validation Loss: {valid_loss:.3f}')

    model.load_state_dict(torch.load(f"../results/best_model_fold_{fold_number}.pt"))
    preds = test(model, test_dataloader)
    test_case_names = test_df['test_case_name'].values
    true_labels = test_y.cpu().numpy() if torch.is_tensor(test_y) else np.array(test_y)

    fold_result = pd.DataFrame({
        "fold": fold_number,
        "test_case_name": test_case_names,
        "true_label": true_labels,
        "pred_label": preds
    })
    all_fold_test_results.append(fold_result)

    print(classification_report(test_y, preds))
    tn, fp, fn, tp = confusion_matrix(test_y, preds, labels=[0, 1]).ravel()
    TN = TN + tn
    FP = FP + fp
    FN = FN + fn
    TP = TP + tp

    del model
    torch.cuda.empty_cache()

    fold_number = fold_number+1

accuracy, F1, Precision, Recall = get_evaluation_scores(TN, FP, FN, TP)

result = pd.DataFrame(columns=['Accuracy', 'F1', 'Precision', 'Recall', 'TN', 'FP', 'FN', 'TP'])
new_row = pd.DataFrame([[accuracy, F1, Precision, Recall, TN, FP, FN, TP]], columns=result.columns)
result = pd.concat([result, new_row], ignore_index=True)

result.to_csv(results_file, index=False)
all_fold_results_df = pd.concat(all_fold_test_results, ignore_index=True)
all_fold_results_df.to_csv("../results/RGCN_test_fold_predictions.csv", index=False)

print("The processed is completed in : (%s) seconds. " % round((time.time() - execution_time), 5))