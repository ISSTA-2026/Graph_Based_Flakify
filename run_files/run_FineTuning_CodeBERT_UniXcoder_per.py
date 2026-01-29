import time
import gc
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
import random
import argparse
from tqdm import tqdm
from models.model_FineTuning_CodeBERT_UniXcoder import BERT_Arch

# setting the seed for reproducibility
def set_deterministic(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)                    
    torch.cuda.manual_seed(seed)               
    torch.cuda.manual_seed_all(seed)           
    torch.backends.cudnn.deterministic = True 

parser = argparse.ArgumentParser(description="Select Model Name")
parser.add_argument("model", type=str, choices=["CodeBERT", "UniXcoder"],
                    help="Specify the model name (CodeBERT or UniXcoder)")
args = parser.parse_args()

llm_name = args.model.lower()

# specify GPU
device = torch.device("cuda")

# choose dataset
dataset_path = "../dataset/Flakify_FlakeFlagger_dataset.csv"
results_file = "../results/Flakify_cross_validation_results.csv"

df = pd.read_csv(dataset_path)
input_data = df['full_code'] 
target_data = df['flaky']
case_names = df['test_case_name']
df.head()

project_name=df['project'].unique()

def custom_collate_fn(batch):
    seqs, masks, labels = map(list, zip(*batch))
    return (
        torch.stack(seqs),
        torch.stack(masks),
        torch.tensor(labels)
    )

class MultiInputDataset(Dataset):
    def __init__(self, seqs, masks, labels):
        self.seqs = seqs
        self.masks = masks
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        mask = self.masks[idx]
        label = self.labels[idx]

        return seq, mask, label
    

def sampling(train_df, valid_df):

    oversampling = RandomOverSampler(sampling_strategy='minority', random_state=seed)
    train_df = train_df.reset_index(drop=True)
    train_index = train_df.index.values.reshape(-1, 1)
    y_train = train_df['label'].values
    resampled_train_index, _ = oversampling.fit_resample(train_index, y_train)
    train_df_sampled = train_df.iloc[resampled_train_index.ravel()].reset_index(drop=True)
    return train_df_sampled, valid_df.reset_index(drop=True)


model_name = f"microsoft/{llm_name}-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# convert code into tokens and then vector representation
def tokenize_data(train_text, val_text, test_text):

    tokens_train = tokenizer.batch_encode_plus(
        train_text.tolist(),
        max_length=510,
        pad_to_max_length=True,
        truncation=True)

    tokens_val = tokenizer.batch_encode_plus(
        val_text.tolist(),
        max_length=510,
        pad_to_max_length=True,
        truncation=True)

    tokens_test = tokenizer.batch_encode_plus(
        test_text.tolist(),
        max_length=510,
        pad_to_max_length=True,
        truncation=True)
    return tokens_train, tokens_val, tokens_test


def text_to_tensors(tokens_train, tokens_val, tokens_test):
    train_seq = torch.tensor(tokens_train['input_ids'])
    train_mask = torch.tensor(tokens_train['attention_mask'])

    val_seq = torch.tensor(tokens_val['input_ids'])
    val_mask = torch.tensor(tokens_val['attention_mask'])

    test_seq = torch.tensor(tokens_test['input_ids'])
    test_mask = torch.tensor(tokens_test['attention_mask'])

    return train_seq, train_mask, val_seq, val_mask, test_seq, test_mask


# sett seed for data_loaders for output reproducibility
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


g = torch.Generator()
seed = 42 # any number 
set_deterministic(seed)

def data_loaders(train_seq, val_seq, test_seq, train_mask, val_mask, test_mask, train_y, val_y, test_y):
    batch_size = 4

    train_dataset = MultiInputDataset(train_seq, train_mask, train_y)
    val_dataset = MultiInputDataset(val_seq, val_mask, val_y)
    test_dataset = MultiInputDataset(test_seq, test_mask, test_y)

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

        seqs, masks, labels = batch

        seqs = seqs.to(device)
        masks = masks.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        preds = model(seqs, masks)
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

        seqs, masks, labels = batch

        seqs = seqs.to(device)
        masks = masks.to(device)
        labels = labels.to(device)

        with torch.no_grad():

            preds = model(seqs, masks)
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

        seqs, masks, labels = batch  

        seqs = seqs.to(device)
        masks = masks.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            preds = model(seqs, masks)
            preds = preds.detach().cpu().numpy()
            all_preds.append(preds)

    all_preds = np.concatenate(all_preds, axis=0)

    return np.argmax(all_preds, axis=1)


execution_time = time.time()
print("Start time of the experiment", execution_time)
result = pd.DataFrame(columns = ['project_name','Accuracy','F1', 'Precision', 'Recall', 'TN', 'FP', 'FN', 'TP'])
TN = FP = FN = TP = 0
all_fold_test_results = []
x='full_code'
y='flaky'
z = 'test_case_name'

for i in project_name:
    print('testing on project: ', i)
    project_Name=i

    train_dataset=  df.loc[(df['project'] != i)]
    test_dataset= df.loc[(df['project']== i)]
    
    X_train, X_valid, y_train, y_valid, case_train, case_valid=train_test_split(train_dataset[x], train_dataset[y], train_dataset[z], random_state=seed, test_size=0.2, stratify=train_dataset[y])
    
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
        'code': test_dataset[x],
        'label': test_dataset[y],
        'test_case_name': test_dataset[z]
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
    y_test = pd.DataFrame(test_df['label'])

    Y_train.columns = ['which_tests']
    y_val.columns = ['which_tests']
    y_test.columns = ['which_tests']

    train_y = torch.tensor(Y_train['which_tests'].values)
    val_y = torch.tensor(y_val['which_tests'].values)
    test_y = torch.tensor(y_test['which_tests'].values)

    tokens_train, tokens_val, tokens_test = tokenize_data(
        X_train_sampled, X_valid_sampled, test_df['code'])
    
    train_seq, train_mask, val_seq, val_mask, test_seq, test_mask = text_to_tensors(tokens_train, tokens_val, tokens_test)

    train_dataloader, val_dataloader, test_dataloader = data_loaders(
    train_seq, val_seq, test_seq, 
    train_mask, val_mask, test_mask,
    train_y, val_y, test_y
)

    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(Y_train.values), y=np.ravel(Y_train.values))
    weights = torch.tensor(class_weights, dtype=torch.float)
    weights = weights.to(device)
    cross_entropy = nn.NLLLoss(weight=weights)
    print("Class Weights:", class_weights)

    epochs = 30

    model = BERT_Arch(model_name).to(device)

    optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)

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
            torch.save(model.state_dict(), f"../results/best_model_{i}.pt")
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        print(f'\nTraining Loss: {train_loss:.3f}')
        print(f'Validation Loss: {valid_loss:.3f}')

    model.load_state_dict(torch.load(f"../results/best_model_{i}.pt"))
    preds = test(model, test_dataloader)
    test_case_names = test_df['test_case_name'].values
    true_labels = test_y.cpu().numpy() if torch.is_tensor(test_y) else np.array(test_y)

    fold_result = pd.DataFrame({
        "project": i,
        "test_case_name": test_case_names,
        "true_label": true_labels,
        "pred_label": preds
    })
    all_fold_test_results.append(fold_result)

    print(classification_report(test_y, preds))
    TN, FP, FN, TP = confusion_matrix(test_y, preds, labels=[0, 1]).ravel()

    del model
    torch.cuda.empty_cache()

    accuracy, F1, Precision, Recall = get_evaluation_scores(TN, FP, FN, TP)

    accuracy, F1, Precision, Recall = get_evaluation_scores(TN, FP, FN, TP)

    print('accuracy, F1, Precision, Recall',accuracy, F1, Precision, Recall)

    new_row = pd.DataFrame([[project_Name, accuracy, F1, Precision, Recall, TN, FP, FN, TP]], 
                        columns=result.columns)
    result = pd.concat([result, new_row], ignore_index=True)

result.to_csv(results_file,  index=False)