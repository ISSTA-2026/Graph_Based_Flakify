from __future__ import absolute_import, division, print_function
import time
import gc
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer
from tqdm import tqdm
import argparse
from parser import DFG_python,DFG_java,DFG_ruby,DFG_go,DFG_php,DFG_javascript
from parser import (remove_comments_and_docstrings, tree_to_token_index, index_to_code_token)
from tree_sitter import Language, Parser
from models.model_FineTuning_GraphCodeBERT import Model

def set_deterministic(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)                    
    torch.cuda.manual_seed(seed)               
    torch.cuda.manual_seed_all(seed)           
    torch.backends.cudnn.deterministic = True 

# specify GPU
device = torch.device("cuda")

parser = argparse.ArgumentParser(description="Select Dataset and Model Name")
parser.add_argument("dataset", type=str, choices=["FlakeFlagger", "IDoFT"],
                    help="Specify the dataset name (FlakeFlagger or IDoFT)")
args = parser.parse_args()

dataset_name = args.dataset

# choose dataset
dataset_path = f"../dataset/Flakify_{dataset_name}_dataset.csv"
results_file = "../results/Flakify_cross_validation_results.csv"

df = pd.read_csv(dataset_path)
input_data = df['full_code'] 
target_data = df['flaky']
case_names = df['test_case_name']
df.head()

model_name = "microsoft/graphcodebert-base"
model_tokenizer = AutoTokenizer.from_pretrained(model_name) 

def sampling(X_train, y_train, X_valid, y_valid):
    oversampling = RandomOverSampler(
        sampling_strategy='minority', random_state=seed)
    
    x_train = X_train.values.reshape(-1, 1)
    y_train = y_train.values.reshape(-1, 1)
    x_val = X_valid.values.reshape(-1, 1)
    y_val = y_valid.values.reshape(-1, 1)

    x_train_os, y_train_os = oversampling.fit_resample(x_train, y_train)
    x_train_os = pd.Series(x_train_os.ravel())
    y_train_os = pd.Series(y_train_os.ravel())
    x_val = pd.Series(x_val.ravel())
    y_val = pd.Series(y_val.ravel())

    return x_train_os, y_train_os, x_val, y_val

dfg_function={
    'python':DFG_python,
    'java':DFG_java,
    'ruby':DFG_ruby,
    'go':DFG_go,
    'php':DFG_php,
    'javascript':DFG_javascript
}

parsers={}        
for lang in dfg_function:
    LANGUAGE = Language('parser/my-languages.so', lang)
    parser = Parser()
    parser.set_language(LANGUAGE) 
    parser = [parser,dfg_function[lang]]    
    parsers[lang]= parser
    
    
#remove comments, tokenize code and extract dataflow                                        
def extract_dataflow(code, parser,lang):
    #remove comments
    try:
        code=remove_comments_and_docstrings(code,lang)
    except:
        pass    
    #obtain dataflow
    if lang=="php":
        code="<?php"+code+"?>"    
    try:
        tree = parser[0].parse(bytes(code,'utf8'))    
        root_node = tree.root_node  
        tokens_index=tree_to_token_index(root_node)     
        code=code.split('\n')
        code_tokens=[index_to_code_token(x,code) for x in tokens_index]  
        index_to_code={}
        for idx,(index,code) in enumerate(zip(tokens_index,code_tokens)):
            index_to_code[index]=(idx,code)  
        try:
            DFG,_=parser[1](root_node,index_to_code,{}) 
        except:
            DFG=[]
        DFG=sorted(DFG,key=lambda x:x[1])
        indexs=set()
        for d in DFG:
            if len(d[-1])!=0:
                indexs.add(d[1])
            for x in d[-1]:
                indexs.add(x)
        new_DFG=[]
        for d in DFG:
            if d[1] in indexs:
                new_DFG.append(d)
        dfg=new_DFG
    except:
        dfg=[]
    return code_tokens,dfg


class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
             input_tokens,
             input_ids,
             position_idx,
             dfg_to_code,
             dfg_to_dfg,
             label

    ):
        #The first code function
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.position_idx=position_idx
        self.dfg_to_code=dfg_to_code
        self.dfg_to_dfg=dfg_to_dfg
        self.label=label
    

code_length = 512
data_flow_length = 128

def convert_examples_to_features(item):
    #source
    func, tokenizer, label=item
    parser=parsers['java']
    
    #extract data flow
    code_tokens,dfg=extract_dataflow(func,parser,'java')
    code_tokens=[tokenizer.tokenize('@ '+x)[1:] if idx!=0 else tokenizer.tokenize(x) for idx,x in enumerate(code_tokens)]
    ori2cur_pos={}
    ori2cur_pos[-1]=(0,0)
    for i in range(len(code_tokens)):
        ori2cur_pos[i]=(ori2cur_pos[i-1][1],ori2cur_pos[i-1][1]+len(code_tokens[i]))    
    code_tokens=[y for x in code_tokens for y in x]  
    
    #truncating
    code_tokens=code_tokens[:code_length+data_flow_length-3-min(len(dfg),data_flow_length)][:512-3]
    source_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
    source_ids =  tokenizer.convert_tokens_to_ids(source_tokens)
    position_idx = [i+tokenizer.pad_token_id + 1 for i in range(len(source_tokens))]
    dfg=dfg[:code_length+data_flow_length-len(source_tokens)]
    source_tokens+=[x[0] for x in dfg]
    position_idx+=[0 for x in dfg]
    source_ids+=[tokenizer.unk_token_id for x in dfg]
    padding_length=code_length+data_flow_length-len(source_ids)
    position_idx+=[tokenizer.pad_token_id]*padding_length
    source_ids+=[tokenizer.pad_token_id]*padding_length      
    
    #reindex
    reverse_index={}
    for idx,x in enumerate(dfg):
        reverse_index[x[1]]=idx
    for idx,x in enumerate(dfg):
        dfg[idx]=x[:-1]+([reverse_index[i] for i in x[-1] if i in reverse_index],)    
    dfg_to_dfg=[x[-1] for x in dfg]
    dfg_to_code=[ori2cur_pos[x[1]] for x in dfg]
    length=len([tokenizer.cls_token])
    dfg_to_code=[(x[0]+length,x[1]+length) for x in dfg_to_code]        

    return InputFeatures(source_tokens,source_ids,position_idx,dfg_to_code,dfg_to_dfg,label)

class TextDataset(Dataset):
    def __init__(self, tokenizer, dataset,labels):
        self.examples = []
        dataset.tolist()

        #load code function according to index
        data=[]
        #for func, label_idx in dataset, label:
        for func, label in zip(dataset, labels):
            data.append((func, tokenizer, label))
            
        #convert example to input features    
        self.examples=[convert_examples_to_features(x) for x in tqdm(data,total=len(data))]
        

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, item):
        #calculate graph-guided masked function
        attn_mask= np.zeros((code_length+data_flow_length,
                        code_length+data_flow_length), dtype = bool)
        #calculate begin index of node and max length of input
        node_index=sum([i>1 for i in self.examples[item].position_idx])
        max_length=sum([i!=1 for i in self.examples[item].position_idx])
        #sequence can attend to sequence
        attn_mask[:node_index,:node_index]=True
        #special tokens attend to all tokens
        for idx,i in enumerate(self.examples[item].input_ids):
            if i in [0,2]:
                attn_mask[idx,:max_length]=True
        #nodes attend to code tokens that are identified from
        for idx,(a,b) in enumerate(self.examples[item].dfg_to_code):
            if a<node_index and b<node_index:
                attn_mask[idx+node_index,a:b]=True
                attn_mask[a:b,idx+node_index]=True
        #nodes attend to adjacent nodes 
        for idx,nodes in enumerate(self.examples[item].dfg_to_dfg):
            for a in nodes:
                if a+node_index<len(self.examples[item].position_idx):
                    attn_mask[idx+node_index,a+node_index]=True      

        return (torch.tensor(self.examples[item].input_ids),
                torch.tensor(self.examples[item].position_idx),
                torch.tensor(attn_mask)         
                ,torch.tensor(self.examples[item].label))


g = torch.Generator()
seed = 42 # any number 
set_deterministic(seed)

# sett seed for data_loaders for output reproducibility
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def data_loaders(train_item, val_item, test_item):
    # define a batch size
    batch_size = 4

    # sampler for sampling the data during training
    train_sampler = RandomSampler(train_item)

    # dataLoader for train set
    train_dataloader = DataLoader(
        train_item, sampler=train_sampler, batch_size=batch_size, worker_init_fn=seed_worker)

    # sampler for sampling the data during training
    val_sampler = SequentialSampler(val_item)

    # dataLoader for validation set
    val_dataloader = DataLoader(val_item, sampler=val_sampler, batch_size=batch_size, worker_init_fn=seed_worker)

    # sampler for sampling the data during training
    test_sampler = SequentialSampler(test_item)

    # dataLoader for validation set
    test_dataloader = DataLoader(test_item, sampler=test_sampler, batch_size=batch_size, worker_init_fn=seed_worker)

    return train_dataloader, val_dataloader, test_dataloader


# train the model
def train():

    model.train()

    total_loss = 0
    total_preds = []

    for batch in tqdm(train_dataloader):

        batch = [t.to(device) for t in batch]

        sent_id, pos, mask, labels = batch

        optimizer.zero_grad()
        preds = model(sent_id, pos, mask)
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

# evaluate the model
def evaluate():

    print("\nEvaluating..")

    model.eval()

    total_loss = 0
    total_preds = []

    for batch in tqdm(val_dataloader):

        batch = [t.to(device) for t in batch]

        sent_id, pos, mask, labels = batch

        with torch.no_grad():

            preds = model(sent_id, pos, mask)
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

# give test data to the model in chunks to avoid Cuda out of memory error
def test(model, test_dataloader):
    print("\nPredicting on test data...")

    model.eval()
    all_preds = []

    for batch in tqdm(test_dataloader):

        batch = [t.to(device) for t in batch]

        sent_id, pos, mask, labels = batch

        with torch.no_grad():
            preds = model(sent_id, pos, mask)
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
    
    X_train, X_valid, y_train, y_valid=train_test_split(X_train, y_train, random_state=seed, test_size=0.2, stratify=y_train)

    test_df = pd.DataFrame({
        'code': X_test,
        'label': y_test,
        'test_case_name': case_test
    })
    
    X_train, y_train, X_valid, y_valid = sampling(
        X_train, y_train, X_valid, y_valid)

    Y_train = pd.DataFrame(y_train)
    y_val = pd.DataFrame(y_valid)
    y_test = pd.DataFrame(y_test)

    Y_train.columns = ['flaky']
    y_val.columns = ['flaky']
    y_test.columns = ['flaky']

    # convert labels of train, validation and test into tensors
    train_y = torch.tensor(Y_train['flaky'].values)
    val_y = torch.tensor(y_val['flaky'].values)
    test_y = torch.tensor(y_test['flaky'].values)


    train_item = TextDataset(model_tokenizer, X_train, Y_train['flaky'].values)
    val_item = TextDataset(model_tokenizer, X_valid, y_val['flaky'].values)
    test_item = TextDataset(model_tokenizer, X_test, y_test['flaky'].values)


    # create data_loaders for train and validation dataset
    train_dataloader, val_dataloader, test_dataloader = data_loaders(train_item, val_item, test_item)

    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(Y_train.values), y=np.ravel(Y_train.values))
    weights = torch.tensor(class_weights, dtype=torch.float)
    weights = weights.to(device)
    cross_entropy = nn.NLLLoss(weight=weights)
    print("Class Weights:", class_weights)

    epochs = 30

    model = Model().to(device)

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

    # for each epoch
    for epoch in range(epochs):

        print('\n Epoch {:} / {:}'.format(epoch + 1, epochs))

        # train the model
        train_loss, _ = train()

        # evaluate the model
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

        # append training and validation loss
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        print(f'\nTraining Loss: {train_loss:.3f}')
        print(f'Validation Loss: {valid_loss:.3f}')

    # load weights of best model
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
all_fold_results_df.to_csv("../results/GraphCodeBERT_test_fold_predictions.csv", index=False)

print("The processed is completed in : (%s) seconds. " % round((time.time() - execution_time), 5))