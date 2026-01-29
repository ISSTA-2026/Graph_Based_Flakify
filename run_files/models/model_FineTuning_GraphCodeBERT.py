import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

device = torch.device("cuda")

class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(768, 512)
        self.dropout = nn.Dropout(0.1)
        self.out_proj = nn.Linear(512, 2)
        self.relu = nn.ReLU()

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = x.reshape(-1,x.size(-1))
        x = self.dense(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
        
class Model(nn.Module):   
    def __init__(self):
        super(Model, self).__init__()
        model_name = "microsoft/graphcodebert-base"
        self.config=AutoConfig.from_pretrained(model_name)
        self.config.num_labels=2
        self.encoder = AutoModelForSequenceClassification.from_pretrained(model_name,config=self.config)  
        self.tokenizer=AutoTokenizer.from_pretrained(model_name)
        self.classifier=RobertaClassificationHead(self.config)
    
        
    def forward(self, inputs_ids,position_idx,attn_mask): 

        #embedding
        nodes_mask=position_idx.eq(0)
        token_mask=position_idx.ge(2)        
        inputs_embeddings=self.encoder.roberta.embeddings.word_embeddings(inputs_ids)
        nodes_to_token_mask=nodes_mask[:,:,None]&token_mask[:,None,:]&attn_mask
        nodes_to_token_mask=nodes_to_token_mask/(nodes_to_token_mask.sum(-1)+1e-10)[:,:,None]
        avg_embeddings=torch.einsum("abc,acd->abd",nodes_to_token_mask,inputs_embeddings)
        inputs_embeddings=inputs_embeddings*(~nodes_mask)[:,:,None]+avg_embeddings*nodes_mask[:,:,None]
        outputs = self.encoder.roberta(inputs_embeds=inputs_embeddings,attention_mask=attn_mask,position_ids=position_idx, token_type_ids=torch.zeros((position_idx.size()), dtype=torch.long, device=device))[0]
        final=self.classifier(outputs)
        softmax=nn.LogSoftmax(dim=-1)
        final_output = softmax(final)
        return final_output

    