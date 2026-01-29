import torch.nn as nn
from transformers import AutoModel, AutoConfig

# set up the neural network for fine-tuning
class BERT_Arch(nn.Module):

    def __init__(self, model_name):

        super(BERT_Arch, self).__init__()

        model_config = AutoConfig.from_pretrained(model_name, return_dict=True, output_hidden_states=True)
        self.codebert = AutoModel.from_pretrained(model_name, config=model_config)

        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 2)
        self.softmax = nn.LogSoftmax(dim=-1)

    # define the forward pass
    def forward(self, sent_id, mask):
        codebert_output = self.codebert(input_ids=sent_id, attention_mask=mask)
        codebert_cls = codebert_output.last_hidden_state[:, 0, :] 
        fc1_output = self.fc1(codebert_cls)
        relu_output = self.relu(fc1_output)
        dropout_output = self.dropout(relu_output)
        fc2_output = self.fc2(dropout_output)
        final_output = self.softmax(fc2_output)
        return final_output