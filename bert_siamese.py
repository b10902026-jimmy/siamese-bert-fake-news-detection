import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from random import randrange
import torch.nn.functional as F
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt
from argparse import ArgumentParser
import sys
from datetime import datetime
from tqdm import tqdm
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay


parser = ArgumentParser()
parser.add_argument('-num_labels', action="store", dest="num_labels", type=int)
args = parser.parse_args()

num_labels = args.num_labels

#home = str(Path.home())

train_path = './LIAR-PLUS/dataset/train2.tsv'
test_path = './LIAR-PLUS/dataset/test2.tsv'
val_path = './LIAR-PLUS/dataset/val2.tsv'


train_df = pd.read_csv(train_path, sep="\t", header=None)
test_df = pd.read_csv(test_path, sep="\t", header=None)
val_df = pd.read_csv(val_path, sep="\t", header=None)

# Fill nan (empty boxes) with 0
train_df = train_df.fillna(0)
test_df = test_df.fillna(0)
val_df = val_df.fillna(0)

train = train_df.values
test = test_df.values
val = val_df.values


labels = {'train':[train[i][2] for i in range(len(train))], 'test':[test[i][2] for i in range(len(test))], 'val':[val[i][2] for i in range(len(val))]}
statements = {'train':[train[i][3] for i in range(len(train))], 'test':[test[i][3] for i in range(len(test))], 'val':[val[i][3] for i in range(len(val))]}
subjects = {'train':[train[i][4] for i in range(len(train))], 'test':[test[i][4] for i in range(len(test))], 'val':[val[i][4] for i in range(len(val))]}
speakers = {'train':[train[i][5] for i in range(len(train))], 'test':[test[i][5] for i in range(len(test))], 'val':[val[i][5] for i in range(len(val))]}
jobs = {'train':[train[i][6] for i in range(len(train))], 'test':[test[i][6] for i in range(len(test))], 'val':[val[i][6] for i in range(len(val))]}
states = {'train':[train[i][7] for i in range(len(train))], 'test':[test[i][7] for i in range(len(test))], 'val':[val[i][7] for i in range(len(val))]}
affiliations = {'train':[train[i][8] for i in range(len(train))], 'test':[test[i][8] for i in range(len(test))], 'val':[val[i][8] for i in range(len(val))]}
credits = {'train':[train[i][9:14] for i in range(len(train))], 'test':[test[i][9:14] for i in range(len(test))], 'val':[val[i][9:14] for i in range(len(val))]}
contexts = {'train':[train[i][14] for i in range(len(train))], 'test':[test[i][14] for i in range(len(test))], 'val':[val[i][14] for i in range(len(val))]}
justification = {'train':[train[i][15] for i in range(len(train))], 'test':[test[i][15] for i in range(len(test))], 'val':[val[i][15] for i in range(len(val))]}

if num_labels == 6:

    def to_onehot(a):
        a_cat = [0]*len(a)
        for i in range(len(a)):
            if a[i]=='true':
                a_cat[i] = [1,0,0,0,0,0]
            elif a[i]=='mostly-true':
                a_cat[i] = [0,1,0,0,0,0]
            elif a[i]=='half-true':
                a_cat[i] = [0,0,1,0,0,0]
            elif a[i]=='barely-true':
                a_cat[i] = [0,0,0,1,0,0]
            elif a[i]=='false':
                a_cat[i] = [0,0,0,0,1,0]
            elif a[i]=='pants-fire':
                a_cat[i] = [0,0,0,0,0,1]
            else:
                print('Incorrect label')
        return a_cat

elif num_labels == 2:

    def to_onehot(a):
        a_cat = [0]*len(a)
        for i in range(len(a)):
            if a[i]=='true':
                a_cat[i] = [1,0]
            elif a[i]=='mostly-true':
                a_cat[i] = [1,0]
            elif a[i]=='half-true':
                a_cat[i] = [1,0]
            elif a[i]=='barely-true':
                a_cat[i] = [0,1]
            elif a[i]=='false':
                a_cat[i] = [0,1]
            elif a[i]=='pants-fire':
                a_cat[i] = [0,1]
            else:
                print('Incorrect label')
        return a_cat

else:

    print('Invalid number of labels. The number of labels should be either 2 or 6')

    sys.exit()


labels_onehot = {'train':to_onehot(labels['train']), 'test':to_onehot(labels['test']), 'val':to_onehot(labels['val'])}


# Preparing meta data

#credit['train'][2]

metadata = {'train':[0]*len(train), 'val':[0]*len(val), 'test':[0]*len(test)}

for i in range(len(train)):
    subject = subjects['train'][i]
    if subject == 0:
        subject = 'None'

    speaker = speakers['train'][i]
    if speaker == 0:
        speaker = 'None'

    job = jobs['train'][i]
    if job == 0:
        job = 'None'

    state = states['train'][i]
    if state == 0:
        state = 'None'

    affiliation = affiliations['train'][i]
    if affiliation == 0:
        affiliation = 'None'

    context = contexts['train'][i]
    if context == 0 :
        context = 'None'

    meta = subject + ' ' + speaker + ' ' + job + ' ' + state + ' ' + affiliation + ' ' + context

    metadata['train'][i] = meta

for i in range(len(val)):
    subject = subjects['val'][i]
    if subject == 0:
        subject = 'None'

    speaker = speakers['val'][i]
    if speaker == 0:
        speaker = 'None'

    job = jobs['val'][i]
    if job == 0:
        job = 'None'

    state = states['val'][i]
    if state == 0:
        state = 'None'

    affiliation = affiliations['val'][i]
    if affiliation == 0:
        affiliation = 'None'

    context = contexts['val'][i]
    if context == 0 :
        context = 'None'

    meta = subject + ' ' + speaker + ' ' + job + ' ' + state + ' ' + affiliation + ' ' + context

    metadata['val'][i] = meta

for i in range(len(test)):
    subject = subjects['test'][i]
    if subject == 0:
        subject = 'None'

    speaker = speakers['test'][i]
    if speaker == 0:
        speaker = 'None'

    job = jobs['test'][i]
    if job == 0:
        job = 'None'

    state = states['test'][i]
    if state == 0:
        state = 'None'

    affiliation = affiliations['test'][i]
    if affiliation == 0:
        affiliation = 'None'

    context = contexts['test'][i]
    if context == 0 :
        context = 'None'

    meta = subject + ' ' + speaker + ' ' + job + ' ' + state + ' ' + affiliation + ' ' + context

    metadata['test'][i] = meta


# Credit score calculation
credit_score = {'train':[0]*len(train), 'val':[0]*len(val), 'test':[0]*len(test)}
for i in range(len(train)):
    credit = credits['train'][i]
    if sum(credit) == 0:
        score = 0.5
    else:
        score = (credit[3]*0.2 + credit[2]*0.5 + credit[0]*0.75 + credit[1]*0.9 + credit[4]*1)/(sum(credit))
    credit_score['train'][i] = [score for i in range(2304)]

for i in range(len(val)):
    credit = credits['val'][i]
    if sum(credit) == 0:
        score = 0.5
    else:
        score = (credit[3]*0.2 + credit[2]*0.5 + credit[0]*0.75 + credit[1]*0.9 + credit[4]*1)/(sum(credit))
    credit_score['val'][i] = [score for i in range(2304)]

for i in range(len(test)):
    credit = credits['test'][i]
    if sum(credit) == 0:
        score = 0.5
    else:
        score = (credit[3]*0.2 + credit[2]*0.5 + credit[0]*0.75 + credit[1]*0.9 + credit[4]*1)/(sum(credit))
    credit_score['test'][i] = [score for i in range(2304)]


class BertLayerNorm(nn.Module):
        def __init__(self, hidden_size, eps=1e-12):
            """Construct a layernorm module in the TF style (epsilon inside the square root).
            """
            super(BertLayerNorm, self).__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.bias = nn.Parameter(torch.zeros(hidden_size))
            self.variance_epsilon = eps

        def forward(self, x):
            u = x.mean(-1, keepdim=True)
            s = (x - u).pow(2).mean(-1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.variance_epsilon)
            return self.weight * x + self.bias


class BertForSequenceClassification(nn.Module):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.
    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `num_labels`: the number of classes for the classifier. Default = 2.
    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary. Items in the batch should begin with the special "CLS" token. (see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
            with indices selected in [0, ..., num_labels].
    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, num_labels].
    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
    num_labels = 2
    model = BertForSequenceClassification(config, num_labels)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, num_labels=2): # Change number of labels here.
        super(BertForSequenceClassification, self).__init__()
        self.num_labels = num_labels
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size*3, num_labels)
        #self.fc1 = nn.Linear(config.hidden_size*2, 512)
        nn.init.xavier_normal_(self.classifier.weight)

    '''def forward_once(self, x):
        # Forward pass
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output'''

    def forward_once(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        #logits = self.classifier(pooled_output)

        return pooled_output

    def forward(self, input_ids1, input_ids2, input_ids3, credit_sc):
        output1 = self.forward_once(input_ids1)
        output2 = self.forward_once(input_ids2)
        output3 = self.forward_once(input_ids3)

        out = torch.cat((output1, output2, output3), 1)

        # ✅ Debug device mismatch
        if not hasattr(self, 'printed_debug_info') or self.printed_debug_info is False:
            print("[DEBUG] credit_sc device:", credit_sc.device)
            print("[DEBUG] bert output (concat) device:", out.device)
            self.printed_debug_info = True
        # ✅ Device consistency check — 強制檢查 credit_sc 裝置
        if credit_sc.device != out.device:
            raise RuntimeError(f"[Device Mismatch] credit_sc on {credit_sc.device}, but BERT output on {out.device}")

        out = torch.add(credit_sc, out)
        logits = self.classifier(out)

        return logits


    def freeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = True


from pytorch_pretrained_bert import BertConfig

config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)


model = BertForSequenceClassification(num_labels)


# Loading the statements
X_train = statements['train']
y_train = labels_onehot['train']

X_val = statements['val']
y_val = labels_onehot['val']

X_train = X_train + X_val
y_train = y_train + y_val


X_test = statements['test']
y_test = labels_onehot['test']

# Loading the justification
X_train_just = justification['train']

X_val_just = justification['val']

X_train_just = X_train_just + X_val_just

X_test_just = statements['test']


# Loading the meta data
X_train_meta = metadata['train']
X_val_meta = metadata['val']
X_train_meta = X_train_meta + X_val_meta
X_test_meta = metadata['test']

# Loading Credit scores

X_train_credit = credit_score['train']
X_val_credit = credit_score['val']
X_train_credit = X_train_credit+X_val_credit
X_test_credit = credit_score['test']


# Small data partitioned for debugging
'''X_train = X_train[:100]
y_train = y_train[:100]

X_test = X_test[:100]
y_test = y_test[:100]

X_train_just = X_train_just[:100]
X_test_just = X_test_just[:100]

X_train_meta = X_train_meta[:100]
X_test_meta = X_test_meta[:100]

X_train_credit = X_train_credit[:100]
X_test_credit = X_test_credit[:100]'''

max_seq_length_stat = 64
max_seq_length_just = 256
max_seq_length_meta = 32

class text_dataset(Dataset):
    def __init__(self,x_y_list, transform=None):

        self.x_y_list = x_y_list
        self.transform = transform

    def __getitem__(self,index):

        # Tokenize statements
        tokenized_review = tokenizer.tokenize(self.x_y_list[0][index])

        if len(tokenized_review) > max_seq_length_stat:
            tokenized_review = tokenized_review[:max_seq_length_stat]

        ids_review  = tokenizer.convert_tokens_to_ids(tokenized_review)

        padding = [0] * (max_seq_length_stat - len(ids_review))

        ids_review += padding

        assert len(ids_review) == max_seq_length_stat

        #print(ids_review)
        ids_review = torch.tensor(ids_review)

        fakeness = self.x_y_list[4][index] # color
        list_of_labels = [torch.from_numpy(np.array(fakeness))]


        # Tokenize justifications
        #print(self.x_y_list[1][6833])
        #print(index)

        # Making sure that if there is no justification in a row(nan value converted to 0 using pandas), give it a justification called 'No justification' for training to be possible.
        if self.x_y_list[1][index] == 0:
            self.x_y_list[1][index] = 'No justification'

        tokenized_review_just = tokenizer.tokenize(self.x_y_list[1][index])

        if len(tokenized_review_just) > max_seq_length_just:
            tokenized_review_just = tokenized_review_just[:max_seq_length_just]

        ids_review_just  = tokenizer.convert_tokens_to_ids(tokenized_review_just)

        padding = [0] * (max_seq_length_just - len(ids_review_just))

        ids_review_just += padding

        assert len(ids_review_just) == max_seq_length_just

        #print(ids_review)
        ids_review_just = torch.tensor(ids_review_just)

        fakeness = self.x_y_list[4][index] # color
        list_of_labels = [torch.from_numpy(np.array(fakeness))]

        # Tokenize metadata

        tokenized_review_meta = tokenizer.tokenize(self.x_y_list[2][index])

        if len(tokenized_review_meta) > max_seq_length_meta:
            tokenized_review_meta = tokenized_review_meta[:max_seq_length_meta]

        ids_review_meta  = tokenizer.convert_tokens_to_ids(tokenized_review_meta)

        padding = [0] * (max_seq_length_meta - len(ids_review_meta))

        ids_review_meta += padding

        assert len(ids_review_meta) == max_seq_length_meta

        #print(ids_review)
        ids_review_meta = torch.tensor(ids_review_meta)

        fakeness = self.x_y_list[4][index] # color
        list_of_labels = [torch.from_numpy(np.array(fakeness))]

        credit_scr = self.x_y_list[3][index] # Credit score

        #ones_768 = np.ones((768))

        # 明確指定 dtype 和 device
        credit_scr = torch.tensor(self.x_y_list[3][index], dtype=torch.float32, device=device)


        return [ids_review, ids_review_just, ids_review_meta, credit_scr], list_of_labels[0]

    def __len__(self):
        return len(self.x_y_list[0])


batch_size = 32

# Train Statements and Justifications
train_lists = [X_train, X_train_just, X_train_meta, X_train_credit, y_train]

# Test Statements and Justifications
test_lists = [X_test, X_test_just, X_train_meta, X_test_credit, y_test]

# Preparing the data (Tokenize)
training_dataset = text_dataset(x_y_list = train_lists)
test_dataset = text_dataset(x_y_list = test_lists)


# Prepare the training dictionaries
dataloaders_dict = {'train': torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=0),
                   'val':torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
                   }
dataset_sizes = {'train':len(train_lists[0]),
                'val':len(test_lists[0])}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

train_acc = []
val_acc = []
train_loss = []
val_loss = []

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')  # 動態圖檔命名
    save_dir = f"training_results_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    print(f"[Info] Created results folder: {save_dir}")

    print('starting')
    printed_debug_info = False  # ✅ 這裡
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 100
    best_acc = 0

    for epoch in range(num_epochs):
        epoch_start = time.time()
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)


        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            fakeness_corrects = 0

            print(f"[{phase.upper()}] Processing...")
            for inputs, fakeness in tqdm(dataloaders_dict[phase], desc=f"{phase}"):

                inputs1 = inputs[0].to(device)
                inputs2 = inputs[1].to(device)
                inputs3 = inputs[2].to(device)
                inputs4 = inputs[3].to(device)
                inputs4 = inputs4.to(device).float()
                fakeness = fakeness.to(device)
                if not printed_debug_info:
                    print("[DEBUG] Device Check — inputs4 (credit_scr):", inputs4.device)
                    print("[DEBUG] Device Check — model param device:", next(model.parameters()).device)
                    printed_debug_info = True


                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs1, inputs2, inputs3, inputs4)
                    outputs = F.softmax(outputs, dim=1)
                    loss = criterion(outputs, torch.max(fakeness.float(), 1)[1])

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs1.size(0)
                fakeness_corrects += torch.sum(torch.max(outputs, 1)[1] == torch.max(fakeness, 1)[1])

            epoch_loss = running_loss / dataset_sizes[phase]
            fakeness_acc = fakeness_corrects.double() / dataset_sizes[phase]

            print('{} total loss: {:.4f} '.format(phase, epoch_loss))
            print('{} fakeness_acc: {:.4f}'.format(phase, fakeness_acc))

            # ✅ append 正確分類 acc/loss
            if phase == 'train':
                train_acc.append(fakeness_acc.cpu().numpy())
                train_loss.append(epoch_loss)
                scheduler.step()  # <-- 建議在每個 epoch 結束時調整學習率
            else:
                val_acc.append(fakeness_acc.cpu().numpy())
                val_loss.append(epoch_loss)

            # ✅ Save model if best
            if phase == 'val' and fakeness_acc > best_acc:
                print('Saving with accuracy of {}'.format(fakeness_acc),
                    'improved over previous {}'.format(best_acc))
                best_acc = fakeness_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))

        print('Time taken for epoch {} is {:.2f} minutes'.format(epoch + 1, (time.time() - epoch_start) / 60))
        print()


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(float(best_acc)))

    # load best model weights
    model.load_state_dict(best_model_wts)
        # Plot accuracy
    plt.figure()
    plt.plot(val_acc, label='Validation Accuracy')
    plt.plot(train_acc, label='Training Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(save_dir, f'accuracy_plot.png'))
    plt.close()
    print("[Info] Saved accuracy plot to accuracy_plot.png")

    # Plot loss
    plt.figure()
    plt.plot(val_loss, label='Validation Loss')
    plt.plot(train_loss, label='Training Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(save_dir, f'loss_plot.png'))

    plt.close()
    print("[Info] Saved loss plot to loss_plot.png")

    return model, train_acc, val_acc, train_loss, val_loss

model.to(device)


lrlast = .0001
lrmain = .00001
optim1 = optim.Adam(
    [
        {"params":model.bert.parameters(),"lr": lrmain},
        {"params":model.classifier.parameters(), "lr": lrlast},

   ])

#optim1 = optim.Adam(model.parameters(), lr=0.001)#,momentum=.9)
# Observe that all parameters are being optimized
optimizer_ft = optim1
criterion = nn.CrossEntropyLoss()

'''import focal_loss
loss_args = {"alpha": 0.5, "gamma": 2.0}
criterion = focal_loss.FocalLoss(*loss_args)'''

# Decay LR by a factor of 0.1 every 3 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=3, gamma=0.1)


model_ft1, train_acc, val_acc, train_loss, val_loss = train_model(model, criterion, optimizer_ft, exp_lr_scheduler,num_epochs=1)

# ===== Model Analysis and Visualization =====

def evaluate_model_with_visualizations(model, dataloader, save_dir="model_analysis"):
    """
    Comprehensive model evaluation with visualizations for presentation
    """
    os.makedirs(save_dir, exist_ok=True)
    print(f"[Info] Created analysis folder: {save_dir}")
    
    # Set model to evaluation mode
    model.eval()
    
    # Lists to store predictions and ground truth
    all_preds = []
    all_labels = []
    all_statements = []
    all_justifications = []
    all_probs = []
    
    # Process validation data
    print("Generating predictions for analysis...")
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluation"):
            inputs1 = inputs[0].to(device)
            inputs2 = inputs[1].to(device)
            inputs3 = inputs[2].to(device)
            inputs4 = inputs[3].to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(inputs1, inputs2, inputs3, inputs4)
            probs = F.softmax(outputs, dim=1)
            
            # Get predictions
            _, preds = torch.max(outputs, 1)
            _, true_labels = torch.max(labels, 1)
            
            # Store results
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(true_labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # 1. Confusion Matrix
    create_confusion_matrix(all_labels, all_preds, save_dir)
    
    # 2. Classification Report
    create_classification_report(all_labels, all_preds, save_dir)
    
    # 3. Error Analysis
    perform_error_analysis(all_labels, all_preds, all_probs, save_dir)
    
    # 4. Example Predictions
    analyze_example_predictions(all_labels, all_preds, all_probs, save_dir)
    
    print(f"[Info] All analysis results saved to {save_dir}")

def create_confusion_matrix(y_true, y_pred, save_dir):
    """
    Create and save confusion matrix visualization
    """
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred)
    
    if num_labels == 2:
        labels = ["Real", "Fake"]
    else:
        labels = ["true", "mostly-true", "half-true", "barely-true", "false", "pants-fire"]
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"), dpi=300)
    plt.close()
    
    # Calculate and print metrics
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    print(f"Overall Accuracy: {accuracy:.4f}")
    
    # Normalized confusion matrix (percentages)
    plt.figure(figsize=(10, 8))
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=labels)
    disp.plot(cmap="Blues", values_format=".2f")
    plt.title("Normalized Confusion Matrix (Row Percentages)")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrix_normalized.png"), dpi=300)
    plt.close()
    
    print(f"[Info] Confusion matrices saved to {save_dir}")

def create_classification_report(y_true, y_pred, save_dir):
    """
    Generate and save classification report
    """
    if num_labels == 2:
        target_names = ["Real", "Fake"]
    else:
        target_names = ["true", "mostly-true", "half-true", "barely-true", "false", "pants-fire"]
    
    report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    
    # Save as text file
    with open(os.path.join(save_dir, "classification_report.txt"), "w") as f:
        f.write(classification_report(y_true, y_pred, target_names=target_names))
    
    # Create visualization of the report
    plt.figure(figsize=(10, 6))
    
    # Extract precision, recall, and f1-score
    classes = list(report.keys())[:-3]  # Exclude 'accuracy', 'macro avg', 'weighted avg'
    precision = [report[cls]['precision'] for cls in classes]
    recall = [report[cls]['recall'] for cls in classes]
    f1 = [report[cls]['f1-score'] for cls in classes]
    
    x = np.arange(len(classes))
    width = 0.25
    
    plt.bar(x - width, precision, width, label='Precision')
    plt.bar(x, recall, width, label='Recall')
    plt.bar(x + width, f1, width, label='F1-score')
    
    plt.xlabel('Classes')
    plt.ylabel('Scores')
    plt.title('Classification Performance by Class')
    plt.xticks(x, classes, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "classification_performance.png"), dpi=300)
    plt.close()
    
    print(f"[Info] Classification report saved to {save_dir}")

def perform_error_analysis(y_true, y_pred, y_probs, save_dir):
    """
    Perform error analysis to understand model mistakes
    """
    # Find misclassified examples
    errors = y_true != y_pred
    error_indices = np.where(errors)[0]
    
    # Count errors by true and predicted class
    error_counts = {}
    for i in error_indices:
        true_label = int(y_true[i])
        pred_label = int(y_pred[i])
        key = (true_label, pred_label)
        if key in error_counts:
            error_counts[key] += 1
        else:
            error_counts[key] = 1
    
    # Sort error types by frequency
    sorted_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Create error analysis report
    with open(os.path.join(save_dir, "error_analysis.txt"), "w") as f:
        f.write("ERROR ANALYSIS REPORT\n")
        f.write("====================\n\n")
        f.write(f"Total examples: {len(y_true)}\n")
        f.write(f"Correctly classified: {len(y_true) - len(error_indices)}\n")
        f.write(f"Misclassified: {len(error_indices)}\n\n")
        f.write("Most common error types:\n")
        
        if num_labels == 2:
            class_names = ["Real", "Fake"]
        else:
            class_names = ["true", "mostly-true", "half-true", "barely-true", "false", "pants-fire"]
        
        for (true_label, pred_label), count in sorted_errors:
            f.write(f"  {class_names[true_label]} classified as {class_names[pred_label]}: {count} instances\n")
    
    # Create visualization of error distribution
    if len(sorted_errors) > 0:
        plt.figure(figsize=(12, 8))
        error_types = [f"{class_names[true]}->{class_names[pred]}" for (true, pred), _ in sorted_errors]
        error_counts = [count for _, count in sorted_errors]
        
        plt.bar(error_types, error_counts)
        plt.xlabel('Error Type (True->Predicted)')
        plt.ylabel('Count')
        plt.title('Distribution of Error Types')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "error_distribution.png"), dpi=300)
        plt.close()
    
    print(f"[Info] Error analysis saved to {save_dir}")

def analyze_example_predictions(y_true, y_pred, y_probs, save_dir):
    """
    Analyze and save example predictions (correct and incorrect)
    """
    # Get statements and justifications from test set
    test_statements = [X_test[i] for i in range(len(X_test))]
    test_justifications = [X_test_just[i] for i in range(len(X_test_just))]
    
    # Find correct and incorrect predictions
    correct = y_true == y_pred
    correct_indices = np.where(correct)[0]
    error_indices = np.where(~correct)[0]
    
    # Get confidence scores (probability of predicted class)
    confidences = np.max(y_probs, axis=1)
    
    # Find high confidence correct predictions
    if len(correct_indices) > 0:
        correct_confidences = confidences[correct_indices]
        high_conf_correct_idx = correct_indices[np.argsort(correct_confidences)[-10:]]  # Top 10
    else:
        high_conf_correct_idx = []
    
    # Find high confidence incorrect predictions
    if len(error_indices) > 0:
        error_confidences = confidences[error_indices]
        high_conf_error_idx = error_indices[np.argsort(error_confidences)[-10:]]  # Top 10
    else:
        high_conf_error_idx = []
    
    # Create example predictions report
    with open(os.path.join(save_dir, "example_predictions.txt"), "w") as f:
        if num_labels == 2:
            class_names = ["Real", "Fake"]
        else:
            class_names = ["true", "mostly-true", "half-true", "barely-true", "false", "pants-fire"]
        
        f.write("EXAMPLE PREDICTIONS\n")
        f.write("===================\n\n")
        
        # High confidence correct predictions
        f.write("HIGH CONFIDENCE CORRECT PREDICTIONS\n")
        f.write("-----------------------------------\n\n")
        for i, idx in enumerate(high_conf_correct_idx):
            f.write(f"Example {i+1}:\n")
            f.write(f"Statement: {test_statements[idx]}\n")
            f.write(f"Justification: {test_justifications[idx]}\n")
            f.write(f"True label: {class_names[y_true[idx]]}\n")
            f.write(f"Predicted label: {class_names[y_pred[idx]]}\n")
            f.write(f"Confidence: {confidences[idx]:.4f}\n\n")
        
        # High confidence incorrect predictions
        f.write("\nHIGH CONFIDENCE INCORRECT PREDICTIONS\n")
        f.write("-------------------------------------\n\n")
        for i, idx in enumerate(high_conf_error_idx):
            f.write(f"Example {i+1}:\n")
            f.write(f"Statement: {test_statements[idx]}\n")
            f.write(f"Justification: {test_justifications[idx]}\n")
            f.write(f"True label: {class_names[y_true[idx]]}\n")
            f.write(f"Predicted label: {class_names[y_pred[idx]]}\n")
            f.write(f"Confidence: {confidences[idx]:.4f}\n\n")
    
    print(f"[Info] Example predictions saved to {save_dir}")

# Run the evaluation and visualization
print("\n\n===== Starting Model Analysis and Visualization =====\n")
evaluate_model_with_visualizations(model_ft1, dataloaders_dict['val'], save_dir="model_analysis")
