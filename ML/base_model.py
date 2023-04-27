from create_training_examples import TrainingExample
import numpy as np
import torch
import torch.nn as nn
import pickle
import time
import math

from torch.utils.data import Dataset, DataLoader, random_split

# Tensorboard
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
writer = SummaryWriter(log_dir=f'runs/{datetime.now().strftime("%H-%M-%S")}')



class StockDataset(Dataset):
    def __init__(self, training_examples, days_attention, days_stride) -> None:
        self.days_attention = days_attention # Length of each sample - i.e. maximum number of days model can attend to
        self.days_stride = days_stride
        self.n_samples = math.floor((len(training_examples) - days_attention) / days_stride)

        # Input feature size if [n_days, 4]
        #   input_feature[i] = [close_price, sentiment_positive, sentiment_neutral, sentiment_negative]
        self.x = torch.from_numpy(np.zeros((self.n_samples, self.days_attention, 4), dtype=np.float64))
        self.y = torch.from_numpy(np.zeros((self.n_samples, self.days_attention), dtype=np.float64))

        training_examples.append(training_examples[len(training_examples) -1]) # Duplicate last entry so that we don't have index out of bounds when computing gold label
        for i in range(self.n_samples):
            idx = i * days_stride
            for j in range(days_attention):
                self.x[i, j, 0] = training_examples[idx + j].price
                self.x[i, j, 1] = training_examples[idx + j].sentiment_scores[0]
                self.x[i, j, 2] = training_examples[idx + j].sentiment_scores[1]
                self.x[i, j, 3] = training_examples[idx + j].sentiment_scores[2]
                # self.x[idx, j, [1, 2, 3]] = training_examples[idx + j].sentiment_scores

                if training_examples[idx + j + 1].price >= 0:
                    self.y[i, j] = 1  # Stock increases at next timestep
                else:
                    self.y[i, j] = 0  # Stock decreases at next timestep
                
    def __getitem__(self, index):   # return shape: x -> [num_days, 4]      y -> [num_days]
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples




class TransformerLanguageModel(nn.Module):
    def __init__(self, device, d_model=4, nhead=8, n_layers=3):
        super().__init__()

        self.device = device

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead).to(dtype=torch.float64)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # 2 output dim, 1 is goes up, 0 it goes down
        self.W = nn.Linear(d_model, 2).to(dtype=torch.float64)
        self.log_softmax=nn.LogSoftmax(dim=1).to(dtype=torch.float64)

        nn.init.xavier_uniform_(self.W.weight)


    def forward(self, training_examples):
        seq_len = len(training_examples)
        
        mask = (torch.triu(torch.ones(seq_len, seq_len)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        mask = mask.to(dtype=torch.float64).to(self.device)

        outputs = self.transformer_encoder(src=training_examples, mask=mask)

        outputs = self.W(outputs)
        outputs = self.log_softmax(outputs)
        
        return outputs


def evaluate_model(model, dataset, device):
    eval_correct = 0
    eval_total = 0

    log_interval = 1 if len(dataset) < 10 else 5
    for i, (data, actual) in enumerate(dataset):
        data = data.to(device)
        actual = actual.to(device)
    
        # forward pass and loss
        log_probs = model.forward(data)

        predicted_label = torch.argmax(log_probs, dim=1)
        for label, gold_label in zip(predicted_label, actual):
            eval_total += 1
            if label == gold_label:
                eval_correct += 1

        model.zero_grad()
        
        if i % log_interval == 0:
            print(f'\n[{i}] pred:\t {predicted_label.cpu().numpy()}')
            print(f'[{i}] gold:\t {actual.to(dtype=torch.int64).cpu().numpy()}')

    print(f"\nEvaluation Accuracy: {eval_correct}/{eval_total} = {eval_correct / eval_total}\n")


def train_stock_model():
    # device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f'Loading model')

    d_model = 4
    nhead = 2
    n_layers = 2
    lr = 1e-4

    model = TransformerLanguageModel(device=device, d_model=d_model, nhead=nhead, n_layers=n_layers)
    model.zero_grad()
    model.train()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fcn = nn.NLLLoss()


    # print(f"\nTotal epochs:{epochs} lr:{lr} d_model:{d_model} nhead:{nhead} n_layers:{n_layers}")

    print(f'Loading data')

    with open('training_examples/AMD_training_example.pkl', 'rb') as f:
        training_examples = pickle.load(f) # A list of TrainingExample

    TRAINING_DAY_LENGTH = 10
    stock_dataset = StockDataset(
        training_examples,
        days_attention=TRAINING_DAY_LENGTH,
        days_stride=3 # Determines how much day windows will overlap
        # if days_stride == days_attention -> there is no overlap
    )

    train_length = math.floor(len(stock_dataset) * 0.9)
    stocks_train, stocks_test = random_split(stock_dataset, [train_length, len(stock_dataset) - train_length])

    train_data_loader = DataLoader(
        dataset=stocks_train,
        # batch_size= 20 * max(1, torch.cuda.device_count()),
        batch_size=10,
        shuffle=True)
    

    print(f'Beginning training')

    num_epochs = 400
    n_total_steps = len(train_data_loader)
    log_interval = 1
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (data, actual) in enumerate(iter(train_data_loader)):
            data = data.to(device)
            actual = actual.to(device)
        
            # forward pass and loss
            log_probs = model.forward(data)

            # actual and predicted have to be flattened for NLLLoss because it doesn't support batching
            # NLLLoss isn't implemented for torch.float64 so we convert to torch.LongTensor
            loss = loss_fcn(log_probs.view(-1, log_probs.shape[2]), actual.view(-1).type(torch.LongTensor).to(device))

            # backward pass
            loss.backward()

            # update
            optimizer.step()
            # scheduler.step()
            
            optimizer.zero_grad()
            model.zero_grad()

            running_loss += loss.item()
    
            if (i + 1) % log_interval == 0:
                print(f'actual: \t{actual[0].cpu().numpy()}')
                print(f'predicted: \t{[torch.argmax(log_probs[0, i]).item() for i in range(log_probs.shape[1])]}')
                print(f'epoch {epoch + 1} / {num_epochs}, step {i + 1}/{n_total_steps}, loss = {loss.item():.4f}')
                # writer.add_scalar('training loss', running_loss / log_interval, (epoch * n_total_steps) + i)
                # writer.add_scalar('training accuracy', running_correct / (actual.shape[1] * log_interval), (epoch * n_total_steps) + i)
                running_loss = 0.0
                # running_correct = 0

    # Evaluate the Model
    model.eval()
    
    print("\nEvaluating model on training data\n")
    evaluate_model(
        model=model,
        dataset=stocks_train,
        device=device
    )

    print("\nEvaluating model on test data\n")
    evaluate_model(
        model=model,
        dataset=stocks_test,
        device=device
    )
    


if __name__== "__main__":
    train_stock_model()