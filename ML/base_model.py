from create_training_examples import TrainingExample
import numpy as np
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import pickle
import time


class TransformerLanguageModel(nn.Module):
    def __init__(self, d_model=4, nhead=8, n_layers=3):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # 2 output dim, 1 is goes up, 0 it goes down
        self.W = nn.Linear(d_model, 2)
        self.log_softmax=nn.LogSoftmax(dim=1)

        nn.init.xavier_uniform_(self.W.weight)


    def forward(self, training_examples):
        seq_len = len(training_examples)
        
        mask = (torch.triu(torch.ones(seq_len, seq_len)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

        outputs = self.transformer_encoder(src=training_examples, mask=mask)

        outputs = self.W(outputs)
        outputs = self.log_softmax(outputs)
        
        return outputs
    


def train_stock_model():

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(torch.cuda.is_available())

    TRAINING_DAY_LENGTH = 10
    d_model = 4
    nhead = 1
    n_layers = 6
    lr = 1e-5

    model = TransformerLanguageModel(d_model=d_model, nhead=nhead, n_layers=n_layers)
    model.zero_grad()
    model.train()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.999)

    # class_weights = torch.tensor([1.2, 1.0])  # Class weights for three classes
    loss_fcn = nn.NLLLoss()

    epoch = 100

    start_time = time.time()
    print(f"\ne:{epoch} lr:{lr} d_model:{d_model} nhead:{nhead} n_layers:{n_layers}")

    with open('training_examples/NVDA_training_example.pkl', 'rb') as f:
        training_examples = pickle.load(f)

    training_examples_values = []
    for example in training_examples:
        values = [example.price]
        values.extend(example.sentiment_scores)
        training_examples_values.append(values)

    # training_examples_values = training_examples_values[:500]

    print(len(training_examples))
    
    validation_length = 100
    validation = torch.tensor(training_examples_values[-validation_length:])
    training_examples_tensor = torch.tensor(training_examples_values[:-validation_length])
    
    skipped_training = []
    
    for e in range(epoch):
        total_loss = 0
        total_down = 0
        skipped_populated = False
        # Need to subtract by 1 more so we have a gold label at the end of training examples
        for i in range(0, len(training_examples_tensor) - TRAINING_DAY_LENGTH - 1, TRAINING_DAY_LENGTH):
            if len(skipped_training) > 0 and i in skipped_training:
                skipped_populated = True
                continue

            up = 0
            down = 0

            example_tensor = training_examples_tensor[i:i+TRAINING_DAY_LENGTH]

            # This is the reason we need to subtract by 1
            gold_examples = []
            for gold_example in training_examples_tensor[i + 1:i+len(example_tensor) + 1]:
                if gold_example[0] >= 0:
                    gold_examples.append(1)
                    up += 1
                else:
                    gold_examples.append(0)
                    down+=1

            # Used to balance the classes
            if total_down < 148 and up >= down + 4 and i != 0 and not skipped_populated:
                total_down += up - down
                skipped_training.append(i)
                continue

            gold_example_tensor = torch.tensor(gold_examples)
            
            log_probs = model.forward(example_tensor)

            if i == 0:
                print("gold", gold_example_tensor.tolist())
                print("pred", torch.argmax(log_probs, dim=1).tolist())
                print()

            loss = loss_fcn(log_probs, gold_example_tensor)

            model.zero_grad()
            loss.backward()
            optimizer.step()

            # scheduler.step()

            total_loss += loss
        # new_lr = optimizer.param_groups[0]['lr']
        new_lr = lr
        print(f'epoch:{e} total_loss:{total_loss} lr:{new_lr} time:{time.time() - start_time:.2f}\n')

    print(f"\ne:{epoch} lr:{lr} nhead:{nhead} n_layers:{n_layers} total_time:{time.time() - start_time:.2f}")


    # Evaluate the Model
    model.eval()
    print("Evaluating model on training data\n")

    correct = 0
    total = 0
    for i in range(0, len(training_examples_tensor), TRAINING_DAY_LENGTH):
        if i in skipped_training:
            continue
        example_tensor = training_examples_tensor[i:i+TRAINING_DAY_LENGTH]

        gold_examples = []
        for gold_example in training_examples_tensor[i+1:i+TRAINING_DAY_LENGTH+1]:
            if gold_example[0] >= 0:
                gold_examples.append(1)
            else:
                gold_examples.append(0)

        log_probs = model.forward(example_tensor)
        predicted_label = torch.argmax(log_probs, dim=1)

        for label, gold_label in zip(predicted_label, gold_examples):
            total += 1
            if label == gold_label:
                correct += 1
        if i == 0:
            print("gold", gold_examples)
            print("pred", predicted_label.tolist())
            print()

    print(f"Training Accuracy: {correct}/{total} = {correct / total}\n")


    print("Evaluating model on validation data\n")

    correct = 0
    total = 0
    for i in range(0, len(validation), TRAINING_DAY_LENGTH):
        example_tensor = validation[i:i+TRAINING_DAY_LENGTH]

        gold_examples = []
        for gold_example in validation[i+1:i+TRAINING_DAY_LENGTH+1]:
            if gold_example[0] >= 0:
                gold_examples.append(1)
            else:
                gold_examples.append(0)

        log_probs = model.forward(example_tensor)
        predicted_label = torch.argmax(log_probs, dim=1)

        for label, gold_label in zip(predicted_label, gold_examples):
            total += 1
            if label == gold_label:
                correct += 1

        print("gold", gold_examples)
        print("pred", predicted_label.tolist())
        print()

    print(f"Validation Accuracy: {correct}/{total} = {correct / total} ")


if __name__== "__main__":
    train_stock_model()