from genericpath import exists
import torch
import itertools
import torchmetrics
from model import MediaBiasNN
from torch.optim import Adam
from matplotlib import pyplot as plt

def move_to_cuda(data):
    data = [x.cuda() for x in data]
    return data

def train_one_epoch(model, dataloader, optimizer):
    '''
    Train one epochs.
    Args:
        model:
            trainable model
        dataloader:
            dataloader for training/train_loader
        optimizer:
            optimizer with model parameters loaded in.
    Return:
        tensor: mean loss value in one epoch with train loader
    '''
    model.train()
    loss_list = []
    for data, target in dataloader:
        data = move_to_cuda(data)
        target = target.cuda()
        loss = model(data, target)
        loss_list.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return torch.mean(torch.tensor(loss_list))

@torch.no_grad()
def validation_one_epoch(model, dataloader):
    '''
    Validatae one epochs.
    Args:
        model:
            trainable model
        dataloader:
            dataloader for training/val_loader
        optimizer:
            optimizer with model parameters loaded in.
    Return:
        tensor: mean loss value in one epoch with validation loader
    '''
    model.train()
    loss_list = []
    for data, target in dataloader:
        data = move_to_cuda(data)
        target = target.cuda()
        loss = model(data, target)
        loss_list.append(loss.item())
    return torch.mean(torch.tensor(loss_list))

@torch.no_grad()
def evaluate(model, dataloader):
    '''
    Evaluate model performance using the dataloader.
    Args:
        model:
            trainable model
        dataloader:
            dataloader for training/val_loader
    Return:
        float, float: accuracy over test set, f1-score over test set
    '''
    acc_metrics = torchmetrics.Accuracy()
    f1_metrics = torchmetrics.F1Score(num_classes=3, average='macro')
    model.eval()
    for data, target in dataloader:
        data = move_to_cuda(data)
        pred = model(data)
        pred = pred.softmax(dim=-1)
        acc = acc_metrics(pred.cpu(), target.cpu())
        f1 = f1_metrics(pred.cpu(), target.cpu())
        #print(f"acc {acc}")
        #print(f"f1 {f1}")
    return acc_metrics.compute(), f1_metrics.compute()

def plot_loss_over_epochs(loss):
    '''
    Plot loss over epoch, helper function.
    Args:
        loss:
            list of loss over all epochs/
    '''
    plt.plot(loss)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()

def hyper_parameters_search(
        model_type,
        hyperparameters,
        validation_dataloader,
        validation_epoch=50):
    '''
    Hyperparameter seach function. Select hyperparameters base on last epoch loss.
    Args:
        model_type:
            "rnn" or "baseline"
            Tell model which type of content embedding to use.
        hyperparameters:
            list of hyperparameters to seach from. need to have format 
            (learning weight, weight_decay, hidden_dim for linear layer)
        validation_dataloader:
            dataloader for validation dataset.
        validation_epoch:
            number of epoch to search each set of hyperparamters from.
            default: 50
    Return:
        float, float, list: best set of hyperparameters 
                            (lr, weight_decay, hidden_dim)
    '''
    param_loss = []
    for lr, weight_decay, hidden_dim in hyperparameters:
        model = MediaBiasNN(hidden_dim, model_type=model_type)
        model.cuda()
        model_params = [param for param in model.parameters() if param.requires_grad]
        optimizer = Adam(model_params, lr=lr, weight_decay=weight_decay)
        loss = []
        for epoch in range(validation_epoch):
            epoch_loss = train_one_epoch(model, validation_dataloader, optimizer)
            print(f"epoch {epoch}", f"loss: {epoch_loss}")
            loss.append(epoch_loss)
        plot_loss_over_epochs(loss)
        param_loss.append(((lr, weight_decay, hidden_dim), loss[-1]))
    param_loss.sort(key=lambda x: x[1])
    best_lr, best_weight_decay, best_hidden_dim = param_loss[0][0]
    return best_lr, best_weight_decay, best_hidden_dim

def train_model(model,
        train_loader,
        val_loader,
        test_loader,
        optimizer,
        training_epoch,
        save_path):
    '''
    Training function. 
    Args:
        model:
            trainable model.
        val_loader:
            dataloader for validation dataset. keep track of validation loss.
        test_loader:
            dataloader for test dataset. keep track of performance.
        training_epoch:
            number of epoch to train the model.
        save_path:
            pathway to save the model state_dict.
    '''
    model.train()
    model.cuda()
    train_loss = []
    val_loss = []
    save_path.mkdir(exist_ok=True)
    for epoch in range(training_epoch):
        epoch_train_loss = train_one_epoch(model, train_loader, optimizer)
        # Saving model every epochs
        torch.save(model.state_dict(), str(save_path/f'epoch-{epoch}.pth'))
        train_loss.append(epoch_train_loss)
        epoch_val_loss = validation_one_epoch(model, val_loader)
        val_loss.append(epoch_val_loss)
        acc, f1 = evaluate(model, test_loader)
        print(
            f"epoch {epoch}",
            f"train loss: {epoch_train_loss}",
            f"val loss: {epoch_val_loss}",
            f"test acc: {acc}",
            f"test f1: {f1}")
    plot_loss_over_epochs(train_loss)
    plot_loss_over_epochs(val_loss)











