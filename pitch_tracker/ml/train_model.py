from typing import Tuple

import numpy as np
import torch

from torch.optim.lr_scheduler import ReduceLROnPlateau
from pitch_tracker.ml.earlystopping import EarlyStopping
from pitch_tracker.ml.measures import melody_evaluate

# Manually set seed for reproducibility
SEED = 1
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def train(
        model,
        dataloader,
        loss_fn,
        optimizer,
        device: str) -> Tuple[float, float]:
    """Trains a model on a given dataset for one epoch.

    Args:
        model: The model to be trained.
        dataloader: The dataloader for the training dataset.
        loss_fn: The loss function to be used during training.
        optimizer: The optimizer to be used during training.
        device (str): The device on which to run the training.

    Returns:
         Tuple[float, float]: A tuple containing the average loss and average accuracy of the model on the training dataset for one epoch.
    """
    total_batches = len(dataloader)
    total_target_size = 0
    running_loss = 0
    total_correct = 0

    model.train()
    for batch, (X, (y1, y2, y3)) in enumerate(dataloader):
        X, y3 = X.to(device), y3.to(device)

        # Compute prediction error
        y_pred = model(X)
        loss = loss_fn(y_pred, y3)

        # output definition:
        # 89 classes:
        # - 1st class: non melody
        # - 2nd->89th classes: pitch from A0->C8 (inclusive)
        
        pos_neg_arr = (y_pred.argmax(2) == y3.argmax(2)).flatten()
        batch_target_size = pos_neg_arr.numel()
        batch_correct = torch.nonzero(pos_neg_arr).numel()
        batch_accuracy = batch_correct/batch_target_size
        
        melody_score = melody_evaluate(y_true=y3, y_pred=y_pred)
        
        total_target_size += batch_target_size
        total_correct += batch_correct

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch % 50 == 0:
            print((f"[{batch+1:>5d}/{total_batches:>5d}]\t"
                  f"Batch Accuracy: {(100*batch_accuracy):>0.1f}%, "
                  f"current loss: {running_loss/(batch+1):>7f}, "
                  f"Voicing Recall: {melody_score['Voicing Recall']:>.4f}, "
                  f"Voicing False Alarm: {melody_score['Voicing False Alarm']:>.4f}, "
                  f"Raw Pitch Accuracy: {melody_score['Raw Pitch Accuracy']:>.4f}, "
                  f"Raw Chroma Accuracy: {melody_score['Raw Chroma Accuracy']:>.4f}, "
                  f"Overall Accuracy: {melody_score['Overall Accuracy']:>.4f}"
                  ))

    avg_loss = running_loss / total_batches
    avg_accuracy = total_correct / total_target_size
    print(f"[{batch+1:>5d}/{total_batches:>5d}]  Avg Accuracy: {(100*avg_accuracy):>0.1f}%, Avg loss: {avg_loss:>7f}")
    return avg_loss, avg_accuracy


def test(
        model,
        dataloader,
        loss_fn,
        device: str) -> Tuple[float, float]:
    """Tests a model on a given dataset.

    Args:
        model: The model to be tested.
        dataloader: The dataloader for the test dataset.
        loss_fn: The loss function to be used during testing.
        device (str): The device on which to run the test.

    Returns:
        Tuple[float, float]: A tuple containing the average loss and average accuracy of the model on the test dataset.
    """
    num_batches = len(dataloader)
    n_size = 0
    running_loss = 0
    n_correct = 0
    model.eval()
    y_trues = []
    y_preds = []
    with torch.no_grad():
        for X, (y1, y2, y3) in dataloader:
            X, y3 = X.to(device), y3.to(device)
            y_pred:torch.Tensor = model(X)
            running_loss += loss_fn(y_pred, y3).item()


            y3_to_label = y3.argmax(2)
            y_pred_to_label = y_pred.argmax(2)

            y_trues.append(y3_to_label)
            y_preds.append(y_pred_to_label)

            pos_neg_arr = (y_pred_to_label == y3_to_label).flatten()
            n_size += pos_neg_arr.numel()
            n_correct += torch.nonzero(pos_neg_arr).numel()

    y_trues = torch.cat(y_trues,dim=0)
    y_preds = torch.cat(y_preds,dim=0)

    melody_score = melody_evaluate(y_true=y_trues, y_pred=y_preds)
    
    avg_loss = running_loss / num_batches
    avg_accuracy = n_correct / n_size
    print((f"Test Error: \n Accuracy: {(100*avg_accuracy):>0.1f}%, Avg loss: {avg_loss:>8f} \n"
            f"Voicing Recall: {melody_score['Voicing Recall']:>.4f}, "
            f"Voicing False Alarm: {melody_score['Voicing False Alarm']:>.4f}, "
            f"Raw Pitch Accuracy: {melody_score['Raw Pitch Accuracy']:>.4f}, "
            f"Raw Chroma Accuracy: {melody_score['Raw Chroma Accuracy']:>.4f}, "
            f"Overall Accuracy: {melody_score['Overall Accuracy']:>.4f}"))
    
    return avg_loss, avg_accuracy


def train_model(
        model,
        train_dataloader,
        validation_dataloader,
        loss_fn,
        optimizer,
        n_epochs: int,
        early_stopping: EarlyStopping = None,
        lr_scheduler=None,
        device: str = 'cpu',
        trace_func=print):
    """Trains a model on a given dataset.

    Args:
        model: The model to be trained.
        train_dataloader: The dataloader for the training dataset.
        validation_dataloader: The dataloader for the validation dataset.
        loss_fn: The loss function to be used during training.
        optimizer: The optimizer to be used during training.
        n_epochs (int): The number of epochs to train for.
        early_stopping (EarlyStopping, optional): An instance of EarlyStopping. Defaults to None.
        lr_scheduler (optional): A learning rate scheduler. Defaults to None.
        device (str, optional): The device on which to run the training. Defaults to 'cpu'.
        trace_func (optional): A function used for logging. Defaults to print.

    Returns:
         Tuple[Model, List[float], List[float]]: A tuple containing the latest trained model and lists of average training and validation losses per epoch.
    """
    # logging history
    avg_train_losses = []
    avg_train_accuracies = []
    avg_validation_losses = []
    avg_validation_accuracies = []

    for epoch in range(1, n_epochs + 1):
        trace_func(f"Epoch {epoch}\n-------------------------------")

        avg_train_loss, avg_train_accuracy = train(
            model, train_dataloader, loss_fn, optimizer, device)

        avg_validation_loss, avg_validation_accuracy = test(
            model, validation_dataloader, loss_fn, device)

        # save avg loss & accuracy
        avg_train_losses.append(avg_train_loss)
        avg_train_accuracies.append(avg_train_accuracy)
        avg_validation_losses.append(avg_validation_loss)
        avg_validation_accuracies.append(avg_validation_accuracy)

        # early_stopping needs the validation loss to check if it has decresed,
        # and if it has, it will make a checkpoint of the current model
        if early_stopping:
            early_stopping(avg_validation_loss, model)
            if early_stopping.early_stop:
                trace_func("Early stopping")
                break

        if lr_scheduler:
            lr_scheduler.step(avg_validation_loss)

    return model, avg_train_losses, avg_validation_losses
