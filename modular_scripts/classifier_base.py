from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
import torch.nn.init as init
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
from torch import nn
from tqdm.auto import tqdm


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        acc = accuracy(out, labels)  # Calculate accuracy
        return {"val_loss": loss.detach(), "val_acc": acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x["val_loss"] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        batch_accs = [x["val_acc"] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
        return {"val_loss": epoch_loss.item(), "val_acc": epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print(
            "Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
                epoch, result["train_loss"], result["val_loss"], result["val_acc"]
            )
        )


def evaluate_classification_model(model, dataloader, device):
    predictions = []
    model.eval()
    with torch.inference_mode():  # Disable gradient calculation
        for batch, (X, y) in enumerate(dataloader):
            inputs = X.to(device)  # Send data to the device (CPU or GPU)
            outputs = model(inputs)  # Forward pass to get predictions
            predictions.append(outputs)
    true_labels = []  # Initialize an empty list to store true labels
    predictions = torch.cat(predictions, dim=0)
    probabilities = torch.softmax(predictions, dim=1)
    _, predicted_labels = torch.max(probabilities, dim=1)
    predicted_labels = predicted_labels.to("cpu")
    for batch, (X, y) in enumerate(dataloader):
        for labels in y:  # Iterate through the dataset
            true_labels.append(labels)

    precision = precision_score(true_labels, predicted_labels, average="weighted")
    recall = recall_score(true_labels, predicted_labels, average="weighted")
    f1 = f1_score(true_labels, predicted_labels, average="weighted")
    accuracy = accuracy_score(true_labels, predicted_labels)
    mcc = matthews_corrcoef(true_labels, predicted_labels)
    return {
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "MCC": mcc,
        "Accuracy": accuracy,
    }


def plot_confusion_matrix(model, data, class_names, device, save=False, name=None):
    predictions = []
    model.eval()
    with torch.inference_mode():  # Disable gradient calculation
        for batch, (X, y) in enumerate(data):
            inputs = X.to(device)  # Send data to the device (CPU or GPU)
            outputs = model(inputs)  # Forward pass to get predictions
            predictions.append(outputs)
    true_labels = []  # Initialize an empty list to store true labels

    for batch, (X, y) in enumerate(data):
        for labels in y:  # Iterate through the dataset
            true_labels.append(labels)
    # Concatenate predictions for all batches
    predictions = torch.cat(predictions, dim=0)
    probabilities = torch.softmax(predictions, dim=1)
    _, predicted_labels = torch.max(probabilities, dim=1)
    predicted_labels = predicted_labels.to("cpu")
    cm = confusion_matrix(true_labels, predicted_labels)
    # Plot the confusion matrix as a heatmap
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.2)  # Adjust the font size for better visualization
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()
    if save:
        plt.savefig("./cm+" + name + ".png")


def plot_loss_curves(results, name, save=False):
    loss = results["train_loss"]
    test_loss = results["test_loss"]

    accuracy = results["train_acc"]
    test_accuracy = results["test_acc"]

    epochs = range(len(results["train_loss"]))

    # Plot loss
    plt.plot(epochs, loss, label="training_loss")
    plt.plot(epochs, test_loss, label="val_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()
    plt.savefig("./loss_" + name + ".png")
    # Plot accuracy
    plt.figure()
    plt.plot(epochs, accuracy, label="training_accuracy")
    plt.plot(epochs, test_accuracy, label="val_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()
    if save:
        plt.savefig("./accuracy_loss_" + name + ".png")


def plot_roc_curve(model, test_dataloader, name, device, save=False):
    y_score = []
    model.eval()
    with torch.inference_mode():  # Disable gradient calculation
        for batch, (X, y) in enumerate(test_dataloader):
            inputs = X.to(device)  # Send data to the device (CPU or GPU)
            outputs = model(inputs)  # Forward pass to get y_scores
            y_score.append(outputs)

    true_labels = []
    for batch, (X, y) in enumerate(test_dataloader):
        for labels in y:  # Iterate through the dataset
            true_labels.append(labels)
    # Concatenate y_scores for all batches
    y_score = torch.cat(y_score, dim=0)
    # Binarize the true labels (one-hot encoding)
    n_classes = 6
    y_test_bin = label_binarize(true_labels, classes=np.arange(n_classes))
    y_score = y_score.to("cpu")
    # Initialize variables for ROC curve
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # Compute ROC curve and ROC AUC for each class
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot ROC curves for each class
    plt.figure(figsize=(10, 8))
    colors = cycle(
        ["blue", "red", "green", "purple", "orange"]
    )  # You can add more colors if needed
    for i, color in zip(range(n_classes), colors):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=2,
            label=f"Class {i} (AUC = {roc_auc[i]:.2f})",
        )

    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve for Multi-class")
    plt.legend(loc="lower right")
    plt.show()
    if save:
        plt.savefig("./roc_curve_" + name + ".png")
