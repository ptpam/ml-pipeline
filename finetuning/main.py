"""
Finetuning Torchvision Models
=============================
Base code belongs to the author `Nathan Inkawhich <https://github.com/inkawhich>`__
"""
import sys

sys.path.append("..")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
from model import initialize_model
from dataset import load_data
from engine import train_model

if __name__ == "__main__":
    model_name_list = [
        "resnet",
        "alexnet",
        "vgg",
        "squeezenet",
        "densenet",
        "inception",
    ]
    num_classes = 10
    feature_extract = True
    batch_size = 4
    num_epochs = 10

    for model_name in model_name_list:
        # Initialize the model for this run
        model_ft, input_size = initialize_model(
            model_name, num_classes, feature_extract, use_pretrained=True
        )

        # Print the model we just instantiated
        print(model_ft)

        trainset, testset = load_data(input_size=input_size)

        test_abs = int(len(trainset) * 0.8)
        train_subset, val_subset = random_split(
            trainset, [test_abs, len(trainset) - test_abs]
        )

        trainloader = torch.utils.data.DataLoader(
            train_subset, batch_size=batch_size, shuffle=True, num_workers=8
        )
        valloader = torch.utils.data.DataLoader(
            val_subset, batch_size=batch_size, shuffle=True, num_workers=8
        )
        dataloaders_dict = {"train": trainloader, "val": valloader}

        # Detect if we have a GPU available
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Send the model to GPU
        model_ft = model_ft.to(device)

        # Gather the parameters to be optimized/updated in this run. If we are
        #  finetuning we will be updating all parameters. However, if we are
        #  doing feature extract method, we will only update the parameters
        #  that we have just initialized, i.e. the parameters with requires_grad
        #  is True.
        params_to_update = model_ft.parameters()
        print("Params to learn:")
        if feature_extract:
            params_to_update = []
            for name, param in model_ft.named_parameters():
                if param.requires_grad == True:
                    params_to_update.append(param)
                    print("\t", name)
        else:
            for name, param in model_ft.named_parameters():
                if param.requires_grad == True:
                    print("\t", name)

        # Observe that all parameters are being optimized
        optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

        # Setup the loss fxn
        criterion = nn.CrossEntropyLoss()

        # Train and evaluate
        model_ft, hist = train_model(
            model_ft,
            dataloaders_dict,
            criterion,
            optimizer_ft,
            device,
            num_epochs=num_epochs,
            is_inception=(model_name == "inception"),
        )
