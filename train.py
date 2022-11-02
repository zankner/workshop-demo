import argparse

from models.network import LeNet
from utils.data import get_dataloaders

import torch


def main(args):
    
    # Init model
    model = LeNet(64, 32)

    # Load data
    train_loader, val_loader = get_dataloaders(args.batch_size)

    # Define optimizers
    opt = torch.optim.SGD(model.parameters(), lr=args.lr)

    # Define loss function
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        opt.zero_grad()
        for batch_idx, (input_, output) in enumerate(train_loader):
            # Clear opt state
            # print(torch.all(input_))

            # Call the model
            pred = model(input_)
            
            # Compute the loss of model
            loss = loss_fn(pred, output)

            # Update model based on loss
            loss.backward()
            opt.step()

            if (batch_idx + 1) % 20 == 0:
                print(f"Train loss: {loss.item():.2f}")

        val_loss = 0
        examples_seen = 0
        for input_, output in val_loader:
            pred = model(input_)
            loss = loss_fn(pred,output)
            val_loss += loss.item()
            examples_seen += input_.shape[0]
        
        print("*"*15)
        print(f"Average val loss: {(val_loss / examples_seen):.2f}")
        print("*"*15)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    print("STARTING TRAINING")
    main(args)