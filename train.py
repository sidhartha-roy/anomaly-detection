import time
import numpy as np
from config import cfg
import torch
import pandas as pd


def train(net, dataset):
    """
    Used for training the network
    :param net: network/model instance. Contains the parameters.
    :return: trained_net, training history
    """
    # early stopping initilization
    epochs_no_improve = 0
    valid_loss_min = np.Inf

    valid_max_acc = 0
    history = []

    # Number of epochs already trained (if using loaded in model weights)
    try:
        print(f'Model has been trained for: {net.model.epochs} epochs.\n')
    except:
        net.model.epochs = 0
        print(f'Starting Training from Scratch.\n')

    overall_start = time.time()

    # Main loop
    for epoch in range(cfg.MODEL.N_EPOCHS):

        # keep track of training and validation loss each epoch
        train_loss = 0.0
        valid_loss = 0.0

        train_acc = 0
        valid_acc = 0

        # Set to training
        net.model.train()
        start = time.time()

        # Training loop
        for ii, (data, target) in enumerate(dataset.dataloaders[dataset.TRAIN]):

            # tensors to gpu
            if cfg.CONST.USE_GPU:
                data, target = data.cuda(), target.cuda()

            # clear gradients
            net.optimizer.zero_grad()

            # Predicted outputs are log probabilities
            output = net.model(data)

            # Loss and backpropagation of gradients
            loss = net.criterion(output, target)
            loss.backward()

            # Update the parameters
            net.optimizer.step()

            # Track train loss by multiplying average loss by number of examples in batch

            train_loss += loss.item() * data.size(0)

            # Calculate accuracy by finding max log probability
            _, pred = torch.max(output, dim=1)
            correct_tensor = pred.eq(target.data.view_as(pred))

            # Need to convert correct tensor from int to float to average
            accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
            # Multiply average accuracy times the number of examples in batch
            train_acc += accuracy.item() * data.size(0)

            # Track training progress
            print(
                f'Epoch: {epoch}\t{100 * (ii + 1) / len(dataset.dataloaders[dataset.TRAIN]):.2f}% complete. {time.time() - start:.2f} seconds elapsed in epoch.',
                end='\r')

        # After training loops ends, start validation
        else:
            net.model.epochs += 1

            # Don't need to keep track of gradients
            with torch.no_grad():
                # Set to evaluation mode
                net.model.eval()

                # Validation loop
                for data, target in dataset.dataloaders[dataset.VAL]:
                    # Tensors to gpu
                    if cfg.CONST.USE_GPU:
                        data, target = data.cuda(), target.cuda()

                    # Forward pass
                    output = net.model(data)

                    # Validation loss
                    loss = net.criterion(output, target)
                    # Multiply average loss times the number of examples in batch
                    valid_loss += loss.item() * data.size(0)

                    # Calculate validation accuracy
                    _, pred = torch.max(output, dim=1)
                    correct_tensor = pred.eq(target.data.view_as(pred))
                    accuracy = torch.mean(
                        correct_tensor.type(torch.FloatTensor))
                    # Multiply average accuracy times the number of examples
                    valid_acc += accuracy.item() * data.size(0)

                # Calculate average losses
                train_loss = train_loss / len(dataset.dataloaders[dataset.TRAIN].dataset)
                valid_loss = valid_loss / len(dataset.dataloaders[dataset.VAL].dataset)

                # Calculate average accuracy
                train_acc = train_acc / len(dataset.dataloaders[dataset.TRAIN].dataset)
                valid_acc = valid_acc / len(dataset.dataloaders[dataset.VAL].dataset)

                history.append([train_loss, valid_loss, train_acc, valid_acc])

                # Print training and validation results
                if (epoch + 1) % cfg.MODEL.TRAIN_PRINT_EVERY == 0:
                    print(
                        f'\nEpoch: {epoch} \tTraining Loss: {train_loss:.4f} \tValidation Loss: {valid_loss:.4f}'
                    )
                    print(
                        f'\t\tTraining Accuracy: {100 * train_acc:.2f}%\t Validation Accuracy: {100 * valid_acc:.2f}%'
                    )

                    # Save the model if validation loss decreases
                if valid_loss < valid_loss_min:
                    # Save model
                    torch.save(net.model.state_dict(), cfg.MODEL.FILENAME)
                    # Track improvement
                    epochs_no_improve = 0
                    valid_loss_min = valid_loss
                    valid_best_acc = valid_acc
                    best_epoch = epoch

                # Otherwise increment count of epochs with no improvement
                else:
                    epochs_no_improve += 1
                    # Trigger early stopping
                    if epochs_no_improve >= cfg.MAX_EPOCHS_STOP:
                        print(
                            f'\nEarly Stopping! Total epochs: {epoch}. Best epoch: {best_epoch} with loss: {valid_loss_min:.2f} and acc: {100 * valid_acc:.2f}%'
                        )
                        total_time = time.time() - overall_start
                        print(
                            f'{total_time:.2f} total seconds elapsed. {total_time / (epoch + 1):.2f} seconds per epoch.'
                        )

                        # Load the best state dict
                        net.model.load_state_dict(torch.load(cfg.MAX_EPOCHS_STOP))
                        # Attach the optimizer
                        net.model.optimizer = net.optimizer

                        # Format history
                        history = pd.DataFrame(
                            history,
                            columns=[
                                'train_loss', 'valid_loss', 'train_acc',
                                'valid_acc'
                            ])
                        return net.model, history

    # Attach the optimizer
    net.model.optimizer = net.optimizer
    # Record overall time and print out stats
    total_time = time.time() - overall_start
    print(
        f'\nBest epoch: {best_epoch} with loss: {valid_loss_min:.2f} and acc: {100 * valid_acc:.2f}%'
    )
    print(
        f'{total_time:.2f} total seconds elapsed. {total_time / (epoch):.2f} seconds per epoch.'
    )
    # Format history
    history = pd.DataFrame(
        history,
        columns=['train_loss', 'valid_loss', 'train_acc', 'valid_acc'])
    return net, history
