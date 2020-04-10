import time
from config import cfg
import torch


def evaluate(net, dataset, criterion=cfg.MODEL.CRITERION):
    model = net.model
    dataloaders = dataset.dataloaders
    since = time.time()
    avg_loss = 0
    avg_acc = 0
    test_loss = 0
    test_acc = 0

    test_batches = len(dataloaders[cfg.CONST.TEST])

    print("Evaluating model")
    print('-' * 10)

    with torch.no_grad():
        # Set to evaluation mode
        model.eval()

        # Validation loop
        for i, (data, target) in enumerate(dataloaders[cfg.CONST.TEST]):
            # Tensors to gpu
            if cfg.CONST.USE_GPU:
                data, target = data.cuda(), target.cuda()

            # Forward pass
            output = model(data)

            # Validation loss
            loss = criterion(output, target)
            # Multiply average loss times the number of examples in batch
            test_loss += loss.item() * data.size(0)

            # Calculate validation accuracy
            _, pred = torch.max(output, dim=1)
            correct_tensor = pred.eq(target.data.view_as(pred))
            accuracy = torch.mean(
                correct_tensor.type(torch.FloatTensor))
            # Multiply average accuracy times the number of examples
            test_acc += accuracy.item() * data.size(0)

            del data, target, output, pred, _, correct_tensor
            torch.cuda.empty_cache()

        # Calculate average losses
        test_loss = test_loss / len(dataloaders[cfg.CONST.TEST].dataset)

        # Calculate average accuracy
        test_acc = test_acc / len(dataloaders[cfg.CONST.TEST].dataset)

    elapsed_time = time.time() - since
    print()
    print("Evaluation completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
    print("Avg loss (test): {:.4f}".format(test_loss))
    print("Avg acc (test): {:.4f}".format(test_acc))
    print('-' * 10)