import torch


def train_loop(dataloader, model, loss_fn, optimizer, device="cuda"):
  size = len(dataloader.dataset)
  num_batches = len(dataloader)
  train_loss = 0.0
  train_accuracy = 0.0
  for batch, (X, y) in enumerate(dataloader):
    # Compute prediction and loss
    pred = model(X.to(device))
    loss = loss_fn(pred, y.to(device))
    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    train_loss += loss.item()
    train_accuracy += (pred.argmax(1) == y.to(device)).type(torch.float).sum().item()


    if batch % 10000 == 0:
      loss, current = loss.item(), batch * len(X)
      print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
      print()

  train_loss /= num_batches
  train_accuracy /= size
  return train_loss, train_accuracy


def valid_loop(dataloader, model, loss_fn, metric_fn ,device="cuda"):
  size = len(dataloader.dataset)
  num_batches = len(dataloader)
  test_loss, test_metric = 0, 0

  with torch.no_grad():
    for X, y in dataloader:
      pred = model(X.to(device))
      test_loss += loss_fn(pred, y.to(device)).item()
      test_metric += metric_fn(pred, y.to(device)).item()
      #test_accuracy += (pred.argmax(1) == y.to(device)).type(torch.float).sum().item()

  test_loss /= num_batches
  test_metric /= size
  print(f"valid Error: \n Accuracy: {test_metric}%, Avg loss: {test_loss:>8f} \n")
  return test_loss, test_metric


def test_loop(dataloader, model, loss_fn, metric_fn, device="cuda"):
  size = len(dataloader.dataset)
  num_batches = len(dataloader)

  test_losses_to_print = []
  test_metrics_to_print = []

  with torch.no_grad():
    for X, y in dataloader:
      pred = model(X.to(device))
      test_loss = loss_fn(pred, y.to(device)).item()
      test_metrics = metric_fn(pred, y.to(device)).item()
      #test_accuracy += (pred.argmax(1) == y.to(device)).type(torch.float).sum().item()

      test_losses_to_print.append(test_loss)
      test_metrics_to_print.append(test_metrics)
      
  return test_losses_to_print, test_metrics_to_print