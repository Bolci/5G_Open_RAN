def test_loop(dataloader_test, model, loss_fn, device="cuda"):
    test_losses_to_print = []

    with torch.no_grad():
        for X, y in dataloader_test:
            pred = model(X.to(device))
            test_loss = loss_fn(pred, X).item()
            test_losses_to_print.append(([copy(y.item()), copy(test_loss)]))

    return test_losses_to_print