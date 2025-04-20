import torch

def train_model(model, data_loader, optimizer, device, num_epochs, model_save_path="model.pth", patience=5):
    """
    Train the object detection model using the provided DataLoader and optimizer.
    Includes early stopping based on validation loss improvement.

    Args:
        model: The Faster R-CNN model to train.
        data_loader: PyTorch DataLoader for training data.
        optimizer: Optimizer for training.
        device: The device (CPU or GPU) to train on.
        num_epochs: Total number of epochs to train.
        model_save_path: Path to save the best-performing model.
        patience: Number of epochs to wait for improvement before stopping.
    """
    model.train()
    best_loss = float('inf')           # Best validation loss seen so far
    patience_counter = 0               # Counter for early stopping

    for epoch in range(num_epochs):
        print(f"Start training Epoch {epoch+1}/{num_epochs}")
        epoch_loss = 0.0

        for i, batch in enumerate(data_loader):
            if batch is None:
                continue
            images, targets, _ = batch
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items() if k != 'filename'} for t in targets]

            # Forward pass and compute total loss
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # Backpropagation
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()

        avg_loss = epoch_loss / len(data_loader)
        print(f"Epoch {epoch+1} completed, Average Loss: {avg_loss:.4f}")

        # Early stopping: check if loss improved
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)
            print(f"✅ Model improved. Saved to {model_save_path}")
        else:
            patience_counter += 1
            print(f"⚠️ No improvement. Patience counter: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("⛔ Early stopping triggered. Training stopped.")
                break

    return model
