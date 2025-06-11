import torch

def train(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=30):
    try:
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0
            for X_batch, Y_batch in train_loader:
                X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, Y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * X_batch.size(0)
            train_loss /= len(train_loader.dataset)

            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_batch, Y_batch in val_loader:
                    X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
                    outputs = model(X_batch)
                    loss = criterion(outputs, Y_batch)
                    val_loss += loss.item() * X_batch.size(0)
                val_loss /= len(val_loader.dataset)

            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
    except:
        return ["X_batch:", X_batch.shape, "outputs:", outputs.shape, "Y_batch:", Y_batch.shape]