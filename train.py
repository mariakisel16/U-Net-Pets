from torch.utils.data.dataloader import DataLoader
import matplotlib.pyplot as plt

def train():
  #Hyperparameters
  learning_rate = 0.0001
  batch_size = 10
  epoch = 20
  num_workers = 2
  data_path = path
  model_save_path = "kaggle_working_models"

  os.makedirs(model_save_path, exist_ok=True)



  train_transform = A.Compose([
    A.Resize(256, 256), # Added Resize to ensure consistent input size
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])
  test_transform = A.Compose([
      A.Resize(256, 256),
      A.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
      ToTensorV2()
  ])

  train_dataset = PetDataset(root=data_path, is_train=True, transform=train_transform)
  val_dataset = PetDataset(root=data_path, is_train=False, transform=test_transform)

 

  train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
  val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  #Initializing model
  model = UNet(in_channels=3, out_channels=2).to(device)

  #Initialize optimizer and loss function
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',        # we observe IoU
    patience=3,        # how many epoch we need to wait without improvement
    factor=0.5         # decrease lr by 2
)

  best_val_iou = 0.0
  train_losses = []
  val_losses = []
  train_ious = []
  val_ious = []

  for epoch_idx in range(epoch): # Renamed 'epoch' to 'epoch_idx' to avoid shadowing the variable 'epoch'
        print(f"Epoch {epoch_idx+1}/{epoch}")

        # ------------------ TRAIN ------------------
        model.train()
        train_loss = 0.0
        train_iou = 0.0

        for images, masks in tqdm(train_dataloader):
            images = images.to(device)
            masks = masks.long().to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = torch.argmax(outputs, dim=1)
            iou = calculate_iou(preds.cpu(), masks.cpu())

            train_loss += loss.item()
            train_iou += iou.item()

        train_loss /= len(train_dataloader)
        train_iou /= len(train_dataloader)

        # ------------------ VALIDATION ------------------
        model.eval()
        val_loss = 0.0
        val_iou = 0.0

        with torch.no_grad():
            for images, masks in tqdm(val_dataloader):
                images = images.to(device)
                masks = masks.long().to(device)

                outputs = model(images)
                loss = criterion(outputs, masks)

                preds = torch.argmax(outputs, dim=1)
                iou = calculate_iou(preds.cpu(), masks.cpu())

                val_loss += loss.item()
                val_iou += iou.item()

        val_loss /= len(val_dataloader)
        val_iou /= len(val_dataloader)

        print(f"Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f}")
        print(f"Val Loss:   {val_loss:.4f}, Val IoU:   {val_iou:.4f}")
        print("-" * 50)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_ious.append(train_iou)
        val_ious.append(val_iou)

        scheduler.step(val_iou)

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            model.eval() # Set model to evaluation mode before saving
            torch.save(model.state_dict(), os.path.join(model_save_path, "best_model.pth"))
            print("Model saved!")


  print("Training finished.")


  # ------------------ PLOTS ------------------
  epochs_range = range(1, len(train_losses) + 1)
  plt.figure()
  plt.plot(epochs_range, train_losses)
  plt.plot(epochs_range, val_losses)
  plt.xlabel("Epoch")
  plt.ylabel("Loss")
  plt.legend(["Train Loss", "Val Loss"])
  plt.show()

  plt.figure()
  plt.plot(epochs_range, train_ious)
  plt.plot(epochs_range, val_ious)
  plt.xlabel("Epoch")
  plt.ylabel("IoU")
  plt.title("IoU over Epochs")
  plt.legend(["Train IoU", "Val IoU"])
  plt.show()

  return model, val_dataset, device # Return the trained model and validation dataset

if __name__ == "__main__":
    model, val_dataset, device = train()
