import torch
import torch.nn as nn
import torch.optim as optim
import mlflow

def train(model,trainloader,config,lr:int=0.001): 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size=config["batch_size"]
    epochs=config['epoch']
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):  # 10 époques pour l'exemple
       
        running_loss = 0.0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f"Époque {epoch+1}, perte: {running_loss/len(trainloader):.4f}")
        mlflow.log_metric("loss",running_loss/len(trainloader))




