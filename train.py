import torch
import torch.nn as nn
import numpy as np
from model import *
from dataset import *
import matplotlib.pyplot as plt
import tqdm

def train_model(model, trainloader, validloader, epochs=100, visualize_learning_curve=True):
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    valid_loss_min = np.Inf
    train_losses, test_losses = [], []
    for e in range(epochs):
        model.train()
        running_loss = 0
        tr_accuracy = 0
        for images, labels in trainloader:
            images = images.cuda()
            labels = labels.long().cuda()
            optimizer.zero_grad()
            
            log_ps  = model(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            # tr_accuracy += torch.mean(equals.type(torch.FloatTensor))
            tr_accuracy += torch.mean(equals.type(torch.FloatTensor).cuda())
        
        test_loss = 0
        accuracy = 0
        with torch.no_grad():
            model.eval()

            progress_bar = tqdm(trainloader, desc=f"Epoch {e+1}/{epochs}", unit="batch")

            for images, labels in validloader:
                images = images.cuda()
                labels = labels.long().cuda()
                log_ps = model(images)
                test_loss += criterion(log_ps, labels).item()
                
                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                # accuracy += torch.mean(equals.type(torch.FloatTensor))
                accuracy += torch.mean(equals.type(torch.FloatTensor).cuda())

            progress_bar = tqdm(trainloader, desc=f"Epoch {e+1}/{epochs}", unit="batch")

        train_losses.append(running_loss/len(trainloader))
        test_losses.append(test_loss/len(validloader))

        print("Epoch: {}/{} ".format(e+1, epochs),
            "Training Loss: {:.3f} ".format(train_losses[-1]),
            "Training Acc: {:.3f} ".format(tr_accuracy/len(trainloader)),
            "Val Loss: {:.3f} ".format(test_losses[-1]),
            "Val Acc: {:.3f}".format(accuracy/len(validloader)))
        if test_loss/len(validloader) <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            test_loss/len(validloader)))
            torch.save(model.state_dict(), 'best_model.pt')
            valid_loss_min = test_loss/len(validloader)

    if visualize_learning_curve:        
        plt.plot(train_losses, 'b', label='Training Loss')
        plt.plot(test_losses, 'r', label='Validation Loss')
        plt.legend()
        plt.show()
    return model



def main():
    # trainloader, validloader = get_dataloaders()
    # print('Data Preprocessed and got DataLoaders...')

    path_to_fer_csv = r'C:\Users\Vandana\Desktop\fer code\Facial-Emotion-Recognition-PyTorch-ONNX\fer2013.csv'  
    trainloader, validloader = get_dataloaders(path_to_fer_csv)  # Pass the path to get_dataloaders
    print('Data Preprocessed and got DataLoaders...')

    model = Face_Emotion_CNN()
    if torch.cuda.is_available():
        model.cuda()
        print('GPU Found!!!, Moving Model to CUDA.')
    else:
        model.cpu()

    print('GPU not found!!, using model with CPU.')

    print('Starting Training loop...\n')
    model = train_model(model, trainloader, validloader, epochs=200)    


if __name__ == '__main__':
    main()