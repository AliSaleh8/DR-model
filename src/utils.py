import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import json
from torchvision import transforms as T
from PIL import Image
import random
import numpy as np
import cv2
from sklearn.metrics import classification_report




def model_inference(model,train_loader,val_loader,epochs,optimizer,criterion,best_path,save_path,device,best_accuracy=None):


    running_loss=0.0
    train_accuracy=[]
    val_accuracy=[]
    best_acc=0.0 if best_accuracy is None else best_accuracy



    for epoch in range(epochs):

        total=0
        correct=0

        model.train()

        for i,data in enumerate(train_loader):

            inputs,labels=data
            inputs,labels=inputs.to(device),labels.to(device)

            optimizer.zero_grad()

            outputs=model(inputs)

            loss=criterion(outputs,labels)
            loss.backward()

            optimizer.step()
            running_loss+=loss.item()

            _,predicted=torch.max(outputs.data,1)
            correct+=(predicted==labels).sum().item()
            total+=labels.size(0)

            if i%100==99:
                print('[%d,%5d] loss: %.3f'%(epoch+1,i+1,running_loss/100))
                running_loss=0.0

        accuracy=correct/total
        print('Train Accuracy: %.3f'%accuracy)
        train_accuracy.append(accuracy)
        torch.save(model.state_dict(),save_path)

        model.eval()

        total=0
        correct=0
        predicted_list=[]
        labels_list=[]

        for i,data in enumerate(val_loader,0):
            inputs,labels=data
            inputs,labels=inputs.to(device),labels.to(device)

            with torch.no_grad():
                outputs=model(inputs)

            _,predicted=torch.max(outputs.data,1)
            correct+=(predicted==labels).sum().item()
            total+=labels.size(0)

            predicted_list.extend(predicted.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())

        accuracy=correct/total
        print('Validation Accuracy: %.3f'%accuracy)
        val_accuracy.append(accuracy)


        if accuracy>best_acc:
            best_acc=accuracy
            torch.save(model.state_dict(),best_path)
            print('Model saved with accuracy: %.3f'%best_acc)

        cm=confusion_matrix(labels_list,predicted_list)
        disp=ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.show()

        print(classification_report(labels_list,predicted_list))

    return train_accuracy,val_accuracy

def accuracy_plot(accuaracy_dict):

    dict_length=len(accuaracy_dict)

    value=next(iter(accuaracy_dict.values()))

    if isinstance(value,dict):
        cols=3
        rows=dict_length//cols + (dict_length%cols>0)
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
        axes = axes.flatten()
        for i, (key, value) in enumerate(accuaracy_dict.items()):
            train_acc = value['train_acc']
            val_acc = value['val_acc']
            ax = axes[i]
            ax.plot(train_acc, label='Train Accuracy')
            ax.plot(val_acc, label='Validation Accuracy')
            ax.set_title(f'Model: {key}')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy')
            ax.set_ylim(0, 1)
            ax.legend()

        for j in range(i+1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()

    else:
        train_acc, val_acc = accuaracy_dict['train_acc'], accuaracy_dict['val_acc']
        plt.plot(train_acc,label='Train Accuracy')
        plt.plot(val_acc,label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        plt.legend()
        plt.show()

def accuracy_test(model,testloader,num_classes,device):

    correct_labels=[0]*num_classes
    total_labels=[0]*num_classes

    with torch.no_grad():
        model.eval()
        for data in testloader:
            images,labels=data
            images,labels=images.to(device),labels.to(device)
            outputs=model(images)
            _,predictions=torch.max(outputs,1)
            correct_predictions=(predictions==labels)
            for i in range (len(data)):
                label=labels[i]
                correct_labels[label]+=correct_predictions[i].item()
                total_labels[label]+=1

    for i in range(num_classes):
        print('Accuracy of %5s : %.2f %%'%(i,100*correct_labels[i]/total_labels[i]))

