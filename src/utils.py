import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,classification_report, ConfusionMatrixDisplay
import json
from torchvision import transforms as T
from PIL import Image
import random
import numpy as np
import cv2
import mlflow
import tempfile
import os
from mlflow.tracking import MlflowClient

def get_start_epoch(model_accuracy_dict):
    return len(model_accuracy_dict["metrics"]["train_acc"])


def start_mlflow_run(
    experiment_name,
    model,
    optimizer,
    criterion,
    start_epoch,
    epochs,
):
    mlflow.set_experiment(experiment_name)

    run = mlflow.start_run(
        run_name=f"{experiment_name}_epochs_{start_epoch}_to_{start_epoch + epochs - 1}"
    )

    # Static metadata (log ONCE per run)
    mlflow.log_param("model_type", model.__class__.__name__)
    mlflow.log_param("criterion", criterion.__class__.__name__)
    mlflow.log_param("optimizer", optimizer.__class__.__name__)

    mlflow.log_param("start_epoch", start_epoch)
    mlflow.log_param("end_epoch", start_epoch + epochs - 1)

    # Optimizer static params
    mlflow.log_param("weight_decay", optimizer.defaults.get("weight_decay", None))

    # Tags for lineage
    mlflow.set_tag("training_mode", "incremental")
    mlflow.set_tag("model_family", experiment_name.split("_")[1])
    mlflow.set_tag("task", experiment_name.split("_")[-1])

    return run


def model_inference(experiment_name,model,train_loader,val_loader,epochs,optimizer,criterion,
                    best_path,save_path,device,model_accuracy_dict,model_dict_path):

    start_epoch = get_start_epoch(model_accuracy_dict)
    best_acc = max(model_accuracy_dict["metrics"]["val_acc"]) \
        if model_accuracy_dict["metrics"]["val_acc"] else 0.0

    run = start_mlflow_run(
        experiment_name=experiment_name,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        start_epoch=start_epoch,
        epochs=epochs,
    )

    running_loss=0.0
  

    for local_epoch in range(epochs):
        global_epoch = start_epoch + local_epoch

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
                print('[%d,%5d] loss: %.3f'%(local_epoch+1,i+1,running_loss/100))
                running_loss=0.0

        train_acc=correct/total
        print('Train Accuracy: %.3f'%train_acc)
        model_accuracy_dict["metrics"]["train_acc"].append(train_acc)

        mlflow.log_metric("train_accuracy", train_acc, step=global_epoch)
        for i, group in enumerate(optimizer.param_groups):
            mlflow.log_metric(
                f"lr_group_{i}",
                group["lr"],
                step=global_epoch
            )
            
        torch.save(model.state_dict(),save_path)
       
        with tempfile.TemporaryDirectory() as tmpdir:
            epoch_model_path = os.path.join(
                tmpdir, f"model_epoch_{global_epoch}.pth"
            )

            torch.save(model.state_dict(), epoch_model_path)

            mlflow.log_artifact(
                epoch_model_path,
                artifact_path="epoch_models"
            )
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

        val_acc = correct / total
        print('Validation Accuracy: %.3f'%val_acc)

        model_accuracy_dict["metrics"]["val_acc"].append(val_acc)
        mlflow.log_metric("val_accuracy", val_acc, step=global_epoch)


        if val_acc > best_acc:

            if model_accuracy_dict["mode"] == "Transfer_learning":
                Registered_model_name = "DR_Transfer_Classifier"
            elif model_accuracy_dict["mode"] == "Binary_cls":
                Registered_model_name = "DR_Binary_Classifier"

            best_acc = val_acc
            torch.save(model.state_dict(), best_path)
            print(f"Model saved with accuracy: {best_acc:.3f}")

            # 1️⃣ Log ONLY the weights
            mlflow.log_artifact(best_path, artifact_path="weights")

            client = MlflowClient()
            run_id = mlflow.active_run().info.run_id

            # 2️⃣ Register weights as a new model version
            model_uri = f"runs:/{run_id}/weights/{os.path.basename(best_path)}"
            mv = client.create_model_version(
                name=Registered_model_name,
                source=model_uri,
                run_id=run_id
            )
            new_version = mv.version

            # 3️⃣ Archive previous Production model (if exists)
            for mv in client.search_model_versions(f"name='{Registered_model_name}'"):
                if mv.current_stage == "Production":
                    client.transition_model_version_stage(
                        name=Registered_model_name,
                        version=mv.version,
                        stage="Archived"
                    )

            # 4️⃣ Promote new model to Production
            client.transition_model_version_stage(
                name=Registered_model_name,
                version=new_version,
                stage="Production"
            )

            # 5️⃣ Tag metadata
            client.set_model_version_tag(
                name=Registered_model_name,
                version=new_version,
                key="best_val_accuracy",
                value=str(best_acc)
            )
            client.set_model_version_tag(
                name=Registered_model_name,
                version=new_version,
                key="best_epoch",
                value=str(global_epoch)
            )

            mlflow.log_metric("best_val_accuracy", best_acc)
            mlflow.set_tag("best_epoch", global_epoch)

        cm=confusion_matrix(labels_list,predicted_list)
        disp=ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.show()

        print(classification_report(labels_list,predicted_list))

    with open(model_dict_path, "w") as f:
        json.dump(model_accuracy_dict, f, indent=4)

    mlflow.log_artifact(model_dict_path)

    mlflow.end_run()

    return model_accuracy_dict

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

