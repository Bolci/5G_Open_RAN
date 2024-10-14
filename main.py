# Place to work on the optimization of the model

import argparse
from Framework.utils.utils import load_json_as_dict, save_txt
from Framework.preprocessors.data_preprocessor import DataPreprocessor
from Framework.preprocessors.data_path_worker import get_all_paths
from Framework.preprocessors.data_utils import get_data_loaders, get_datasets
from Framework.metrics.metrics import RMSELoss
from Framework.Model_bank.autoencoder_cnn import CNNAutoencoder, CNNAutoencoderV2
from Framework.loops.loops import train_loop, valid_loop
from Framework.postprocessors.postprocessor_functions import plot_data_by_labels, mean_labels_over_epochs
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.optim.lr_scheduler as lr_scheduler
import wandb


def train_with_hp_setup(datasets, model, batch_size, learning_rate, no_epochs, device):
    dataloaders = get_data_loaders(datasets, batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=no_epochs//2)
    criterion = RMSELoss()

    model.to(device)
    criterion.to(device)

    #do one validation loop to ini everything
    _, _, _ = valid_loop(dataloaders['Valid'][0],
                      model,
                      criterion,
                      device=device)

    train_loss_mean_save = []
    valid_loss_mean_save = []
    valid_loss_all_save = []
    for epoch in range(no_epochs):
        print(f"Epoch {epoch + 1}\n-------------------------------")

        model.train()
        train_loss = train_loop(dataloaders['Train'][0],
                                model,
                                criterion,
                                optimizer,
                                device=device)
        before_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        after_lr = optimizer.param_groups[0]["lr"]
        print("Epoch %d: SGD lr %.8f -> %.8f" % (epoch, before_lr, after_lr))

        model.eval()
        valid_loss_mean, valid_loss_all, _ = valid_loop(dataloaders['Valid'][0],
                                                        model,
                                                        criterion,
                                                        device=device)
        wandb.log({"train_loss": train_loss, "val_loss": valid_loss_mean, "epoch": epoch})
        train_loss_mean_save.append(train_loss)
        valid_loss_mean_save.append(valid_loss_mean)
        valid_loss_all_save.append(valid_loss_all)

    _, _, train_dist_score = valid_loop(dataloaders['Train'][0],
                                                 model,
                                                 criterion,
                                                 device=device,
                                                 is_train=True)


    return train_loss_mean_save, valid_loss_mean_save, valid_loss_all_save, train_dist_score



def main(path, args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Dataset_paths
    all_paths = get_all_paths(path)

    #Data preparing according to preprocessing
    data_preprocessor = DataPreprocessor()
    data_preprocessor.set_cache_path(path["Data_cache_path"])
    data_preprocessor.set_original_seg(path["True_sequence_path"])

    paths_for_datasets = data_preprocessor.preprocess_data(all_paths,
                                                           args.preprocesing_type,
                                                           rewrite_data = True,
                                                           merge_files = True)

    #prepare datasets and data_loaders
    datasets = get_datasets(paths_for_datasets)
    model = CNNAutoencoderV2()

    train_loss_mean_save, valid_loss_mean_save, valid_loss_all_save, train_dist_score = (
        train_with_hp_setup(datasets, model, args.batch_size, args.learning_rate, args.epochs, device))

    valid_metrics =  mean_labels_over_epochs(valid_loss_all_save)

    #preparing saving paths
    saving_folder_name = f"Try_Preprocessing={args.preprocesing_type}_no-epochs={args.epochs}_lr={args.learning_rate}_bs={args.batch_size}_model={model.model_name}"
    saving_path = os.path.join(path['Saving_path'], saving_folder_name)

    if not os.path.exists(saving_path):
        os.makedirs(saving_path)

    #saving metrics
    path_training_over_epochs = os.path.join(saving_path, 'train_over_epoch.txt')
    save_txt(path_training_over_epochs, train_loss_mean_save)
    path_valid_over_epochs = os.path.join(saving_path, 'valid_over_epoch.txt')
    save_txt(path_valid_over_epochs, valid_loss_mean_save)
    path_valid_epochs_labels = os.path.join(saving_path, 'valid_epochs_labels.txt')
    save_txt(path_valid_epochs_labels, valid_loss_all_save)
    path_train_final_per_batch = os.path.join(saving_path, 'train_final_per_batch.txt')
    save_txt(path_train_final_per_batch, train_dist_score)


    #saving loss over epochs
    plt.figure()
    plt.plot(train_loss_mean_save, label='Train')
    plt.plot(valid_loss_mean_save, label='Valid')
    fig_path = os.path.join(saving_path, 'fig_1_train_valid.png')
    plt.xlabel("epochs")
    plt.ylabel("RMSE")
    plt.grid()
    plt.legend()
    plt.savefig(fig_path)

    plt.figure()
    plt.plot(train_loss_mean_save, label='Train')
    plt.plot(valid_metrics['Class_0'], label='Valid_class_0')
    plt.plot(valid_metrics['Class_1'], label='Valid_class_1')
    fig_path = os.path.join(saving_path, 'fig_1_train_valid_labels.png')
    plt.xlabel("epochs")
    plt.ylabel("RMSE")
    plt.grid()
    plt.legend()
    plt.savefig(fig_path)

    plt.figure()
    plt.plot(np.arange(len(train_loss_mean_save))[10:], train_loss_mean_save[10:], label='Train')
    plt.plot(np.arange(len(train_loss_mean_save))[10:], valid_metrics['Class_0'][10:], label='Valid_class_0')
    plt.plot(np.arange(len(train_loss_mean_save))[10:], valid_metrics['Class_1'][10:], label='Valid_class_1')
    fig_path = os.path.join(saving_path, 'fig_1_train_valid_labels_zoomed.png')
    plt.xlabel("epochs")
    plt.ylabel("RMSE")
    plt.grid()
    plt.legend()
    plt.savefig(fig_path)

    #save model
    model_path = os.path.join(saving_path, 'model.pt')
    torch.save(model.state_dict(), model_path)

    #valid_graph
    sv_path = os.path.join(saving_path, 'valid_graph.png')
    plot_data_by_labels(valid_loss_all_save, sv_path)
    wandb.finish()


if __name__ == "__main__":
    #wandb.init(project="Anomaly_detection", config={"epochs": 10, "batch_size": 32})
    paths_config = load_json_as_dict('./data_paths.json')

    parser = argparse.ArgumentParser(description="OpenRAN neural network")
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="Learning rate"
    )
    parser.add_argument(
        "--log_interval", type=int, default=1, help="Log interval"
    )
    parser.add_argument(
        "--preprocesing_type", type=str, default="abs_only_by_one_sample", help="Log interval"
    )
    args = parser.parse_args()

    # os.environ["WANDB_SILENT"] = "true"
    wandb.init(
        project="Anomaly_detection",
        entity="OPEN_5G_RAN_team",
        config=vars(parser.parse_args()),
        mode="online",
        # tags=[f"NewV{i}.{j}.4"],
    )

    main(paths_config, args)
