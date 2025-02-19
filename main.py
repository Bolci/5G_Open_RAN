# Place to work on the optimization of the model

import argparse
from Framework.utils.utils import load_json_as_dict, save_txt
from Framework.preprocessors.data_preprocessor import DataPreprocessor
from Framework.preprocessors.data_path_worker import get_all_paths
from Framework.preprocessors.data_utils import get_data_loaders, get_datasets
from Framework.metrics.metrics import RMSELoss
from Framework.Model_bank.autoencoder_cnn import CNNAutoencoder, CNNAutoencoderV2, CNNAutoencoderDropout
from Framework.Model_bank.autoencoder_LSTM import LSTMAutoencoder, LSTMAutoencoderCustom
from Framework.Model_bank.AE_CNN_v2 import CNNAEV2
from Framework.Model_bank.transformer_ae import TransformerAutoencoder
from Framework.Model_bank.autoencoder_cnn1d import Autoencoder1D
from Framework.Model_bank.autoencoder_rnn import RNNAutoencoder
from Framework.loops.loops import train_loop, valid_loop, test_loop, test_loop_general
from Framework.postprocessors.postprocessor_functions import plot_data_by_labels, mean_labels_over_epochs
from Framework.postprocessors.tester import Tester
from Framework.postprocessors.postprocessor_utils import get_print_message
from Framework.postprocessors.graph_worker import get_distribution_plot
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.optim.lr_scheduler as lr_scheduler
import wandb
import datetime

def train_with_hp_setup(datasets, model, batch_size, learning_rate, no_epochs, device, criterion):
    dataloaders = get_data_loaders(datasets, batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=no_epochs//2)

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
        print("Epoch %d: lr %.8f -> %.8f" % (epoch, before_lr, after_lr))

        model.eval()
        valid_loss_mean, valid_loss_all, _ = valid_loop(dataloaders['Valid'][0],
                                                        model,
                                                        criterion,
                                                        device=device)

        if wandb.run is not None:
            wandb.log({"train_loss": train_loss, "val_loss": valid_loss_mean, "lr": before_lr, "epoch": epoch})
        train_loss_mean_save.append(train_loss)
        valid_loss_mean_save.append(valid_loss_mean)
        valid_loss_all_save.append(valid_loss_all)

    model.eval()
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
                                                           rewrite_data = False,
                                                           merge_files = True,
                                                           mix_test = True,
                                                           split_train_into_train_and_valid=True)

    #prepare datasets and data_loaders
    datasets = get_datasets(paths_for_datasets)

    '''
    model = LSTMAutoencoderCustom(input_dimensions=72,
                                  expansion_dim=args.expansion_dim,
                                  no_layers_per_module=args.no_layers_per_module,
                                  num_layers_per_layer = args.num_layers_per_layer,
                                  init_channels=args.init_channels,
                                  dropout=args.dropout,
                                  device=device)
    '''
    # model = CNNAutoencoder(48)
    # model = TransformerAutoencoder(embed_dim=args.embed_dim, num_heads=args.num_heads, num_layers=args.num_layers, dropout=args.dropout)
    options = [64, 32, 16, 8, 4]
    hidden_dims = options[:args.num_layers]
    # model = LSTMAutoencoder(72, hidden_dims, args.dropout)
    model = Autoencoder1D()
    # model = RNNAutoencoder(72, [16, 8, 4], "lstm")
    criterion = RMSELoss()

    train_loss_mean_save, valid_loss_mean_save, valid_loss_all_save, train_dist_score = (
        train_with_hp_setup(datasets, model, args.batch_size, args.learning_rate, args.epochs, device, criterion))

    valid_metrics =  mean_labels_over_epochs(valid_loss_all_save)

    #preparing saving paths
    now = datetime.datetime.now()
    folder_name = now.strftime("%Y%m%d%H%M%S")

    saving_folder_name = f"Try_Preprocessing={args.preprocesing_type}_no-epochs={args.epochs}_lr={args.learning_rate}_bs={args.batch_size}_model={model.model_name}_{folder_name}"
    saving_path = os.path.join(path['Saving_path'], saving_folder_name)
    if wandb.run is not None:
        wandb.log({"saving_dir": saving_folder_name})

    if not os.path.exists(saving_path):
        os.makedirs(saving_path)

    #file_names
    train_over_epoch = 'train_over_epoch.txt'
    valid_over_epoch = 'valid_over_epoch.txt'
    valid_over_epoch_over_batch_with_labels = 'valid_epochs_labels.txt'
    train_final_score_per_batch = 'train_final_per_batch.txt'

    #saving metrics
    path_training_over_epochs = os.path.join(saving_path, train_over_epoch)
    save_txt(path_training_over_epochs, train_loss_mean_save)
    path_valid_over_epochs = os.path.join(saving_path, valid_over_epoch)
    save_txt(path_valid_over_epochs, valid_loss_mean_save)
    path_valid_epochs_labels = os.path.join(saving_path, valid_over_epoch_over_batch_with_labels)
    save_txt(path_valid_epochs_labels, valid_loss_all_save)
    path_train_final_per_batch = os.path.join(saving_path, train_final_score_per_batch)
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

    #testing loop
    tester = Tester(result_folder_path=path['Saving_path'],
                    attempt_name=saving_folder_name,
                    train_score_over_epoch_file_name=train_over_epoch,
                    valid_score_over_epoch_file_name=valid_over_epoch,
                    valid_score_over_epoch_per_batch_file_name=valid_over_epoch_over_batch_with_labels,
                    train_score_final_file_name=train_final_score_per_batch)

    #valid loop
    valid_scores = tester.estimate_decision_lines()
    print('Validation scores is:')
    print(valid_scores)

    #test data loader and loop
    predictions_buffer = []
    performance = []
    metrics_buffer = []

    for id_dat, single_test_dataset in enumerate(datasets['Test']):
        testing_loop = lambda class_metric: test_loop_general(single_test_dataset, model, criterion, class_metric, device=device)
        test_scores, predictions, metrics = tester.test_data(testing_loop=testing_loop)
        print("=============================")
        print(f'Test scores, dataset_id {id_dat}')
        print(f"Dataset path is: {paths_for_datasets['Test'][id_dat]}")
        print(test_scores)
        predictions_buffer.append(predictions)
        performance.append(test_scores)
        metrics_buffer.append(metrics)
        if wandb.run is not None:
            for tester_label, single_scores in test_scores.items():
                for single_scores_type_label, single_score_type_value in test_scores.items():
                    wandb.log({f"tester_{tester_label}_type={paths_for_datasets['Test'][id_dat].split('/')[-1]}": single_score_type_value})

    fig_distribution, subfigures = get_distribution_plot(valid_loss_all_save[-1], predictions_buffer, performance, metrics_buffer)

    graph_valid_test_distribution = os.path.join(saving_path, 'error_distribution.png')
    fig_distribution.savefig(graph_valid_test_distribution)
    for i, fig in enumerate(subfigures):
        graph_valid_test_distribution_separated = os.path.join(saving_path, f"error_distribution_test{i}.png")
        fig.savefig(graph_valid_test_distribution_separated)
    if wandb.run is not None:
        wandb.finish()




if __name__ == "__main__":
    #wandb.init(project="Anomaly_detection", config={"epochs": 10, "batch_size": 32})
    paths_config = load_json_as_dict('./local_data_path_no_valid.json')

    parser = argparse.ArgumentParser(description="OpenRAN neural network")
    parser.add_argument(
        "--epochs", type=int, default=150, help="Number of epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32535, help="Batch size"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.0005, help="Learning rate"
    )
    parser.add_argument(
        "--expansion_dim", type=int, default=2, help="Learning rate"
    )
    parser.add_argument(
        "--no_layers_per_module", type=int, default=5, help="Learning rate"
    )
    parser.add_argument(
        "--num_layers_per_layer", type=int, default=1, help="Learning rate"
    )
    parser.add_argument(
        "--init_channels", type=int, default=12, help="Learning rate"
    )
    parser.add_argument(
        "--dropout", type=float, default=0.5, help="Learning rate"
    )
    # parser.add_argument(
    #     "--log_interval", type=int, default=1, help="Log interval"
    # )
    parser.add_argument(
        "--embed_dim", type=int, default=32, help="Embedding dimension"
    )
    parser.add_argument(
        "--num_heads", type=int, default=8, help="Multihead attention heads"
    )
    parser.add_argument(
        "--num_layers", type=int, default=3, help="Number of endoder and decoder layers"
    )
    parser.add_argument(
        "--preprocesing_type", type=str, default="abs_only_multichannel", help="Log interval"
    )
    parser.add_argument(
        "--wandb_log", type=bool, default=True, help="Log to wandb"
    )
    args = parser.parse_args()

    # os.environ["WANDB_SILENT"] = "true"
    if args.wandb_log:
        wandb.init(
            project="Anomaly_detection",
            entity="OPEN_5G_RAN_team",
            #name="all_50_complex",
            config=vars(parser.parse_args()),
            mode="online",
            #tags=[f"without scheduler"]
        )

    main(paths_config, args)
