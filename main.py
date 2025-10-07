# Place to work on the optimization of the model

import argparse
from Framework.utils.utils import load_json_as_dict, save_txt
from Framework.preprocessors.data_preprocessor import DataPreprocessor
from Framework.preprocessors.data_path_worker import get_all_paths
from Framework.preprocessors.data_utils import get_data_loaders, get_datasets
from Framework.metrics.metrics import RMSELoss, VAELoss,SSIMLoss
from Framework.models.autoencoder_cnn import CNNAutoencoder, CNNAutoencoderV2, CNNAutoencoderDropout
from Framework.models.autoencoder_LSTM import LSTMAutoencoder, LSTMAutoencoderCustom
from Framework.models.AE_CNN_v2 import CNNAEV2
from Framework.models.transformer_ae import TransformerAutoencoder
from Framework.models.transformer_vae import TransformerVAE
from Framework.models.anomaly_transformer import AnomalyTransformer
from Framework.models.autoencoder_cnn1d import Autoencoder1D
from Framework.models.autoencoder_rnn import RNNAutoencoder
# from Framework.models.LSTM_VAE import TemporalVAE
from Framework.models.DSVDD import RNN_DSVDD, CNN_DSVDD, Transformer_DSVDD
from Framework.postprocessors.postprocessor_functions import plot_data_by_labels, mean_labels_over_epochs
from Framework.postprocessors.tester import Tester, TesterV2
from Framework.postprocessors.graph_worker import get_distribution_plot, get_distribution_plot_for_paper
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.optim.lr_scheduler as lr_scheduler
from torch import nn
import wandb
import datetime

from torch import nn as _nn

def apply_init(m):
    if isinstance(m, (_nn.Linear, _nn.Conv1d, _nn.Conv2d)):
        _nn.init.xavier_uniform_(m.weight, gain=_nn.init.calculate_gain('relu'))
        if m.bias is not None:
            _nn.init.zeros_(m.bias)
    elif isinstance(m, (_nn.GRU, _nn.LSTM)):
        for name, p in m.named_parameters():
            if "weight_ih" in name:
                _nn.init.xavier_uniform_(p)
            elif "weight_hh" in name:
                _nn.init.orthogonal_(p)
            elif "bias" in name:
                _nn.init.zeros_(p)
    elif isinstance(m, _nn.Embedding):
        _nn.init.normal_(m.weight, mean=0.0, std=0.02)

def init_transformer_block(block: _nn.Module):
    for name, p in block.named_parameters():
        if p.dim() >= 2 and ('attn' in name or 'proj' in name or 'qkv' in name):
            _nn.init.orthogonal_(p)
        elif p.dim() >= 2:
            _nn.init.xavier_uniform_(p)
        elif p.dim() == 1:
            _nn.init.zeros_(p)

def train_with_hp_setup(dataloaders, model, batch_size, learning_rate, no_epochs, device, criterion):

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=no_epochs)

    model.to(device)
    criterion.to(device)
    if model.model_name == "DeepSVDD":
        from Framework.loops.dsvdd_loops import train_loop, valid_loop
        model.initialize_center(dataloaders['Train'][0], device)
    elif model.model_name == "anomaly_transformer":
        from Framework.loops.anomaly_tr_loops import train_loop, valid_loop
    else:
        from Framework.loops.loops import train_loop, valid_loop
    # warm-up
    _, _, _ = valid_loop(dataloaders['Valid'][0],
                      model,
                      criterion,
                      device=device)

    train_losses, valid_losses, valid_all_losses = [], [], []
    for epoch in range(no_epochs):
        print(f"Epoch {epoch + 1}\n-------------------------------")

        model.train()
        train_loss = train_loop(dataloaders['Train'][0],
                                model,
                                criterion,
                                optimizer,
                                device=device)
        before_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()  # uncomment if you want to update LR
        after_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch}: lr {before_lr:.8f} -> {after_lr:.8f}")
        if model.model_name == "DeepSVDD" and epoch % 5 == 0:
            print(f"Hypersphere radius R:{model.R}, center c:{model.c}")

        model.eval()
        val_loss, val_all, _ = valid_loop(dataloaders['Valid'][0],
                                                        model,
                                                        criterion,
                                                        device=device)

        if wandb.run is not None:
            wandb.log({"train_loss": train_loss, "val_loss": val_loss, "lr": before_lr, "epoch": epoch})

        train_losses.append(train_loss)
        valid_losses.append(val_loss)
        valid_all_losses.append(np.expand_dims(val_all, axis=0))

    model.eval()
    _, _, train_scores = valid_loop(dataloaders['Train'][0],
                                                 model,
                                                 criterion,
                                                 device=device,
                                                 is_train=True)

    return (
        np.array(train_losses),
        np.array(valid_losses),
        np.vstack(valid_all_losses),
        np.array(train_scores),
    )


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
    dataloaders = get_data_loaders(datasets,args.batch_size)
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
    # model = TransformerAutoencoder(input_dim=62, embed_dim=args.embed_dim, num_heads=args.num_heads, num_layers=args.num_layers, dropout=args.dropout)
    # model = TransformerVAE(input_dim=62, embed_dim=args.embed_dim, num_heads=args.num_heads, num_layers=args.num_layers,
    #                                dropout=args.dropout)
    model = AnomalyTransformer(48, enc_in=62, c_out=62, d_model=args.embed_dim, n_heads=args.num_heads,
                               e_layers=args.num_layers, d_ff=None, dropout=args.dropout, activation='ELU',
                               output_attention=True)
    # model = RNNAutoencoder(62, hidden_dims=[args.embed_dim]+[args.embed_dim//(2*i) for i in range(1,args.num_layers-1)], unit_type=args.cell_type, dropout=args.dropout)
    model.apply(apply_init)
    # options = [64, 32, 16, 8, 4]
    # hidden_dims = options[:args.num_layers]
    # model = LSTMAutoencoder(62, args.hidden_dims, args.dropout)
    # model = Autoencoder1D()

    # model = RNN_DSVDD(input_dim=62, hidden_dim=args.embed_dim, num_layers=args.num_layers, dropout=args.dropout, cell_type=args.cell_type)
    # model = CNN_DSVDD(input_dim=62, out_features=args.out_features, num_channels=args.num_channels, kernel_size=args.kernel_size, dropout=args.dropout)
    # model = Transformer_DSVDD(input_dim=62, out_features=args.out_features, num_heads=args.num_heads, num_layers=args.num_layers, dropout=args.dropout)
    criterion = nn.MSELoss(reduction='none')
    # criterion = VAELoss()

    train_losses, valid_losses, valid_loss_all, train_score = (
        train_with_hp_setup(dataloaders, model, args.batch_size, args.learning_rate, args.epochs, device, criterion))
    valid_metrics =  mean_labels_over_epochs(valid_loss_all)

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
    train_over_epoch = 'train_over_epoch.npy'
    valid_over_epoch = 'valid_over_epoch.npy'
    valid_over_epoch_over_batch_with_labels = 'valid_epochs_labels.npy'
    train_final_score_per_batch = 'train_final_per_batch.npy'

    # Saving train loss over epochs
    path_training_over_epochs = os.path.join(saving_path, train_over_epoch)
    np.save(path_training_over_epochs, train_losses)

    # Saving valid loss over epochs
    path_valid_over_epochs = os.path.join(saving_path, valid_over_epoch)
    np.save(path_valid_over_epochs, valid_losses)

    # Saving sample-wise valid loss over epochs with labels
    path_valid_epochs_labels = os.path.join(saving_path, valid_over_epoch_over_batch_with_labels)
    np.save(path_valid_epochs_labels, valid_loss_all)

    # Saving train loss per sample in last epoch
    path_train_final_per_batch = os.path.join(saving_path, train_final_score_per_batch)
    np.save(path_train_final_per_batch, train_score)

    plt.figure()
    plt.plot(train_losses, label='Train')
    plt.plot(valid_metrics['Class_0'], label='Valid_class_0')
    plt.plot(valid_metrics['Class_1'], label='Valid_class_1')
    fig_path = os.path.join(saving_path, 'fig_1_train_valid_labels.png')
    plt.xlabel("epochs")
    plt.ylabel("RMSE")
    plt.grid()
    plt.legend()
    plt.savefig(fig_path)

    #save model
    model_path = os.path.join(saving_path, 'model.pt')
    torch.save(model.state_dict(), model_path)

    #testing loop
    # tester = Tester(result_folder_path=path['Saving_path'],
    #                 attempt_name=saving_folder_name,
    #                 train_score_over_epoch_file_name=train_over_epoch,
    #                 valid_score_over_epoch_file_name=valid_over_epoch,
    #                 valid_score_over_epoch_per_batch_file_name=valid_over_epoch_over_batch_with_labels,
    #                 train_score_final_file_name=train_final_score_per_batch)

    tester = TesterV2(result_folder_path=path['Saving_path'],
                    attempt_name=saving_folder_name,
                    train_score_over_epoch_file_name=train_losses,
                    valid_score_over_epoch_file_name=valid_losses,
                    valid_score_over_epoch_per_batch_file_name=valid_loss_all,
                    train_score_final_file_name=train_score)

    #valid loop
    valid_scores = tester.estimate_decision_lines()
    print('Validation scores is:')
    print(valid_scores)

    if model.model_name == "DeepSVDD":
        from Framework.loops.dsvdd_loops import test_loop
    elif model.model_name == "anomaly_transformer":
        from Framework.loops.anomaly_tr_loops import test_loop
    else:
        from Framework.loops.loops import test_loop

    #test data loader and loop
    predictions_buffer = []
    performance = []
    metrics_buffer = []
    from copy import deepcopy
    tester_backup = deepcopy(tester)
    for id_dat, single_test_dataset in enumerate(dataloaders['Test']):
        tester = deepcopy(tester_backup)
        testing_loop = lambda class_metric: test_loop(single_test_dataset, model, criterion, class_metric, device=device)
        test_scores, predictions, metrics = tester.test_data(testing_loop=testing_loop)
        print("=============================")
        print(f'Test scores, dataset_id {id_dat}')
        print(f"Dataset path is: {paths_for_datasets['Test'][id_dat]}")
        print(test_scores)
        predictions_buffer.append(predictions)
        performance.append(test_scores)
        metrics_buffer.append(metrics)
        if wandb.run is not None:
            for tester_label, single_score_type_value in test_scores.items():
                wandb.log({f"tester_{tester_label}_type={paths_for_datasets['Test'][id_dat].split('/')[-1]}": single_score_type_value})

    # get decision lines
    decision_lines= []
    for single_tester_name, single_tester in tester.tester_buffer.items():
        decision_lines.append((single_tester_name, single_tester.get_decision_lines()))

    # plot_names = path["Test_folders"]
    plot_names = [f"Scenario {i+1}" for i in range(len(path["Test_folders"])-1)]
    plot_names.append("All scenarios combined")
    fig_distribution, subfigures = get_distribution_plot(valid_loss_all[-1,:,:], predictions_buffer, performance, metrics_buffer, decision_lines, plot_names)

    paper_fig = get_distribution_plot_for_paper(valid_loss_all[-1,:,:], predictions_buffer, performance, metrics_buffer, decision_lines, plot_names)

    graph_valid_test_distribution = os.path.join(saving_path, 'error_distribution.png')
    fig_distribution.savefig(graph_valid_test_distribution)

    paper_fig_path = os.path.join(saving_path, 'paper_distribution.png')
    paper_fig.savefig(paper_fig_path)
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
        "--epochs", type=int, default=75, help="Number of epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32535, help="Batch size"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.00474804, help="Learning rate"
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
        "--dropout", type=float, default=0.34597, help="Learning rate"
    )
    # parser.add_argument(
    #     "--log_interval", type=int, default=1, help="Log interval"
    # )
    parser.add_argument(
        "--embed_dim", type=int, default=32, help="Embedding dimension"
    )
    parser.add_argument(
        "--num_heads", type=int, default=4, help="Multihead attention heads"
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

    # DSVDD specific parameters
    parser.add_argument("--nu", type=float, default=0.01, help="Fraction of outliers for DSVDD")
    parser.add_argument("--num_channels", type=int, default=8, help="Embedding dimension for DSVDD")
    parser.add_argument("--out_features", type=int, default=4, help="Output features for DSVDD")
    parser.add_argument("--kernel_size", type=int, default=3, help="Output features for DSVDD")


    parser.add_argument("--cell_type", type=str, default="GRU", help="Model name for DSVDD")


    args = parser.parse_args()

    run_for = 1
    for _ in range(run_for):

        # os.environ["WANDB_SILENT"] = "true"
        if args.wandb_log:
            wandb.init(
                project="Anomaly_detection",
                entity="OPEN_5G_RAN_team",
                #name="all_50_complex",
                config=vars(parser.parse_args()),
                mode="online",
                # tags=[f"idk whats going on v4"]
            )
        main(paths_config, args)
