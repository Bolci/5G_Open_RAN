import os.path

import matplotlib.pyplot as plt

from Framework.models.transformer_ae import TransformerAutoencoder
from Framework.postprocessors.postprocesor_general import PostprocessorGeneral
from Framework.utils.utils import load_json_as_dict
from Framework.loops.loops import test_loop, test_loop_general
from Framework.models.autoencoder_cnn import CNNAutoencoderV2
from Framework.models.autoencoder_cnn import CNNAutoencoder, CNNAutoencoderV2, CNNAutoencoderDropout
from Framework.models.AE_CNN_v2 import CNNAEV2, CNNAutoencoderV2
from Framework.models.autoencoder_LSTM import LSTMAutoencoder
from Framework.metrics.metrics import RMSELoss
from Framework.preprocessors.data_preprocessor import DataPreprocessor
from Framework.preprocessors.data_path_worker import get_all_paths
from Framework.preprocessors.data_utils import get_data_loaders, get_datasets
from Framework.postprocessors.tester import Tester
import torch
import matplotlib

font = {
        'size'   : 16}
matplotlib.rc('font', **font)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

config_path = 'data_paths.json'
config = load_json_as_dict(config_path)


result_folder_path = 'Try_Preprocessing=abs_only_multichannel_no-epochs=30_lr=0.0001_bs=16_model=CNN_AE'
train_over_epoch = 'train_over_epoch.txt'
valid_over_epoch = 'valid_over_epoch.txt'
valid_score_over_epoch_per_batch_file_name = 'valid_epochs_labels.txt'
train_final_batch_score = 'train_final_per_batch.txt'


tester = Tester(result_folder_path=config['Saving_path'],
                attempt_name=result_folder_path,
                train_score_over_epoch_file_name=train_over_epoch,
                valid_score_over_epoch_file_name=valid_over_epoch,
                valid_score_over_epoch_per_batch_file_name=valid_score_over_epoch_per_batch_file_name,
                train_score_final_file_name=train_final_batch_score)


path = load_json_as_dict(config_path)
all_paths = get_all_paths(path)
data_preprocessor = DataPreprocessor()
data_preprocessor.set_cache_path(path["Data_cache_path"])
data_preprocessor.set_original_seg(path["True_sequence_path"])

paths_for_datasets = data_preprocessor.preprocess_data(all_paths,
                                                       'abs_only_multichannel',
                                                       rewrite_data=False,
                                                       merge_files=True,
                                                       additional_folder_label='')

# prepare datasets and data_loaders
datasets = get_datasets(paths_for_datasets)
#model = CNNAutoencoderDropout(48).to(device)
model = CNNAutoencoder(48).to(device)
PATH = os.path.join(config['Saving_path'], result_folder_path, 'model.pt')
model.load_state_dict(torch.load(PATH))
model.eval()
test_dataloader = datasets['Test'][0]
criterion = RMSELoss()

#testing_loop = lambda threshold: test_loop(test_dataloader, model, criterion, threshold, device=device)
scores = tester.estimate_decision_lines()
testing_loop = lambda metric: test_loop_general(test_dataloader, model, criterion, metric, device=device)
scores, predictions, metrics = tester.test_data(testing_loop=testing_loop,
                                                figs_label = "test_scores_over_threshold")
print(scores)


'''Testing on the additional dataset'''

config_path_additional_test = 'additional_test_config.json'
config_additional_test = load_json_as_dict(config_path_additional_test)
all_paths = get_all_paths(config_additional_test)
data_preprocessor = DataPreprocessor()
data_preprocessor.set_cache_path(path["Data_cache_path"])
data_preprocessor.set_original_seg(path["True_sequence_path"])

#plt.tight_layout()
#plt.show()
paths_for_datasets = data_preprocessor.preprocess_data(all_paths,
                                                       'abs_only_multichannel',
                                                       rewrite_data=False,
                                                       merge_files=True,
                                                       additional_folder_label='')

datasets = get_datasets(paths_for_datasets)
test_dataloader = datasets['Test'][0]
testing_loop = lambda metric: test_loop_general(test_dataloader, model, criterion, metric, device=device)

scores, predictions, metrics = tester.test_data(testing_loop=testing_loop,
                                                figs_label = "test_scores_over_threshold_test2")
print(scores)