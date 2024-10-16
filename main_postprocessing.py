import os.path

import matplotlib.pyplot as plt

from Framework.postprocessors.postprocesor import Postprocessor
from Framework.utils.utils import load_json_as_dict
from Framework.loops.loops import test_loop
from Framework.Model_bank.autoencoder_cnn import CNNAutoencoderV2
from Framework.metrics.metrics import RMSELoss
from Framework.preprocessors.data_preprocessor import DataPreprocessor
from Framework.preprocessors.data_path_worker import get_all_paths
from Framework.preprocessors.data_utils import get_data_loaders, get_datasets
import torch


device = 'cuda' if torch.cuda.is_available() else 'cpu'

config_path = 'data_paths.json'
config = load_json_as_dict(config_path)

post_processor = Postprocessor()
result_folder_path = 'Try_Preprocessing=abs_only_by_one_sample_no-epochs=50_lr=0.001_bs=32_model=CNN_AE_1'
train_over_epoch = 'train_over_epoch.txt'
valid_over_epoch = 'valid_over_epoch.txt'
valid_score_over_epoch_per_batch_file_name = 'valid_epochs_labels.txt'
train_final_batch_score = 'train_final_per_batch.txt'
post_processor.set_paths(result_folder_path=config['Saving_path'],
                         attempt_name=result_folder_path,
                         train_score_over_epoch_file_name=train_over_epoch,
                         valid_score_over_epoch_file_name=valid_over_epoch,
                         valid_score_over_epoch_per_batch_file_name=valid_score_over_epoch_per_batch_file_name,
                         train_score_final_file_name=train_final_batch_score)

threshold, classification_score, _ = post_processor.estimate_threshold_on_valid_data()


path = load_json_as_dict('./data_paths.json')
all_paths = get_all_paths(path)
data_preprocessor = DataPreprocessor()
data_preprocessor.set_cache_path(path["Data_cache_path"])
data_preprocessor.set_original_seg(path["True_sequence_path"])

paths_for_datasets = data_preprocessor.preprocess_data(all_paths,
                                                       'abs_only_by_one_sample',
                                                       rewrite_data=False,
                                                       merge_files=True)
# prepare datasets and data_loaders
datasets = get_datasets(paths_for_datasets)
model = CNNAutoencoderV2().to(device)
PATH = os.path.join(config['Saving_path'], result_folder_path, 'model.pt')
model.load_state_dict(torch.load(PATH))
model.eval()
test_dataloader = datasets['Test'][0]
criterion = RMSELoss()

testing_loop = lambda : test_loop(test_dataloader, model, criterion, threshold, device=device)
classification_score_test_0, classification_score_test_1, predicted_results, fig = post_processor.test_data(threshold,
                                                                                                            testing_loop)
print(f"Classification score valid {classification_score}, classification score test = {classification_score_test_0, classification_score_test_1}")

'''
plt.figure()
plt.hist(train_scores, bins=15, density=True, alpha=0.5, label='Histogram of Anomaly Scores', color='orange')
plt.title('DCAE PDF Approach: Estimated PDF of Anomaly Scores')
plt.xlabel('Anomaly Score')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
'''

'''
plt.figure()
ax1.hist(valid_class_0[:, 1].tolist(), bins=15, density=True, alpha=0.5, label='Histogram of Anomaly Scores, valid_class_0', color='orange')
plt.title('DCAE PDF Approach: Estimated PDF of Anomaly Scores')
plt.xlabel('Anomaly Score')
plt.ylabel('Density')
plt.legend()
plt.grid(True)

plt.figure()
ax2.hist(valid_class_1[:, 1].tolist(), bins=15, density=True, alpha=0.5, label='Histogram of Anomaly Scores,  valid_class_1', color='purple')
plt.title('DCAE PDF Approach: Estimated PDF of Anomaly Scores')
plt.xlabel('Anomaly Score')
plt.ylabel('Density')
plt.legend()
plt.grid(True)


plt.show()'''