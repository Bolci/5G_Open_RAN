import os
from Framework.utils.utils import load_txt
import matplotlib.pyplot as plt
from Framework.postprocessors.postprocessor_functions import split_score_by_labels

path = '/Documents/Projekty/5G_OPEN_RAN/Anomaly_detection/5G_Open_RAN/Results/Try_Preprocessing=abs_only_by_one_sample_no-epochs=100_lr=0.0001_bs=32_model=CNN_AE'

train_final_batch_score = 'train_final_per_batch.txt'
full_train_score_path = os.path.join(path, train_final_batch_score)
train_scores = load_txt(full_train_score_path)

valid_final_batch_score = 'valid_epochs_labels.txt'
full_train_score_path = os.path.join(path, valid_final_batch_score)
valid_scores = load_txt(full_train_score_path)[-1]
valid_class_0, valid_class_1 = split_score_by_labels(valid_scores)


plt.figure()
plt.hist(train_scores, bins=15, density=True, alpha=0.5, label='Histogram of Anomaly Scores', color='orange')
plt.title('DCAE PDF Approach: Estimated PDF of Anomaly Scores')
plt.xlabel('Anomaly Score')
plt.ylabel('Density')
plt.legend()
plt.grid(True)


plt.figure()
plt.hist(valid_class_0, bins=15, density=True, alpha=0.5, label='Histogram of Anomaly Scores, valid_class_0', color='blue')
plt.title('DCAE PDF Approach: Estimated PDF of Anomaly Scores')
plt.xlabel('Anomaly Score')
plt.ylabel('Density')
plt.legend()
plt.grid(True)

plt.figure()
plt.hist(valid_class_1, bins=15, density=True, alpha=0.5, label='Histogram of Anomaly Scores,  valid_class_1', color='purple')
plt.title('DCAE PDF Approach: Estimated PDF of Anomaly Scores')
plt.xlabel('Anomaly Score')
plt.ylabel('Density')
plt.legend()
plt.grid(True)




plt.show()