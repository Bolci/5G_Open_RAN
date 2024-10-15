from Framework.postprocessors.postprocesor import Postprocessor
from Framework.utils.utils import load_json_as_dict


config_path = 'data_paths.json'
config = load_json_as_dict(config_path)

post_processor = Postprocessor()
result_folder_path = 'Try_Preprocessing=abs_only_by_one_sample_no-epochs=50_lr=0.001_bs=32_model=CNN_AE_1'
train_over_epoch = 'train_over_epoch.txt'
valid_final_batch_score = 'valid_epochs_labels.txt'
train_final_batch_score = 'train_final_per_batch.txt'
post_processor.set_paths(result_folder_path=config['Saving_path'],
                         attempt_name=result_folder_path,
                         train_score_paths_file_name=train_over_epoch,
                         valid_score_path_file_name=valid_final_batch_score,
                         train_score_final_file_name = train_final_batch_score)


post_processor.estimate_threshold()


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