# from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
# import os
# import matplotlib.pyplot as plt
# import numpy as np
# import re

# log_dir = 'log/20251113-010710'  # Adjust if the path is different
# fold_dirs = [os.path.join(log_dir, d) for d in os.listdir(log_dir) if d.startswith('fold_') and os.path.isdir(os.path.join(log_dir, d))]
# fold_dirs.sort(key=lambda x: int(re.search(r'fold_(\d+)', x).group(1)))

# all_train_aps = []
# all_val_aps = []
# max_epochs = 0

# for fold_ix, fold_dir in enumerate(fold_dirs):
#     version_dir = os.path.join(fold_dir, 'version_0')
#     if not os.path.exists(version_dir):
#         print(f'Skipping {fold_dir}: No version_0 folder found.')
#         continue
    
#     events_files = [os.path.join(version_dir, f) for f in os.listdir(version_dir) if f.startswith('events.out.tfevents')]
#     if not events_files:
#         print(f'Skipping {fold_dir}: No events files found in version_0.')
#         continue
    
#     # Assuming one events file per version_0; load the first
#     event_path = events_files[0]
#     ea = EventAccumulator(event_path)
#     ea.Reload()
    
#     # Check if required scalars are present
#     tags = ea.Tags()['scalars']
#     if 'val_ap' in tags and 'train_ap' in tags:
#         val_ap = [s.value for s in ea.Scalars('val_ap')]
#         train_ap = [s.value for s in ea.Scalars('train_ap')]
        
#         # Ensure lists are the same length
#         if len(val_ap) == len(train_ap):
#             all_train_aps.append(train_ap)
#             all_val_aps.append(val_ap)
#             max_epochs = max(max_epochs, len(train_ap))
#         else:
#             print(f'Skipping {event_path}: Mismatched lengths for train and val AP.')
#     else:
#         print(f'Skipping {event_path}: Missing required metrics (train_ap, val_ap).')

# # Pad shorter lists with NaN to make them equal length for plotting
# def pad_list(lst, length):
#     return lst + [np.nan] * (length - len(lst))

# all_train_aps = [pad_list(ap, max_epochs) for ap in all_train_aps]
# all_val_aps = [pad_list(ap, max_epochs) for ap in all_val_aps]

# epochs = np.arange(max_epochs)

# # First plot: Training PR AUC over epochs, multiple lines for folds
# plt.figure(figsize=(12, 8))
# for i, train_ap in enumerate(all_train_aps):
#     plt.plot(epochs, train_ap, label=f'Fold {i+1}')
# plt.xlabel('Epochs')
# plt.ylabel('Training PR AUC')
# plt.title('Training PR AUC over Epochs across Folds')
# plt.legend()
# plt.grid(True)
# plt.savefig('training_pr_auc_over_epochs.png')
# plt.close()
# print('Saved training_pr_auc_over_epochs.png')

# # Second plot: Validation PR AUC over epochs, multiple lines for folds
# plt.figure(figsize=(12, 8))
# for i, val_ap in enumerate(all_val_aps):
#     plt.plot(epochs, val_ap, label=f'Fold {i+1}')
# plt.xlabel('Epochs')
# plt.ylabel('Validation PR AUC')
# plt.title('Validation PR AUC over Epochs across Folds')
# plt.legend()
# plt.grid(True)
# plt.savefig('validation_pr_auc_over_epochs.png')
# plt.close()
# print('Saved validation_pr_auc_over_epochs.png')

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os
import numpy as np
import re

log_dir = 'log/20251113-010710'  # Adjust if the path is different
fold_dirs = [os.path.join(log_dir, d) for d in os.listdir(log_dir) if d.startswith('fold_') and os.path.isdir(os.path.join(log_dir, d))]
fold_dirs.sort(key=lambda x: int(re.search(r'fold_(\d+)', x).group(1)))

val_recalls_at_best = []

for fold_ix, fold_dir in enumerate(fold_dirs):
    version_dir = os.path.join(fold_dir, 'version_0')
    if not os.path.exists(version_dir):
        print(f'Skipping {fold_dir}: No version_0 folder found.')
        continue
    
    events_files = [os.path.join(version_dir, f) for f in os.listdir(version_dir) if f.startswith('events.out.tfevents')]
    if not events_files:
        print(f'Skipping {fold_dir}: No events files found in version_0.')
        continue
    
    # Assuming one events file per version_0; load the first
    event_path = events_files[0]
    ea = EventAccumulator(event_path)
    ea.Reload()
    
    # Check if required scalars are present
    tags = ea.Tags()['scalars']
    if 'val_auc' in tags and 'val_recall' in tags:
        val_auc = [s.value for s in ea.Scalars('val_auc')]
        val_recall = [s.value for s in ea.Scalars('val_recall')]
        
        if len(val_auc) == len(val_recall):
            best_epoch_ix = np.argmax(val_auc)
            val_recall_at_best = val_recall[best_epoch_ix]
            val_recalls_at_best.append(val_recall_at_best)
    
if val_recalls_at_best:
    average_val_recall = np.mean(val_recalls_at_best)
    std_val_recall = np.std(val_recalls_at_best)
    print(f'Average validation recall over folds (at best AUC epoch): {average_val_recall:.4f}')
    print(f'Standard deviation of validation recall over folds (at best AUC epoch): {std_val_recall:.4f}')
else:
    print('No data found for computing average validation recall.')