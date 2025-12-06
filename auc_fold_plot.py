from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os
import matplotlib.pyplot as plt
import numpy as np
import imageio
import re

log_dir = 'log/20251113-010710'  # Adjust if the path is different
fold_dirs = [os.path.join(log_dir, d) for d in os.listdir(log_dir) if d.startswith('fold_') and os.path.isdir(os.path.join(log_dir, d))]
fold_dirs.sort(key=lambda x: int(re.search(r'fold_(\d+)', x).group(1)))

all_train_aucs = []
all_val_aucs = []
max_epochs = 0

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
    if 'val_auc' in tags and 'train_auc' in tags:
        val_auc = [s.value for s in ea.Scalars('val_auc')]
        train_auc = [s.value for s in ea.Scalars('train_auc')]
        
        # Ensure lists are the same length
        if len(val_auc) == len(train_auc):
            all_train_aucs.append(train_auc)
            all_val_aucs.append(val_auc)
            max_epochs = max(max_epochs, len(train_auc))
        else:
            print(f'Skipping {event_path}: Mismatched lengths for train and val AUC.')
    else:
        print(f'Skipping {event_path}: Missing required metrics (train_auc, val_auc).')

# Pad shorter lists with NaN to make them equal length for plotting
def pad_list(lst, length):
    return lst + [np.nan] * (length - len(lst))

all_train_aucs = [pad_list(auc, max_epochs) for auc in all_train_aucs]
all_val_aucs = [pad_list(auc, max_epochs) for auc in all_val_aucs]

epochs = np.arange(max_epochs)

# First plot: Training AUC over epochs, multiple lines for folds
plt.figure(figsize=(12, 8))
for i, train_auc in enumerate(all_train_aucs):
    plt.plot(epochs, train_auc, label=f'Fold {i+1}')
plt.xlabel('Epochs')
plt.ylabel('Training AUC')
plt.ylim(0,1)
plt.title('Training AUC over Epochs across Folds')
plt.legend()
plt.grid(True)
plt.savefig('training_auc_over_epochs.png')
plt.close()
print('Saved training_auc_over_epochs.png')

# Second plot: Validation AUC over epochs, multiple lines for folds
plt.figure(figsize=(12, 8))
for i, val_auc in enumerate(all_val_aucs):
    plt.plot(epochs, val_auc, label=f'Fold {i+1}')
plt.xlabel('Epochs')
plt.ylabel('Validation AUC')
plt.ylim(0,1)
plt.title('Validation AUC over Epochs across Folds')
plt.legend()
plt.grid(True)
plt.savefig('validation_auc_over_epochs.png')
plt.close()
print('Saved validation_auc_over_epochs.png')