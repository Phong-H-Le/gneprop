from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os
import matplotlib.pyplot as plt
import numpy as np
import imageio
import re

log_dir = 'log/20251112-235245'  # Adjust if the path is different
fold_dirs = [os.path.join(log_dir, d) for d in os.listdir(log_dir) if d.startswith('fold_') and os.path.isdir(os.path.join(log_dir, d))]
fold_dirs.sort(key=lambda x: int(re.search(r'fold_(\d+)', x).group(1)))

plot_files = []    # For GIF
run_index = 0
best_val_aucs = []  # To store best val AUC per fold

for fold_dir in fold_dirs:
    version_dir = os.path.join(fold_dir, 'version_0')
    if not os.path.exists(version_dir):
        print(f'Skipping {fold_dir}: No version_0 folder found.')
        continue
    
    events_files = [os.path.join(version_dir, f) for f in os.listdir(version_dir) if f.startswith('events.out.tfevents')]
    if not events_files:
        print(f'Skipping {fold_dir}: No events files found in version_0.')
        continue
    
    # Load the first events file
    event_path = events_files[0]
    ea = EventAccumulator(event_path)
    ea.Reload()
    
    # Check required scalars
    tags = ea.Tags()['scalars']
    if 'val_auc' in tags and 'train_auc' in tags:
        val_auc = [s.value for s in ea.Scalars('val_auc')]
        train_auc = [s.value for s in ea.Scalars('train_auc')]
        test_auc = [s.value for s in ea.Scalars('test_auc')][0] if 'test_auc' in tags and len(ea.Scalars('test_auc')) > 0 else None
        
        if len(val_auc) == len(train_auc):
            epochs = list(range(len(val_auc)))
            
            # Save best validation AUC for this fold
            best_val_auc_this_fold = max(val_auc)
            best_val_aucs.append(best_val_auc_this_fold)

            # Plotting
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, train_auc, label='Train AUC', marker='o', markersize=4)
            plt.plot(epochs, val_auc, label='Val AUC', marker='x', markersize=6)
            if test_auc is not None:
                plt.axhline(y=test_auc, color='r', linestyle='--', label=f'Test AUC = {test_auc:.4f}')
            plt.xlabel('Epochs')
            plt.ylabel('AUC')
            plt.ylim(0.5, 1)
            plt.title(f'AUC over Epochs (Fold: {os.path.basename(fold_dir)})')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plot_file = f'auc_plot_fold_{run_index}.png'
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            plt.close()
            print(f'AUC plot saved for fold {run_index}: {plot_file} | Best Val AUC: {best_val_auc_this_fold:.4f}')
            plot_files.append(plot_file)
            
            run_index += 1
        else:
            print(f'Skipping {event_path}: Mismatched lengths for train and val AUC.')
    else:
        print(f'Skipping {event_path}: Missing required metrics (train_auc, val_auc).')

# === Print average and std of best validation AUC across folds ===
if best_val_aucs:
    mean_auc = np.mean(best_val_aucs)
    std_auc = np.std(best_val_aucs)
    print("\n" + "="*60)
    print(f"SUMMARY ACROSS {len(best_val_aucs)} FOLDS:")
    print(f"Best Validation AUC per fold: {[f'{x:.4f}' for x in best_val_aucs]}")
    print(f"Average Best Validation AUC: {mean_auc:.4f} ± {std_auc:.4f}")
    print("="*60 + "\n")
else:
    print("No valid folds found to compute AUC statistics.")

# === Generate GIF ===
if plot_files:
    duration_per_frame = 1000  # ms
    gif_path = 'auc_animation.gif'
    with imageio.get_writer(gif_path, mode='I', duration=duration_per_frame, loop=0) as writer:
        for filename in plot_files:
            image = imageio.imread(filename)
            writer.append_data(image)
    print(f'GIF saved: {gif_path}')