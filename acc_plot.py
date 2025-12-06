from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os
import matplotlib.pyplot as plt
import numpy as np
import imageio
import re

def compute_balanced_acc(acc, prec, recall):
    if prec == 0 or acc == 1:  # Edge cases to avoid division by zero
        return recall / 2 if prec == 0 else (recall + 1) / 2
    fp_over_p = recall * (1 / prec - 1)
    r = -(recall - fp_over_p - acc) / (1 - acc) if (1 - acc) != 0 else 0
    spec = 1 - (fp_over_p / r) if r != 0 else 1
    return (recall + spec) / 2

log_dir = 'log/20251112-235245'
fold_dirs = [os.path.join(log_dir, d) for d in os.listdir(log_dir)
             if d.startswith('fold_') and os.path.isdir(os.path.join(log_dir, d))]
fold_dirs.sort(key=lambda x: int(re.search(r'fold_(\d+)', x).group(1)))

plot_files = []
run_index = 0
fold_balanced_accs = []  # <-- store final/best balanced accuracy per fold

for fold_dir in fold_dirs:
    version_dir = os.path.join(fold_dir, 'version_0')
    if not os.path.exists(version_dir):
        print(f'Skipping {fold_dir}: No version_0 folder found.')
        continue

    events_files = [os.path.join(version_dir, f)
                    for f in os.listdir(version_dir)
                    if f.startswith('events.out.tfevents')]
    if not events_files:
        print(f'Skipping {fold_dir}: No events files found in version_0.')
        continue

    event_path = events_files[0]
    ea = EventAccumulator(event_path)
    ea.Reload()

    tags = ea.Tags()['scalars']
    required_val_tags = ['val_acc', 'val_precision', 'val_recall']
    if all(tag in tags for tag in required_val_tags):
        val_acc = [s.value for s in ea.Scalars('val_acc')]
        val_prec = [s.value for s in ea.Scalars('val_precision')]
        val_recall = [s.value for s in ea.Scalars('val_recall')]

        test_acc = [s.value for s in ea.Scalars('test_acc')][0] if 'test_acc' in tags and len(ea.Scalars('test_acc')) > 0 else None
        test_prec = [s.value for s in ea.Scalars('test_precision')][0] if 'test_precision' in tags and len(ea.Scalars('test_precision')) > 0 else None
        test_recall = [s.value for s in ea.Scalars('test_recall')][0] if 'test_recall' in tags and len(ea.Scalars('test_recall')) > 0 else None

        if len(val_acc) == len(val_prec) == len(val_recall):
            epochs = list(range(len(val_acc)))
            val_balanced_accs = [compute_balanced_acc(a, p, r)
                                 for a, p, r in zip(val_acc, val_prec, val_recall)]

            # Save final (or best) validation balanced accuracy
            final_balanced_acc = val_balanced_accs[-1]
            best_balanced_acc = max(val_balanced_accs)
            fold_balanced_accs.append(best_balanced_acc)  # or final_balanced_acc

            test_balanced_acc = None
            if test_acc is not None and test_prec is not None and test_recall is not None:
                test_balanced_acc = compute_balanced_acc(test_acc, test_prec, test_recall)

            plt.figure()
            plt.plot(epochs, val_balanced_accs, label='Val Balanced Accuracy', marker='o')
            plt.plot(epochs, val_acc, label='Val Overall Accuracy', marker='x', linestyle='--')
            if test_balanced_acc is not None:
                plt.axhline(y=test_balanced_acc, color='g', linestyle='--', label='Test Balanced Accuracy')
            if test_acc is not None:
                plt.axhline(y=test_acc, color='r', linestyle='--', label='Test Overall Accuracy')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.ylim(0,1)
            plt.title(f'Balanced vs Overall Accuracy over Epochs (Fold: {os.path.basename(fold_dir)})')
            plt.legend()
            plt.grid(True)
            plot_file = f'balanced_acc_plot_fold_{run_index}.png'
            plt.savefig(plot_file)
            plt.close()
            print(f'Balanced Acc plot saved for fold {run_index}: {plot_file}')
            plot_files.append(plot_file)

            run_index += 1
        else:
            print(f'Skipping {event_path}: Mismatched lengths for val metrics.')
    else:
        print(f'Skipping {event_path}: Missing required val metrics (val_acc, val_precision, val_recall).')

# --- Aggregate stats across folds ---
if fold_balanced_accs:
    mean_bal_acc = np.mean(fold_balanced_accs)
    std_bal_acc = np.std(fold_balanced_accs)
    print("\n=== Aggregate Balanced Accuracy Across Folds ===")
    print(f"Mean Balanced Accuracy: {mean_bal_acc:.4f}")
    print(f"Std Dev: {std_bal_acc:.4f}")
    print(f"Per Fold: {[round(x,4) for x in fold_balanced_accs]}")

if plot_files:
    duration_per_frame = 1000
    with imageio.get_writer('balanced_acc_animation.gif', mode='I', duration=duration_per_frame, loop=0) as writer:
        for filename in plot_files:
            image = imageio.imread(filename)
            writer.append_data(image)
    print('GIF saved: balanced_acc_animation.gif')
