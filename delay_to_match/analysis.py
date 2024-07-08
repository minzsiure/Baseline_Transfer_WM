import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import h5py

rnn_type = 'vRNN' #rnn was 91% and 70% test accuracy; stsp 75% and 58.87%
plot_full = True
# path = 'baseline_result/'
path = f'swap_results/{rnn_type}/'
no_swap = np.load(f'{path}out_hidden_combined_no_swap_{rnn_type}.npz')
no_swap_hidden = no_swap['hidden_states']  
no_swap_y = no_swap['labels']

swap = np.load(f'{path}out_hidden_combined_swap_{rnn_type}.npz')
swap_hidden = swap['hidden_states']  
swap_y = swap['labels']
# with h5py.File(f'{path}out_hidden_combined_no_swap_{rnn_type}.h5', 'r') as hf:
#     no_swap_hidden = hf['hidden_states'][:]
#     no_swap_y = hf['labels'][:]

# with h5py.File(f'{path}out_hidden_combined_swap_{rnn_type}.h5', 'r') as hf:
#     swap_hidden = hf['hidden_states'][:]
#     swap_y = hf['labels'][:]
    
print(no_swap_hidden.shape, swap_hidden.shape)
print(no_swap_y.shape, swap_y.shape)

# Train classifier on no_swap data
clf = LogisticRegression(max_iter=1000)
clf.fit(no_swap_hidden, no_swap_y)

# Test classifier on swap data and calculate accuracies
accuracies = []

time_points = {
    "samp_on": 66,
    "samp_off": 100,
    # "dis_on": 166,
    # "dis_off": 183,
    "test_on": 233,
    "test_off": 266
}

intervals = {
    "samp": (66, 100),
    "dis": (166, 183),
    "test": (233, 266)
}


for i in range(swap_hidden.shape[1]):  # Iterate over time steps
    swap_hidden_at_time_i = swap_hidden[:, i, :]
    predictions = clf.predict(swap_hidden_at_time_i)
    acc = accuracy_score(swap_y, predictions)
    accuracies.append(acc)


# Plotting
plt.figure(figsize=(13, 4.5))
if plot_full:
    plt.plot(accuracies)
else:
    plt.plot([i for i in range(time_points['test_on'], len(accuracies))], accuracies[time_points['test_on']:])
# Adding labels for individual time points
for label, x in time_points.items():
    y_value = np.interp(x, np.arange(len(accuracies)), accuracies)
    plt.text(x, y_value, label, rotation=90, verticalalignment='bottom', color='black')
    plt.axvline(x=x, linestyle='--')

plt.xlabel('Time')
plt.ylabel('Decoder accuracy')
# plt.xlim(1, len(accuracies[time_points['test_on']:]))
if not plot_full:
    plt.xlim(time_points['test_on'], len(accuracies))
nclasses = 4
chance_accuracy = 1 / nclasses
plt.axhline(y=chance_accuracy, color='gray', linestyle='--', label=f'Chance Level (1/{nclasses})')
plt.subplots_adjust(bottom=0.2)
plt.legend()
# plt.title('Accuracy over Time')
if not plot_full:
    plt.savefig(f'{path}baseline_{rnn_type}_paper.pdf')
else:
    plt.savefig(f'{path}baseline_{rnn_type}.pdf')
