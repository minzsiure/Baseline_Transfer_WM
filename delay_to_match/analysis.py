import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
# path = 'baseline_result/'
path = ''
no_swap = np.load(f'{path}out_hidden_combined_no_swap_stsp.npz')
no_swap_hidden = no_swap['hidden_states']  
no_swap_y = no_swap['labels']

swap = np.load(f'{path}out_hidden_combined_swap_stsp.npz')
swap_hidden = swap['hidden_states']  
swap_y = swap['labels']

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
    "dis_on": 166,
    "dis_off": 183,
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
plt.figure(figsize=(10, 6))
plt.plot(accuracies)
# Adding labels for individual time points
for label, x in time_points.items():
    y_value = np.interp(x, np.arange(len(accuracies)), accuracies)
    plt.text(x, y_value, label, rotation=90, verticalalignment='bottom', color='black')
    plt.axvline(x=x, linestyle='--', label=label)

plt.xlabel('Time')
plt.ylabel('Accuracy')
plt.legend
plt.title('Accuracy over Time')
plt.savefig(f'{path}baseline_stsp.pdf')
