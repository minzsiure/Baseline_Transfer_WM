import torch.nn as nn
import torch
from data import generate_data
from model import RNNModel

"""
1. Train RNN on Left Stimuli:
    We first preprocess the left stimuli by reshaping it to include a time-step dimension. 
    The RNN model is then trained on this processed left stimuli. 
    After every 10 epochs, the training loss is printed. 
    At the end of training, we extract the last representation 
    (or feature) from the RNN for the left stimuli.

2. Train Linear Classifier on Left Representations:
    After obtaining the RNN representation of the left stimuli, 
    a linear classifier is trained on these representations. 
    All items in the left stimuli are given a label of 0. 
    This classifier aims to distinguish between left and right stimuli based on these representations.

3. Evaluate Classifier on Left Representations (sanity check):
    After training the classifier, we evaluate its accuracy 
    on the same left representations to verify its training performance.

4. Train RNN on Right Stimuli:
    Similar to the left stimuli, we preprocess the right stimuli to include a time-step dimension. 
    The RNN model is then fine-tuned on this right stimuli. 
    All items in the right stimuli are given a label of 1. 
    
    After every 10 epochs, the training loss is printed along with the 
    classifier's performance on the current RNN representations. 
    
    This evaluation helps us understand how well the RNN's representations of the right stimuli 
    align with the previously trained classifier on left representations.

5. Measure Performance:
    Throughout the training of the RNN on right stimuli, 
    we periodically evaluate the RNN's current representations using the previously trained classifier. 
    This gives insight into how the RNN's understanding of the right stimuli evolves over training, 
    and how similar these representations are to the left stimuli representations.
"""


def calculate_accuracy(pred, true):
    pred_indices = pred.argmax(dim=1)
    correct = (pred_indices == true).float().sum()
    return (correct / len(true)).item()


def test_rnn_accuracy(model, data, labels):
    with torch.no_grad():
        _, outputs = model(data)
        accuracy = calculate_accuracy(outputs, labels)
    return accuracy


def train_model_on_left(model, left_data, right_data, epochs=100, lr=0.001):
    left_data = left_data.unsqueeze(1)
    labels = torch.zeros(left_data.size(
        0), dtype=torch.long).to(left_data.device)

    right_labels = torch.ones(right_data.size(
        0), dtype=torch.long).to(right_data.device)
    right_data = right_data.unsqueeze(1)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        optimizer.zero_grad()
        representations, outputs = model(left_data)
        loss = criterion(outputs, labels)
        loss.backward(retain_graph=True)
        optimizer.step()

        # Check RNN accuracy every 10 epochs
        if (epoch + 1) % 10 == 0:
            left_rnn_accuracy = calculate_accuracy(outputs, labels)

            _, right_outputs = model(right_data)
            right_rnn_accuracy = calculate_accuracy(
                right_outputs, right_labels)

            print(
                f"Left Training Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, RNN Accuracy on Left Data: {left_rnn_accuracy:.4f}, RNN Accuracy on Right Data: {right_rnn_accuracy:.4f}")

    with torch.no_grad():
        rep, _ = model(left_data)
    return rep


def train_classifier(rep, labels, epochs=100, lr=0.001):
    classifier = nn.Linear(rep.size(1), 2).to(rep.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = classifier(rep)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    return classifier


def train_model_on_right(model, classifier, left_data, right_data, epochs=100, lr=0.001):
    left_data = left_data.unsqueeze(1)
    right_data = right_data.unsqueeze(1)
    right_labels = torch.ones(right_data.size(
        0), dtype=torch.long).to(right_data.device)
    left_labels = torch.zeros(right_data.size(
        0), dtype=torch.long).to(right_data.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        optimizer.zero_grad()
        representations, outputs = model(right_data)
        loss = criterion(outputs, right_labels)
        loss.backward(retain_graph=True)
        optimizer.step()

        # Test the representation with the trained classifier every 10 epochs
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                right_rnn_accuracy = calculate_accuracy(outputs, right_labels)
                _, left_outputs = model(left_data)
                left_rnn_accuracy = calculate_accuracy(
                    left_outputs, left_labels)

                classifier_output = classifier(representations)
                # print(classifier_output)
                classifier_loss = criterion(classifier_output, left_labels)
                classifier_accuracy = calculate_accuracy(
                    classifier_output, left_labels)
                print(f"Right Training Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, RNN Accuracy on Right Data: {right_rnn_accuracy:.4f}, RNN Accuracy on Left Data: {left_rnn_accuracy:.4f}, Classifier Accuracy: {classifier_accuracy:.4f}, Classifier Loss: {classifier_loss:.4f}")


left_data, right_data = generate_data()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
left_data = left_data.to(device)
right_data = right_data.to(device)

model = RNNModel(input_dim=40, hidden_dim=40, output_dim=2).to(device)

left_rep = train_model_on_left(model, left_data, right_data)
left_labels = torch.zeros(left_rep.size(0), dtype=torch.long).to(device)
trained_classifier = train_classifier(left_rep, left_labels)
with torch.no_grad():
    classifier_output = trained_classifier(left_rep)
    accuracy = calculate_accuracy(classifier_output, left_labels)
    print(f"Classifier Accuracy on Left Data: {accuracy:.4f}")

train_model_on_right(model, trained_classifier, left_data, right_data)
