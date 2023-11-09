import torch
import torch.nn as nn
import torch.optim as optim
import random
import time
from tqdm import tqdm
import json
import string
from argparse import ArgumentParser
import pickle

unk = '<UNK>'

# Consult the PyTorch documentation for information on the functions used below:
# https://pytorch.org/docs/stable/torch.html
class RNN(nn.Module):
    def __init__(self, input_dim, h):  # Add relevant parameters
        super(RNN, self).__init__()
        self.h = h
        self.numOfLayer = 1
        self.rnn = nn.RNN(input_dim, h, self.numOfLayer, nonlinearity='tanh')
        self.W = nn.Linear(h, 5)
        self.softmax = nn.LogSoftmax(dim=1)
        self.loss = nn.NLLLoss()

    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)

    def forward(self, inputs):
        _, hidden = self.rnn(inputs)
        output = self.W(hidden)
        summed_output = torch.sum(output, dim=0)
        predicted_vector = self.softmax(summed_output)
        return predicted_vector

def load_data(train_data, val_data, test_data):
    with open(train_data) as training_f:
        training = json.load(training_f)
    with open(val_data) as valid_f:
        validation = json.load(valid_f)
    with open(test_data) as test_f:
        test = json.load(test_f)

    tra = []
    val = []
    tst = []
    for elt in training:
        tra.append((elt["text"].split(),int(elt["stars"]-1)))
    for elt in validation:
        val.append((elt["text"].split(),int(elt["stars"]-1)))
    for elt in test:
        tst.append((elt["text"].split(),int(elt["stars"]-1)))
    return tra, val, tst

def main():
    parser = ArgumentParser()
    parser.add_argument("-hd", "--hidden_dim", type=int, required = True, help = "hidden_dim")
    parser.add_argument("-e", "--epochs", type=int, required = True, help = "num of epochs to train")
    parser.add_argument("--train_data", required = True, help = "path to training data")
    parser.add_argument("--val_data", required = True, help = "path to validation data")
    parser.add_argument("--test_data", default = "to fill", help = "path to test data")
    parser.add_argument('--do_train', action='store_true')
    args = parser.parse_args()

    hidden_dim = args.hidden_dim
    epochs = args.epochs
    train_data = args.train_data
    val_data = args.val_data
    test_data = args.test_data

    output_file = "rnn_hd_" + str(hidden_dim) + "_e_" + str(epochs) + ".txt"
    output_write = open('results/' + output_file, 'w')

    print("========== Loading data ==========")
    train_data, valid_data, test_data = load_data(train_data, val_data, test_data) # X_data is a list of pairs (document, y); y in {0,1,2,3,4}

    # Think about the type of function that an RNN describes. To apply it, you will need to convert the text data into vector representations.
    # Further, think about where the vectors will come from. There are 3 reasonable choices:
    # 1) Randomly assign the input to vectors and learn better embeddings during training; see the PyTorch documentation for guidance
    # 2) Assign the input to vectors using pretrained word embeddings. We recommend any of {Word2Vec, GloVe, FastText}. Then, you do not train/update these embeddings.
    # 3) You do the same as 2) but you train (this is called fine-tuning) the pretrained embeddings further.
    # Option 3 will be the most time consuming, so we do not recommend starting with this

    print("========== Vectorizing data ==========")
    word_embedding = pickle.load(open('./word_embedding.pkl', 'rb'))
    print("size of word embedding is:", len(word_embedding))
    model = RNN(input_dim=50, h=hidden_dim)
    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    stopping_condition = False
    epoch = 0

    last_train_accuracy = 0
    last_validation_accuracy = 0

    #while not stopping_condition:
    while epoch < epochs:
        random.shuffle(train_data)
        model.train()
        # You will need further code to operationalize training, ffnn.py may be helpful
        print("Training started for epoch {}".format(epoch + 1))
        train_data = train_data
        correct = 0
        total = 0
        minibatch_size = 16
        N = len(train_data)
        start_time = time.time()
        loss_total = 0
        loss_count = 0
        for minibatch_index in tqdm(range(N // minibatch_size)):
            optimizer.zero_grad()
            loss = None
            for example_index in range(minibatch_size):
                input_words, gold_label = train_data[minibatch_index * minibatch_size + example_index]
                input_words = " ".join(input_words)

                # Remove punctuation
                input_words = input_words.translate(input_words.maketrans("", "", string.punctuation)).split()

                # Look up word embedding dictionary
                vectors = [word_embedding[i.lower()] if i.lower() in word_embedding.keys() else word_embedding['unk'] for i in input_words ]

                # Transform the input into required shape
                vectors = torch.tensor(vectors).view(len(vectors), 1, -1)
                output = model(vectors)

                # Get loss
                example_loss = model.compute_Loss(output.view(1,-1), torch.tensor([gold_label]))

                # Get predicted label
                predicted_label = torch.argmax(output)

                correct += int(predicted_label == gold_label)
                # print(predicted_label, gold_label)
                total += 1
                if loss is None:
                    loss = example_loss
                else:
                    loss += example_loss

            loss = loss / minibatch_size
            loss_total += loss.data
            loss_count += 1
            loss.backward()
            optimizer.step()
        print(loss_total/loss_count)
        print("Training completed for epoch {}".format(epoch + 1))
        output_write.write("Training completed for epoch {}".format(epoch + 1) + "\n")
        print("Training accuracy for epoch {}: {}".format(epoch + 1, correct / total))
        output_write.write("Training accuracy for epoch {}: {}".format(epoch + 1, correct / total) + "\n")
        print("Training time for this epoch: {}".format(time.time() - start_time))
        output_write.write("Training time for this epoch: {}".format(time.time() - start_time) + "\n")
        trainning_accuracy = correct/total

        model.eval()
        correct = 0
        total = 0
        start_time = time.time()
        random.shuffle(valid_data)
        print("Validation started for epoch {}".format(epoch + 1))
        valid_data = valid_data

        for input_words, gold_label in tqdm(valid_data):
            input_words = " ".join(input_words)
            input_words = input_words.translate(input_words.maketrans("", "", string.punctuation)).split()
            vectors = [word_embedding[i.lower()] if i.lower() in word_embedding.keys() else word_embedding['unk'] for i
                       in input_words]

            vectors = torch.tensor(vectors).view(len(vectors), 1, -1)
            output = model(vectors)
            predicted_label = torch.argmax(output)
            correct += int(predicted_label == gold_label)
            total += 1
            # print(predicted_label, gold_label)
        print("Validation completed for epoch {}".format(epoch + 1))
        output_write.write("Validation completed for epoch {}".format(epoch + 1) + "\n")
        print("Validation accuracy for epoch {}: {}".format(epoch + 1, correct / total))
        output_write.write("Validation accuracy for epoch {}: {}".format(epoch + 1, correct / total) + "\n")
        print("Validation time for this epoch: {}".format(time.time() - start_time))
        output_write.write("Validation time for this epoch: {}".format(time.time() - start_time) + "\n")
        validation_accuracy = correct/total

        if validation_accuracy < last_validation_accuracy and trainning_accuracy > last_train_accuracy:
            stopping_condition=True
            print("Training done to avoid overfitting!")
            print("Best validation accuracy is:", last_validation_accuracy)
        else:
            last_validation_accuracy = validation_accuracy
            last_train_accuracy = trainning_accuracy

        epoch += 1

    # Test the model

    # Set the model to evaluation mode
    model.eval()  

    correct = 0
    total = 0

    # Loop through the test data
    for input_words, gold_label in tqdm(test_data):
        input_words = " ".join(input_words)
        input_words = input_words.translate(input_words.maketrans("", "", string.punctuation)).split()
        vectors = [word_embedding[i.lower()] if i.lower() in word_embedding.keys() else word_embedding['unk'] for i
                   in input_words]

        vectors = torch.tensor(vectors).view(len(vectors), 1, -1)
        output = model(vectors)
        predicted_label = torch.argmax(output)
        correct += int(predicted_label == gold_label)
        total += 1

    # Calculate accuracy
    test_accuracy = correct / total  
    print("Test Accuracy: {:.2f}%".format(100 * test_accuracy))
    output_write.write("Test Accuracy: {:.2f}%".format(100 * test_accuracy) + "\n")


if __name__ == "__main__":
    main()

    # You may find it beneficial to keep track of training accuracy or training loss;
    # Think about how to update the model and what this entails. Consider ffnn.py and the PyTorch documentation for guidance
