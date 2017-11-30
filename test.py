import numpy as np

#Random seed for reproducibility
#This must be done before keras is imported!
rseed = 42
np.random.seed(rseed)

#Testsplit (percentage to take as test set)
testset = 0.2

from sklearn.model_selection import train_test_split
from sklearn import metrics

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.regularizers import l2
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Embedding, LSTM, Dropout, Masking
from keras.utils import plot_model

#
# Loading data
#
X_raw = []
y_raw = []
print('Loading data...')
with open("imdb_labelled.txt", 'r') as file:
    for line in file.readlines():
        contents = line.split("\t")
        if (len(contents) != 2): break;
        X_raw.append(contents[0]) # Raw sentence string
        y_raw.append(contents[1][:-1]) # Label as strings (without newline)
y_raw = np.array(y_raw).astype(np.int)

print(len(X_raw), 'train sequences')
print(len(y_raw), 'test sequences')

#
# Tokenizing
#
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_raw)
X_num = tokenizer.texts_to_sequences(X_raw)
X_voc_size = max([max(x) for x in X_num]) + 1   # Max integer in matrix + 1 (Zeroth element)
print("Vocabulary size: %i" % X_voc_size)

#
# Sequenzing
#
X_len = [len(x) for x in X_num]
X_len_max = max(X_len)
print("Max sentence length: %i (THIS MUST BE AT MOST 100)" % X_len_max)
if (X_len_max > 100):
    exit(1)

X_seq = pad_sequences(X_num, maxlen=100, value=0)
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_raw, test_size=testset, random_state=rseed)

#
# Creating the model
#
main_input = Input(shape=(100,), dtype='int32', name='main_input')  #Create input for sequence
embedding = Embedding(output_dim=30, input_dim=X_voc_size,          #Create embedding layer with 2D output
                    input_length=100, name="layer1")(main_input) 
masking = Masking(mask_value=0)(embedding)
lstm_out = LSTM(100, kernel_regularizer=l2(0.05),                   #And pass this output to the LSTM
                    recurrent_regularizer=l2(0.05),
                    bias_regularizer=l2(0.05),
                    name="layer2")(masking)
dense = Dense(50, kernel_regularizer=l2(0.05),                   #And pass values to a dense layer
                  bias_regularizer=l2(0.05),
                  name="layer3")(lstm_out)
main_output = Dense(1, activation='sigmoid', name='main_output')(dense) #And a output node

model = Model(inputs=[main_input], outputs=[main_output])
model.compile(optimizer='adam', loss='binary_crossentropy', 
              metrics=['accuracy'])

plot_model(model, show_shapes=True, show_layer_names=False)

#
# Training and evalutation
#
model.fit(X_train, y_train, validation_split=0.2,
          verbose=1, epochs=30, batch_size=32, shuffle=False)


p_train = np.around(model.predict(X_train).flatten()).astype(np.int)
p_test = np.around(model.predict(X_test).flatten()).astype(np.int)

# Advanced performance analzsis
print("\n\nTraining data:")
print(metrics.classification_report(y_train, p_train, target_names=["NEG", "POS"]))
print("\n\nTest data:")
print(metrics.classification_report(y_test, p_test, target_names=["NEG", "POS"]))

print("DONE! You may insert your own sentence now.")
while True:
    try:
        O_raw = [input("> ")]
        O_num = tokenizer.texts_to_sequences(O_raw)
        O_len = len(O_num[0])
        if (O_len > 100):
            print("Input too long!")
            continue
        O_seq = pad_sequences(O_num, maxlen=100, value=0)
        O_hyp = model.predict(O_seq).flatten()[0]
        print("Your input is a %05.2f%% match with label 'POSITIVE'" % O_hyp)
    except KeyboardInterrupt:
        exit(0)