import tensorflow.keras
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

sentence = ["Câmera incrível. Vale o preço"]

# Tokenização
vocab_size = 10000
oov_tok = "<OOV>"

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(sentence)

# Crie um dicionário chamado word_index
word_index = tokenizer.word_index

# Preenchendo a sequência
sequence = tokenizer.texts_to_sequences(sentence)

padding_type='post'
max_length = 100
trunc_type='post'

padded = pad_sequences(sequence, maxlen=max_length, padding=padding_type, truncating=trunc_type)

# Defina o modelo usando um arquivo .h5
model = tensorflow.keras.models.load_model("Text_Emotion.h5")

# Teste o modelo
result = model.predict(padded)

# Imprima o resultado
predict_class = np.argmax(result,axis = 1)
print(predict_class)

