from conversation_block import read_blockfile
import numpy as np
import re
import tensorflow as tf
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("PATH TO TOKENIZER.json")
print("Tokenizer loaded")

blocks = read_blockfile("DATASET.block")
print("Block file loaded.")

pattern = r'^[0-9]{,2}\.[0-9]{,2}\.[0-9]{4} [0-9]{,2}:[0-9]{,2} - ([^:]*): '

users_messages = {}

vocab_size = tokenizer.get_vocab_size()

for block in blocks:
    for line in block.split("\n"):
        match = re.match(pattern, line)
        #print(match)
        if match is not None:
            username = match.group()
            if username[15] == " ":
                username = username[15:]
            else:
                username = username[16:]
        else:
            continue
        #print(username)
        if line[15] == " ":
            line = line[15:]
        else:
            line = line[16:]
        line = line.replace(username, "")
        id_list = tokenizer.encode(line).ids

        vectorized_line = np.zeros((vocab_size,), dtype=np.int8)  # To lower the required memory in ram.

        for ID in id_list:
            vectorized_line[ID] += 1

        if username is not None and username in users_messages:
            users_messages[username].append(vectorized_line)
        elif username is not None:
            users_messages[username] = [vectorized_line]

del blocks, tokenizer
labels_as_usernames = list(users_messages.keys())

label_username_pair_list = []

summed_vectors = np.zeros((len(labels_as_usernames), vocab_size,), dtype=np.float32)

for i, key in enumerate(labels_as_usernames):
    #temp_list = []
    for message in users_messages[key]:  # message is a numpy.ndarray
        #temp_list.append([*message, i])
        summed_vectors[i] += message
    #temp_list = np.asarray(temp_list)
    print(i)
    np.save(f"persona_{i}.npy", users_messages[key])

del users_messages

#label_username_pair_list = np.asarray(label_username_pair_list)
#print(label_username_pair_list.shape)
print("Sum vector-matrix shape: ", summed_vectors.shape)
print("Vocabulary size: ", vocab_size)
#np.save("labeled_data.npy", label_username_pair_list)
#del label_username_pair_list

factors = np.linalg.norm(summed_vectors, axis=1, keepdims=True)
summed_vectors = summed_vectors / factors
np.save("persona_vectors.npy", summed_vectors)
