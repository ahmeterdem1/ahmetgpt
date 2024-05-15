import numpy as np
import re
from tokenizers import Tokenizer
from conversation_block import read_blockfile

tokenizer = Tokenizer.from_file("TOKENIZER.json")  # Make sure no [UNK] tokens
print("Tokenizer loaded")

blocks = read_blockfile("BLOCKFILE.block")
print("Block file loaded.")

pattern = r'^[0-9]{,2}\.[0-9]{,2}\.[0-9]{4} [0-9]{,2}:[0-9]{,2} - ([^:]*): '

vocab_size = tokenizer.get_vocab_size()
START = vocab_size
STOP = vocab_size + 1
vocab_size += 2  # <start> and <stop> tokens
users_messages = {}

for block in blocks:
    for line in block.split("\n"):
        match = re.match(pattern, line)
        if match is not None:
            username = match.group()
            if username[15] == " ":
                username = username[15:]
            else:
                username = username[16:]
        else:
            continue
        if line[15] == " ":
            line = line[15:]
        else:
            line = line[16:]
        line = line.replace(username, "")
        id_list = tokenizer.encode(line).ids

        if username is not None and username in users_messages:
            users_messages[username].append(id_list)
        elif username is not None:
            users_messages[username] = [id_list]

print("Messages processed.")

count = 0
for user in users_messages:
    grams = np.zeros((vocab_size, vocab_size,), dtype=np.uint16)
    for id_list in users_messages[user]:
        grams[START, id_list[0]] += 1
        for i in range(1, len(id_list)):
            grams[id_list[i - 1], id_list[i]] += 1
        grams[id_list[-1], STOP] += 1
    users_messages[user].clear()  # Save ram
    np.savez(f"2-gram-{count}.npz", gram=grams, compress=9)
    del grams
    count += 1
    print(f"User {count} saved.")
