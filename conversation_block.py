from zlib import compress, decompress
from exceptions import *
from typing import List
import os
import re

delimiter = "\x00\x11\x22\x33\x44\x55".encode("utf-8")
# The reason that I have chosen above byte-string as the delimiter
# is because, it has very low entropy. This means that, statistically
# speaking, it has a lower chance of appearing in a byte coded file than
# some other text.

# Two delimiters indicate that EOF.

# File extension will be .block

pattern = r'(?=.* - )(.*: )'
date_format = "%d.%m.%Y %H:%M"


def list_subdirectories(base_path):
    """
        This is a helper function that I used in my own production work.
        I leave this here, as it can be useful.

        This function lists subdirectories of a directory. Only the first
        children are listed.
    """
    subdirectories = []
    for dirpath, dirnames, filenames in os.walk(base_path):
        subdirectories.extend([os.path.join(dirpath, d) for d in dirnames])
        dirnames[:] = []
    return subdirectories

def defense(textfile: str, destfile: str, name: str) -> None:
    """
        Divides the given chat into defense conversation blocks,
        and saves it in a .block file.

        Args:
             textfile (str): WhatsApp chat file to be divided, in .txt.
             destfile (str): Destination file for block to be saved, in .block.
             name (str): Your name as it appears in WhatsApp chat, in the format;
                " - username: "
    """
    with open(textfile, "r") as file:
        data = file.read()

    lines = data.split("\n")
    total = [lines[0]]

    found: bool  # This definition is so that, we don't regenerate the same variable again and again.

    for line in lines[1:]:
        found = re.search(pattern, line)

        if found:
            total.append(line)
        else:
            total[-1] += " " + line  # Separator here is not a linebreak, but a space.

    # Ignore the first lines where the first conversator is "self".
    while name in total[0]:
        total.pop(0)

    blocks = []

    index = 0
    line_count = len(total)
    found = False

    while True:
        subblock = []

        if index >= line_count:
            break
        last_index = index
        while True:
            if index >= line_count:
                break
            # Collect others messages
            if name not in total[index]:
                subblock.append(total[index])
                index += 1
            else:
                break

        while True:
            if index >= line_count:
                break
            # Collect self messages
            if name in total[index]:
                subblock.append(total[index])
            else:
                break
            index += 1

        if index == last_index:
            index += 1
            continue
        blocks.append(subblock)

    with open(destfile, "wb") as file:  # Write bytes
        for block in blocks:
            file.write(compress("\n".join(block).encode("utf-8")))
            file.write(delimiter)
        file.write(delimiter)  # Two delimiters sign that the file is ended.

def attack(textfile: str, destfile: str, name: str, *others: List[str]) -> None:
    """
            Divides the given chat into attack conversation blocks,
            and saves it in a .block file.

            Args:
                 textfile (str): WhatsApp chat file to be divided, in .txt.
                 destfile (str): Destination file for block to be saved, in .block.
                 name (str): Your name as it appears in WhatsApp chat, in the format;
                    " - username: "
                others (List[str]): Other peoples name as it appears in WhatsApp chat,
                    in the same format as "name" argument.
        """
    with open(textfile, "r") as file:
        data = file.read()

    lines = data.split("\n")
    users = [*others, name]
    total = [lines[0]]

    found: bool
    blocks = []

    for line in lines[1:]:
        found = False
        for user in users:
            if user in users:
                found = True
                break

        if found:
            total.append(line)
        else:
            total[-1] += " " + line

    for other in others:
        while other in total[0]:
            total.pop(0)

    index = 0
    line_count = len(total)
    while True:
        subblock = []

        if index >= line_count:
            break
        last_index = index

        while True:
            if index >= line_count:
                break
            # Collect self messages
            if name in total[index]:
                subblock.append(total[index])
            else:
                break
            index += 1

        while True:
            if index >= line_count:
                break
            # Collect others messages
            if name not in total[index]:
                subblock.append(total[index])
                index += 1
            else:
                break
        if index == last_index:
            index += 1
            continue
        blocks.append(subblock)

    with open(destfile, "wb") as file:
        for block in blocks:
            file.write(compress("\n".join(block).encode("utf-8")))
            file.write(delimiter)
        file.write(delimiter)

def read_blockfile(blockfile: str) -> List[str]:
    """
        Reads the .block file and returns a list of blocks. It does not
        differentiate between attack and conversation block types.

        Args:
            blockfile (str): .block file to be read.

        Returns:
            The list of read conversation blocks.

        Raises:
            FileCorruptedError: If the file structure is corrupted, this
                error is raised.
    """
    with open(blockfile, "rb") as file:  # Read bytes
        data = file.read()

    if not data.endswith(delimiter + delimiter):
        raise FileCorruptedError()

    data.replace(delimiter + delimiter, "".encode("utf-8"))
    compressed_blocks = data.split(delimiter)
    blocks = []

    for block in compressed_blocks:
        if block:
            blocks.append(str(decompress(block), "utf-8"))

    return blocks
