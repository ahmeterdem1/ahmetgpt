import numpy as np
import jpype
import jpype.imports
#from jpype.types import *
import os
from typing import List


#os.environ["JAX_PLATFORM_NAME"] = "cpu"  # Decomment for ONLY apply_Svd() on Apple Silicon.
os.environ["JAX_ENABLE_JIT"] = "True"

import jax.numpy as jnp

jpype.startJVM(classpath=['./zemberek-full.jar'])
from zemberek.tokenization import TurkishTokenizer
from zemberek.morphology import TurkishMorphology
from zemberek.core.turkish import PrimaryPos

tokenizer = TurkishTokenizer.DEFAULT
morphology = TurkishMorphology.createWithDefaults()

def tokenize(text: str) -> List[str]:
    """
        Tokenizes the given Turkish text.

        Args:
            text (str): Text, in Turkish language preferably.

        Returns:
            A list of tokens.

        Notes:
            The text may include other languages than Turkish,
            but the used tokenizer is strictly for Turkish.
            Unrecognized words will be left as is.
    """
    analysis = morphology.analyzeAndDisambiguate(text).bestAnalysis()
    results = []
    for single_analysis in analysis:
        if single_analysis.isUnknown():
            # Check if the word is recognized, and it's not punctuation
            original_word = single_analysis.surfaceForm()
            results.append(str(original_word))
        else:
            if single_analysis.getDictionaryItem().primaryPos == PrimaryPos.Punctuation:
                continue  # Skip punctuation tokens
            lemmas = single_analysis.getLemmas()
            stem = lemmas[0] if lemmas else single_analysis.getDictionaryItem().lemma
            results.append(str(stem))
    return results

def create_bulk_text(textfile: str, destfile: str, *args) -> None:

    """
        Creates bulk text file from the original WhatsApp text file.

        Args:
            textfile (str): Path to WhatsApp chat file, in .txt.
            destfile (str): Path to destination file where the output will be stored, in .txt.
            *args: Usernames that may appear in the given WhatsApp chat file. A complete
                list of usernames is crucial for this function. They need to be in the format
                "- username: ".

    """

    with open(textfile, "r") as file:
        text_data = file.read()
        lines = text_data.split("\n")

    total = [lines[0]]

    last_user = ""
    for user in args:
        if user in lines[0]:
            last_user = user

    for line in lines[1:]:
        if last_user in line:
            total[-1] += line
        else:
            for user in args:
                if user in lines[0]:
                    last_user = user
            total.append(line)

    with open(destfile, "w") as file:
        for line in lines:
            file.write(line[16:] + "\n")

def create_token_file(textfile: str, destfile: str) -> None:
    """
        Creates token file from the given bulk text file.

        Args:
            textfile (str): Path to bulk file, in .txt.
            destfile (str): Path to destination file where the output will be stored, in .txt.
    """
    with open(textfile, "r") as file:
        text = file.read()
    tokens = tokenize(text)
    with open(destfile, "w") as file:
        file.write(",".join(tokens))

def create_reduced_token_file(tokenfile: str, destfile: str) -> None:
    """
        Creates reduced token file where duplicates are deleted, from
        the given token file.

        Args:
            textfile (str): Path to token file, in .txt.
            destfile (str): Path to destination file where the output will be stored, in .txt.
    """
    with open(tokenfile, "r") as file:
        full_tokens = file.read().split(",")

    reduced_tokens = list(set(full_tokens))

    with open(destfile, "w") as file:
        file.write(",".join(reduced_tokens))

def create_document_matrix(textfile: str, tokenfile: str, destfile: str) -> None:
    """
        Creates the document matrix, where each message is turned into
        vectors and form the rows of the matrix. Then saves this numpy
        ndarray as .npy file.

        Args:
            textfile (str): The path to the bulk text file that is created
                via create_bulk_text, in .txt.

            tokenfile (str): Path to the reduced token file, in .txt.

            destfile (str): Path to the numpy array file where the matrix
                will be stored, in .npy.

    """
    with open(textfile, "r") as file:
        lines = file.readlines()

    with open(tokenfile, "r") as file:
        tokens = file.read().split(",")

    A = np.array([[line.count(token) for token in tokens] for line in lines])

    factors = np.linalg.norm(A, axis=1, keepdims=True)
    factors[factors == 0] = 1
    A_normalized = A / factors
    print("Dimensions of the resultant matrix: ", A.shape)
    np.save(destfile, A_normalized)

def apply_Svd(arrayfile: str, destfile: str) -> None:
    """
        Applies Singualr Value Decomposition to the document matrix
        created with create_document_matrix(). Then saves U, S and V.T
        matrices in .npz format.

        Args:
            arrayfile (str): Path to the document matrix file, in .npy.
            destfile (str): Path to the decomposed matrix file, in .npz.

        Notes:
            "S" in the name is in capital, because I get import errors
            otherwise...

    """
    A = np.load(arrayfile)
    u, s, v_t = jnp.linalg.svd(A)
    np.savez(destfile, u=np.asarray(u), s=np.asarray(s), v_t=np.asarray(v_t))

def find_message(arrayfile: str, matrixfile: str, documentfile: str, tokenfile: str, query_s: str, resolution: int = 3000) -> None:
    """
        Finds a matching message to the given query. Prints the related information about the
        found message.

        Args:
            arrayfile (str): Path to the decomposed matrix file, in .npz.
            matrixfile (str): Path to the document matrix file, in .npy.
            documentfile (str): Path to bulk file, in .txt.
            tokenfile (str): Path to reduced token file, in .txt.
            query_s (str): Search query.
            resolution (int): Search resolution. Limited by the total reduced token number.
                Defaults to 3000, which should be good enough for almost all use cases.

    """
    arrays = jnp.load(arrayfile)
    A = jnp.load(matrixfile)

    u = arrays["u"]
    s = arrays["s"]
    v_t = arrays["v_t"]

    inverse_sigma_matrix = np.zeros((max(v_t.shape[1], u.shape[0]),))  # A vector
    for i, val in enumerate(s):
        if np.isclose(val, 0):
            inverse_sigma_matrix[i] = 0
        else:
            inverse_sigma_matrix[i] = 1 / val
    del s, v_t

    with open(tokenfile, "r") as file:
        tokens = file.read().split(",")

    query_vector = jnp.array([query_s.count(token) for token in tokens])
    del tokens

    factor = jnp.linalg.norm(query_vector, axis=0, keepdims=True)
    normalized_query = query_vector / factor
    del query_vector, factor

    query = u[:, :resolution] @ jnp.multiply(inverse_sigma_matrix[:resolution], normalized_query[:resolution])
    query_norm = jnp.linalg.norm(query)
    if query_norm == 0:
        query_norm = 1

    cosines = []

    for row in A:  # Already normalized
        doc = u[:, :resolution] @ jnp.multiply(inverse_sigma_matrix[:resolution], row[:resolution])
        divisor = query_norm * jnp.linalg.norm(doc)
        cosines.append(0 if jnp.isclose(divisor, 0) else jnp.dot(query, doc) / divisor)

    del A

    with open(documentfile, "r") as file:
        lines = file.readlines()

    i = jnp.argmax(jnp.asarray(cosines))
    print(f"Closest matching message to the query is: {lines[i]}")
    print(f"At index: {i} with value: {cosines[i]}")

def peek_at_index(textfile: str, index: int):
    with open(textfile, "r") as file:
        lines = file.readlines()

    for k in range(index-10, index+10):
        print(lines[k])


if __name__ == "__main__":
    create_bulk_text("WHATSAPPCHAT.txt",
                     "WHATSAPPCHAT bulk.txt",
                     "- user1: ", "- user2: ")  # Remove the first empty space.

    create_token_file("WHATSAPPCHAT bulk.txt",
                      "WHATSAPPCHAT token.txt")

    create_reduced_token_file("WHATSAPPCHAT token.txt",
                              "WHATSAPPCHAT reduced token.txt")

    create_document_matrix("WHATSAPPCHAT bulk.txt",
                           "WHATSAPPCHAT reduced token.txt",
                           "document_matrix.npy")

    apply_Svd("document_matrix.npy",
              "decomposed.npz")

    find_message("decomposed.npz",
                 "document_matrix.npy",
                 "WHATSAPPCHAT bulk.txt",
                 "WHATSAPPCHAT reduced token.txt",
                 "WHATEVER YOU WANT, in turkish",
                 3000)

    # Decomment below function, put in the found index. It will show you
    # the messages at ±10 position of the found message.

    #peek_at_index("WHATSAPPCHAT bulk.txt", 5841)

