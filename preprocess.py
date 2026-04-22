import string

def clean_text(text):
    text = text.lower()

    for c in string.punctuation:
        text = text.replace(c, "")

    tokens = text.split()

    return " ".join(tokens)