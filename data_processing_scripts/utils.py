def select_filenames_by_key_word(files: list, key_word: str):
    return [x for x in files if key_word in x]

def select_filenames_by_key_word_neg(files: list, key_word: str):
    return [x for x in files if key_word not in x]