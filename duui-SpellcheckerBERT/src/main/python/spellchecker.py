from symspellpy import SymSpell, Verbosity
from typing import List


def spellchecker(sentence: List[str], begin: List[str], end: List[str], speller: SymSpell, lower_case: bool):
    spellchecked = []
    for c, token in enumerate(sentence):
        if token.isalnum() and not token.isdigit():
            if lower_case:
                word = token.lower()
            else:
                word = token
            suggestions = speller.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)
            if suggestions:
                if suggestions[0].term == word:
                    spellchecked.append({"spellout": "right",
                                         "token": token,
                                         "suggestion": str(suggestions[0].term),
                                         "index": c,
                                         "begin": begin[c],
                                         "end": end[c]
                                         })
                else:
                    spellchecked.append({
                        "spellout": "wrong",
                        "token": token,
                        "suggestion": str(suggestions[0].term),
                        "index": c,
                        "begin": begin[c],
                        "end": end[c]
                    })
            else:
                spellchecked.append({
                    "spellout": "unknown",
                    "token": token,
                    "suggestion": "",
                    "index": c,
                    "begin": begin[c],
                    "end": end[c]
                })
        else:
            spellchecked.append({
                "spellout": "skipped",
                "token": token,
                "suggestion": "",
                "index": c,
                "begin": begin[c],
                "end": end[c]
            })

    return spellchecked


if __name__ == '__main__':
    try:
        sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        dictionary_path = "de-100k.txt"
        sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
    except Exception as ex:
        print(f"de-100k.txt, Error: {ex}")
        exit()
    sentence = ["Ich", "habe", "im", "Landtag3", "L34t3", "angesprochen", ",", "die", "ich", "für", "Int3ll3g3nt", "halte", "!"]
    begin = [0, 4, 9, 12, 21, 27, 40, 42, 46, 50, 54, 66, 72]
    end = [3, 8, 11, 20, 26, 39, 41, 45, 49, 53, 65, 71, 73]
    print(spellchecker(sentence, begin, end, sym_spell, lower_case=True))