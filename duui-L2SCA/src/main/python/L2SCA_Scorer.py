import stanza
# stanza.download('en')
# stanza.download("de")

# NLP_PIPELINES = {
#     'en': stanza.Pipeline('en'),
#     'de': stanza.Pipeline('de')
# }

def safe_div(n, d):
    return round(n / d, 2) if d else 0.0

def l2sca_metrics(text, nlp, filepath="textfile"):
    # nlp = NLP_PIPELINES.get(lang)
    # if not nlp:
    #     raise ValueError(f"Unsupported language code: {lang}")

    doc = nlp(text)

    W = 0
    S = 0
    VP = 0
    C = 0
    DC = 0
    T = 0
    CT = 0
    CP = 0
    CN = 0
    total_clause_words = 0
    total_tunit_words = 0

    for sent in doc.sentences:
        S += 1
        words = sent.words
        words_no_punct = [w for w in words if w.upos != "PUNCT"]
        W += len(words_no_punct)

        # Verb phrases = count full verbs
        VP += sum(1 for w in words if w.upos == "VERB" and "VerbForm=Fin" in (w.feats or ""))

        # Clause heads
        clause_heads = [w for w in words if w.deprel in ("root", "advcl", "ccomp", "acl", "xcomp") and w.upos == "VERB"]
        C += len(clause_heads)
        total_clause_words += len(words_no_punct) if clause_heads else 0

        # Dependent clauses = clause heads that aren't 'root'
        DC += sum(1 for w in clause_heads if w.deprel != "root")

        # T-units = root + all dependents
        tunit_heads = [w for w in words if w.deprel == "root"]
        for root in tunit_heads:
            T += 1
            has_dc = any(child.head == root.id and child.deprel in ("advcl", "ccomp", "acl", "xcomp") for child in words)
            if has_dc:
                CT += 1
            total_tunit_words += len(words_no_punct)

        # Coordinate phrases
        CP += sum(1 for w in words if w.deprel in ("cc", "conj"))

        # Complex nominals = nouns with modifiers
        for w in words:
            if w.upos == "NOUN":
                has_modifier = any(
                    child.head == w.id and child.deprel in ("amod", "nmod", "compound", "poss", "acl", "appos")
                    for child in words
                )
                if has_modifier:
                    CN += 1

    return {
        # "Filepath": filepath,
        # "W": W,
        # "S": S,
        # "VP": VP,
        # "C": C,
        # "T": T,
        # "DC": DC,
        # "CT": CT,
        # "CP": CP,
        # "CN": CN,

        "MLC": safe_div(W, C),   # Mean length of clause
        "MLS": safe_div(W, S),   # Mean length of sentence
        "MLT": safe_div(W, T),   # Mean length of T-unit

        "C/S": safe_div(C, S),   # Sentence complexity ratio
        "C/T": safe_div(C, T),   # T-unit complexity ratio
        "CT/T": safe_div(CT, T), # Complex T-unit ratio

        "DC/C": safe_div(DC, C), # Dependent clause ratio
        "DC/T": safe_div(DC, T), # Dependent clauses per T-unit

        "CP/C": safe_div(CP, C), # Coordinate phrases per clause
        "CP/T": safe_div(CP, T), # Coordinate phrases per T-unit

        "T/S": safe_div(T, S),   # Sentence coordination ratio

        "CN/C": safe_div(CN, C), # Complex nominals per clause
        "CN/T": safe_div(CN, T), # Complex nominals per T-unit

        "VP/T": safe_div(VP, T), # Verb phrases per T-unit
    }

measures = {
    "MLC": "Mean Length of Clause",
    "MLS": "Mean Length of Sentence",
    "MLT": "Mean Length of T-unit",
    "C/S": "Sentence Complexity Ratio",
    "C/T": "T-unit Complexity Ratio",
    "CT/T": "Complex T-unit Ratio",
    "DC/C": "Dependent Clause Ratio (per Clause)",
    "DC/T": "Dependent Clauses per T-unit",
    "CP/C": "Coordinate Phrases per Clause",
    "CP/T": "Coordinate Phrases per T-unit",
    "T/S": "Sentence Coordination Ratio",
    "CN/C": "Complex Nominals per Clause",
    "CN/T": "Complex Nominals per T-unit",
    "VP/T": "Verb Phrases per T-unit",
}

definitions = {
    "MLC": "# of words / # of clauses",
    "MLS": "# of words / # of sentences",
    "MLT": "# of words / # of T-units",
    "C/S": "# of clauses / # of sentences",
    "C/T": "# of clauses / # of T-units",
    "CT/T": "# of complex T-units / # of T-units",
    "DC/C": "# of dependent clauses / # of clauses",
    "DC/T": "# of dependent clauses / # of T-units",
    "CP/C": "# of coordinate phrases / # of clauses",
    "CP/T": "# of coordinate phrases / # of T-units",
    "T/S": "# of T-units / # of sentences",
    "CN/C": "# of complex nominals / # of clauses",
    "CN/T": "# of complex nominals / # of T-units",
    "VP/T": "# of verb phrases / # of T-units",
}

TypeName = {
    "MLC": "Length of production unit",
    "MLS": "Length of production unit",
    "MLT": "Length of production unit",
    "C/S": "Complexity ratio",
    "C/T": "Subordination",
    "CT/T": "Subordination",
    "DC/C": "Subordination",
    "DC/T": "Subordination",
    "CP/C": "Coordination",
    "CP/T": "Coordination",
    "T/S": "Coordination",
    "CN/C": "Particular structures",
    "CN/T": "Particular structures",
    "VP/T": "Particular structures",
}

TypeNumber = {
    "MLC": 1,
    "MLS": 1,
    "MLT": 1,
    "C/S": 2,
    "C/T": 2,
    "CT/T": 2,
    "DC/C": 2,
    "DC/T": 2,
    "CP/C": 3,
    "CP/T": 3,
    "T/S": 3,
    "CN/C": 4,
    "CN/T": 4,
    "VP/T": 4,
}



if __name__ == '__main__':
    # text = "Das ist ein Beispieltext für die L2SCA Scorer Implementierung, welches auch adjektive, wie schön, groß und interessant enthält. Es ist wichtig, dass der Text eine Vielzahl von Wörtern und Strukturen umfasst, um die Genauigkeit des Scorers zu testen. Dieser Text enthählt viele Klauseln, die für die Analyse nützlich sind. Zum Beispiel: 'Die Katze schläft auf dem Sofa.' oder 'Der Hund bellt laut.' Solche Sätze helfen dabei, die Komplexität und Vielfalt der Sprache zu bewerten. Außerdem sollten auch längere Wörter wie 'Verantwortungsbewusstsein' oder 'Herausforderung' enthalten sein, um die Fähigkeit des Scorers zu testen, komplexe Strukturen zu erkennen."
    # language = "de"
    text_en = f"We use it when a girl in our dorm is acting like a spoiled child."
    text_de = f"Wir benutzen es, wenn ein Mädchen in unserem Wohnheim sich wie ein verwöhntes Kind benimmt."
    language = "de"
    # cache_path = f"/home/bagci/stanza_resources"
    # l2sca_scorer(text_de, language)
    all_metrics = l2sca_metrics(text_en, lang='en')
    print(all_metrics)


