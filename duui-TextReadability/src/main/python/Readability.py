import textstat
from diversity import compression_ratio, homogenization_score, ngram_diversity_score
import readability
import syntok.segmenter as segmenter


class ReadabilityMetricsTextStat:
    def __init__(self):
        pass

    def compute_readability(self, texts):
        dict_readability = {}
        dict_readability["Flesch Reading Ease"] = textstat.flesch_reading_ease(texts)
        dict_readability["Flesch-Kincaid Grade Level"] = textstat.flesch_kincaid_grade(texts)
        dict_readability["Smog Index"] = textstat.smog_index(texts)
        dict_readability["Coleman-Liau Index"] = textstat.coleman_liau_index(texts)
        dict_readability["Automated Readability Index"] = textstat.automated_readability_index(texts)
        dict_readability["Dale-Chall Readability Score"] = textstat.dale_chall_readability_score(texts)
        dict_readability["Difficult Words"] = textstat.difficult_words(texts)
        dict_readability["Linsear Write Formula"] = textstat.linsear_write_formula(texts)
        dict_readability["Dale-Chall Readability Score"] = textstat.dale_chall_readability_score(texts)
        dict_readability["Gunning Fog"] = textstat.gunning_fog(texts)
        dict_readability["Text Standard"] = textstat.text_standard(texts, float_output=True)
        dict_readability["Spache Readability"] = textstat.spache_readability(texts)
        dict_readability["McAlpine EFLAW"]= textstat.mcalpine_eflaw(texts)
        dict_readability["Reading Time"] = textstat.reading_time(texts, ms_per_char=14.69)
        dict_readability["Fernandez-Huerta"] = textstat.fernandez_huerta(texts)
        dict_readability["Szigriszt-Pazos (ES)"] = textstat.szigriszt_pazos(texts)
        dict_readability["Gutierrez Polini (ES)"] = textstat.gutierrez_polini(texts)
        dict_readability["Crawford (ES)"] = textstat.crawford(texts)
        dict_readability["Gulpeasse Index (IT)"] = textstat.gulpease_index(texts)
        dict_readability["Osman (AR)"] = textstat.osman(texts)
        dict_readability["Wiener Sachtextformel (DE)"] = textstat.wiener_sachtextformel(texts, 1)
        dict_readability["Syllable Count"] = textstat.syllable_count(texts)
        dict_readability["Lexicon Count"] = textstat.lexicon_count(texts, removepunct=True)
        dict_readability["Sentence Count"] = textstat.sentence_count(texts)
        dict_readability["Character Count"] = textstat.char_count(texts, ignore_spaces=True)
        dict_readability["Letter Count"] = textstat.letter_count(texts)
        dict_readability["Polysyllable Count"] = textstat.polysyllabcount(texts)
        dict_readability["Monosyllable Count"] = textstat.monosyllabcount(texts)

        return dict_readability

class ReadabilityMetricsDiversity:
    def __init__(self):
        pass

    def compute_diversity(self, texts, params):
        dict_diversity = {}

        if params.compression == True:
            dict_diversity["Compression Ratio"] = compression_ratio(texts)


        if params.homogenization == True:
            dict_diversity["Homogenization Score Rougel"] = homogenization_score(texts)

        if params.ngram > 0 :
            dict_diversity["N-gram Diversity Score"] = ngram_diversity_score(texts, params.ngram)  # Example with bigrams

        return dict_diversity



class ReadabilityCompute:
    def __init__(self):
        self.map_infos = {
            "readability grades": "Readability Scores",
            "sentence info": "Sentences Information",
            "word usage": "Word Usage",
            "sentence beginnings": "Sentence Beginnings",
        }
        self.map_units = {
            "Kincaid": "Kincaid",
            "ARI": "Automated Readability Index",
            "Coleman-Liau": "Coleman-Liau Index",
            "FleschReadingEase": "Flesch Reading Ease",
            "GunningFogIndex": "Gunning Fog Index",
            "LIX": "LIX",
            "SMOGIndex": "SMOG Index",
            "RIX": "RIX",
            "DaleChallIndex": "Dale-Chall Index",
            "characters_per_word": "Characters per Word",
            "syll_per_word": "Syllables per Word",
            "words_per_sentence": "Words per Sentence",
            "sentences_per_paragraph": "Sentences per Paragraph",
            "type_token_ratio": "Type-Token Ratio",
            "directspeech_ratio": "Direct Speech Ratio",
            "characters": "Characters",
            "syllables": "Syllables",
            "words": "Words",
            "sentences": "Sentences",
            "paragraphs": "Paragraphs",
            "long_words": "Long Words",
            "complex_words": "Complex Words",
            "complex_words_dc": "Complex Words (Dale-Chall)",
            "tobeverb": "To Be Verbs",
            "auxverb": "Auxiliary Verbs",
            "conjunction": "Conjunctions",
            "pronoun": "Pronouns",
            "nominalization": "Nominalizations",
            "preposition": "Prepositions",
            "interrogative": "Interrogatives",
            "article": "Articles",
            "subordination": "Subordinations",
            "wordtypes": "Word Types",
        }

    def segment_text(self, text):
        """
        Segment the text into sentences and paragraphs.
        """
        return '\n\n'.join(
            '\n'.join(' '.join(token.value for token in sentence)
                      for sentence in paragraph)
            for paragraph in segmenter.analyze(text))

    def get_readability(self, text, lang='en'):
        """
        Get readability measures for the given text.
        """
        tokenized_text = self.segment_text(text)
        readability_scores = readability.getmeasures(tokenized_text, lang=lang)
        dict_output = {
            "Readability Scores": {},
            "Sentences Information": {},
            "Word Usage": {},
            "Sentence Beginnings": {},
        }
        for key, value in readability_scores.items():
            map_info = self.map_infos[key]
            for sub_key, sub_value in value.items():
                dict_output[map_info][self.map_units[sub_key]] = sub_value
        return dict_output

if __name__ == '__main__':
    readability_metrics = ReadabilityMetricsTextStat()
    sample_text = "This is a sample text to compute readability metrics. It is designed to be simple yet effective for testing purposes. Another sentence to ensure we have enough content for analysis."
    results = readability_metrics.compute_readability(sample_text)
    for metric, value in results.items():
        print(f"{metric}: {value}")
    diversity_metrics = ReadabilityMetricsDiversity()
    diversity_results = diversity_metrics.compute_diversity(sample_text)
    for metric, value in diversity_results.items():
        print(f"{metric}: {value}")
