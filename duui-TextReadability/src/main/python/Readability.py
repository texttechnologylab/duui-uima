import textstat
from diversity import compression_ratio, homogenization_score, ngram_diversity_score


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
