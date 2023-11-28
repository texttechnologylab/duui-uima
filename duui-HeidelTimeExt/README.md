[![version](https://img.shields.io/github/license/texttechnologylab/heideltime)]()
[![paper](https://img.shields.io/badge/paper-ACL--anthology-B31B1B.svg)](http://www.lrec-conf.org/proceedings/lrec2022/pdf/2022.lrec-1.505.pdf)

### How to Cite

 A. Lücking, M. Stoeckel, G. Abrami, and A. Mehler, "I still have Time(s): Extending HeidelTime for German Texts," in *Proceedings of the Language Resources and Evaluation Conference*, Marseille, France, 2022, pp. 4723-4728. 
![[PDF]](http://www.lrec-conf.org/proceedings/lrec2022/pdf/2022.lrec-1.505.pdf)

### BibTex
```
@InProceedings{Luecking:Stoeckel:Abrami:Mehler:2022,
  Author         = {L\"{u}cking, Andy and Stoeckel, Manuel and Abrami, Giuseppe and Mehler, Alexander},
  title     = {I still have Time(s): Extending HeidelTime for German Texts},
  booktitle      = {Proceedings of the 13th Language Resources and Evaluation Conference},
  month          = {June},
  series	= {LREC 2022},
  year           = {2022},
  address        = {Marseille, France},
  publisher      = {European Language Resources Association},
  pages     = {4723--4728},
  abstract  = {HeidelTime is one of the most widespread and successful tools for detecting temporal expressions in texts. Since HeidelTime’s pattern matching system is based on regular expression, it can be extended in a convenient way. We present such an extension for the German resources of HeidelTime: HeidelTimeExt. The extension has been brought about by means of observing false negatives within real world texts and various time banks. The gain in coverage is 2.7 \% or 8.5 \%, depending on the admitted degree of potential overgeneralization. We describe the development of HeidelTimeExt, its evaluation on text samples from various genres, and share some linguistic observations. HeidelTimeExt can be obtained from https://github.com/texttechnologylab/heideltime.},
  pdf    = {http://www.lrec-conf.org/proceedings/lrec2022/pdf/2022.lrec-1.505.pdf},
  url       = {https://aclanthology.org/2022.lrec-1.505}
}
```

## About TTLab's Extension of HeidelTime
HeidelTime is one of the most widespread and successful tools for detecting temporal expressions in texts. Since HeidelTime's pattern matching system is based on regular expression, it can be extended in a convenient way. We present such an extension for the German resources of HeidelTime: HeidelTimeext. The extension has been brought about by means of observing false negatives within real world texts and various time banks. The gain in coverage is 2.7 % or 8.5 %, depending on the admitted degree of potential overgeneralization. We describe the development of HeidelTimeext, its evaluation on text samples from various genres, and share some linguistic observations.

## Original HeidelTime

**HeidelTime** is a multilingual, domain-sensitive temporal tagger developed at the [Database Systems Research Group](http://dbs.ifi.uni-heidelberg.de/) at [Heidelberg University](http://www.uni-heidelberg.de/index_e.html). It extracts temporal expressions from documents and normalizes them according to the TIMEX3 annotation standard. HeidelTime is available as [UIMA](http://uima.apache.org/) annotator and as standalone version.

**HeidelTime** currently contains hand-crafted resources for **13 languages**: English, German, Dutch, Vietnamese, Arabic, Spanish, Italian, French, Chinese, Russian, Croatian, Estonian and Portuguese. In addition, starting with version 2.0, HeidelTime contains **automatically created resources for more than 200 languages**. Although these resources are of lower quality than the manually created ones, temporal tagging of many of these languages has never been addressed before. Thus, HeidelTime can be used as a baseline for temporal tagging of all these languages or as a starting point for developing temporal tagging capabilities for them. 

**HeidelTime** distinguishes between **news-style** documents and **narrative-style documents** (e.g., Wikipedia articles) in all languages. In addition, English colloquial (e.g., Tweets and SMS) and scientific articles (e.g., clinical trails) are supported.

Original **HeidelTime** can be obtained at [github](https://github.com/HeidelTime/heideltime).

Want to see what it can do before you delve in? Take a look at **HeidelTime**'s **[online demo](http://heideltime.ifi.uni-heidelberg.de/heideltime/)**.
