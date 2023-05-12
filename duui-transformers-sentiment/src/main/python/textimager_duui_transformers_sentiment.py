import logging
from functools import lru_cache
from itertools import chain
from platform import python_version
from sys import version as sys_version
from threading import Lock
from time import time
from typing import Dict, Union
from datetime import datetime

from cassis import load_typesystem
from fastapi import FastAPI, Response
from fastapi.responses import PlainTextResponse
import torch
from transformers import pipeline, __version__ as transformers_version, AutoTokenizer

from .duui.reqres import TextImagerResponse, TextImagerRequest
from .duui.sentiment import SentimentSentence, SentimentSelection
from .duui.service import Settings, TextImagerDocumentation, TextImagerCapability
from .duui.uima import *
from .models.cardiffnlp_twitter_roberta_base_sentiment import SUPPORTED_MODEL as CARDIFFNLP_TRBS
from .models.cardiffnlp_twitter_roberta_base_sentiment_latest import SUPPORTED_MODEL as CARDIFFNLP_TRBSL
from .models.cardiffnlp_twitter_xlm_roberta_base_sentiment import SUPPORTED_MODEL as CARDIFFNLP_TXRBS
from .models.clampert_multilingual_sentiment_covid19 import SUPPORTED_MODEL as CLAMPERT_MSC19
from .models.cmarkea_distilcamembert_base_sentiment import SUPPORTED_MODEL as CMARKEA_DBS
#from .models.dbaumartz_cardiffnlp_twitter_xlm_roberta_base_sentiment_finetuned_de_3sentiment_2_ep1_cp35057 import SUPPORTED_MODEL as DBAUMARTZ_CARDIFFNLP_TXRBSFD3_2E1
#from .models.dbaumartz_cardiffnlp_twitter_xlm_roberta_base_sentiment_finetuned_de_3sentiment_2_ep2_cp70114 import SUPPORTED_MODEL as DBAUMARTZ_CARDIFFNLP_TXRBSFD3_2E2
#from .models.dbaumartz_cardiffnlp_twitter_xlm_roberta_base_sentiment_finetuned_de_3sentiment_2_ep3_cp105171 import SUPPORTED_MODEL as DBAUMARTZ_CARDIFFNLP_TXRBSFD3_2E3
#from .models.dbaumartz_cardiffnlp_twitter_xlm_roberta_base_sentiment_finetuned_de_3sentiment_2_ep4_cp140228 import SUPPORTED_MODEL as DBAUMARTZ_CARDIFFNLP_TXRBSFD3_2E4
#from .models.dbaumartz_cardiffnlp_twitter_xlm_roberta_base_sentiment_finetuned_de_3sentiment_2_ep5_cp175285 import SUPPORTED_MODEL as DBAUMARTZ_CARDIFFNLP_TXRBSFD3_2E5
#from .models.dbaumartz_cardiffnlp_twitter_xlm_roberta_base_sentiment_finetuned_de_3sentiment_2_unseen_ep1_cp35010 import SUPPORTED_MODEL as DBAUMARTZ_CARDIFFNLP_TXRBSDF3U_2E1
#from .models.dbaumartz_cardiffnlp_twitter_xlm_roberta_base_sentiment_finetuned_de_3sentiment_2_unseen_ep2_cp70020 import SUPPORTED_MODEL as DBAUMARTZ_CARDIFFNLP_TXRBSDF3U_2E2
#from .models.dbaumartz_cardiffnlp_twitter_xlm_roberta_base_sentiment_finetuned_de_3sentiment_2_unseen_ep3_cp105030 import SUPPORTED_MODEL as DBAUMARTZ_CARDIFFNLP_TXRBSDF3U_2E3
#from .models.dbaumartz_cardiffnlp_twitter_xlm_roberta_base_sentiment_finetuned_de_3sentiment_2_unseen_ep4_cp140040 import SUPPORTED_MODEL as DBAUMARTZ_CARDIFFNLP_TXRBSDF3U_2E4
#from .models.dbaumartz_cardiffnlp_twitter_xlm_roberta_base_sentiment_finetuned_de_3sentiment_2_unseen_ep5_cp175050 import SUPPORTED_MODEL as DBAUMARTZ_CARDIFFNLP_TXRBSDF3U_2E5
#from .models.dbaumartz_cardiffnlp_twitter_xlm_roberta_base_sentiment_finetuned_de_3sentiment_2_unseen_adapter_pfeiffer_ep2_cp210060 import SUPPORTED_MODEL as DBAUMARTZ_CARDIFFNLP_TXRBSDF3UAP_2E2
#from .models.dbaumartz_cardiffnlp_twitter_xlm_roberta_base_sentiment_finetuned_de_3sentiment_2_unseen_adapter_pfeiffer_ep4_cp420120 import SUPPORTED_MODEL as DBAUMARTZ_CARDIFFNLP_TXRBSDF3UAP_2E4
#from .models.dbaumartz_cardiffnlp_twitter_xlm_roberta_base_sentiment_finetuned_de_3sentiment_2_unseen_adapter_pfeiffer_ep6_cp630180 import SUPPORTED_MODEL as DBAUMARTZ_CARDIFFNLP_TXRBSDF3UAP_2E6
#from .models.dbaumartz_cardiffnlp_twitter_xlm_roberta_base_sentiment_finetuned_de_3sentiment_2_unseen_adapter_pfeiffer_ep8_cp840240 import SUPPORTED_MODEL as DBAUMARTZ_CARDIFFNLP_TXRBSDF3UAP_2E8
#from .models.dbaumartz_cardiffnlp_twitter_xlm_roberta_base_sentiment_finetuned_de_3sentiment_2_unseen_adapter_pfeiffer_ep10_cp1050300 import SUPPORTED_MODEL as DBAUMARTZ_CARDIFFNLP_TXRBSDF3UAP_2E10
#from .models.dbaumartz_cardiffnlp_twitter_xlm_roberta_base_sentiment_finetuned_de_3sentiment_2_exact_ep1_cp30979 import SUPPORTED_MODEL as DBAUMARTZ_CARDIFFNLP_TXRBSDF3E_2E1
#from .models.dbaumartz_cardiffnlp_twitter_xlm_roberta_base_sentiment_finetuned_de_3sentiment_2_exact_ep2_cp61958 import SUPPORTED_MODEL as DBAUMARTZ_CARDIFFNLP_TXRBSDF3E_2E2
#from .models.dbaumartz_cardiffnlp_twitter_xlm_roberta_base_sentiment_finetuned_de_3sentiment_2_exact_ep3_cp92937 import SUPPORTED_MODEL as DBAUMARTZ_CARDIFFNLP_TXRBSDF3E_2E3
#from .models.dbaumartz_cardiffnlp_twitter_xlm_roberta_base_sentiment_finetuned_de_3sentiment_2_exact_ep4_cp123916 import SUPPORTED_MODEL as DBAUMARTZ_CARDIFFNLP_TXRBSDF3E_2E4
#from .models.dbaumartz_cardiffnlp_twitter_xlm_roberta_base_sentiment_finetuned_de_3sentiment_2_exact_ep5_cp154895 import SUPPORTED_MODEL as DBAUMARTZ_CARDIFFNLP_TXRBSDF3E_2E5
#from .models.dbaumartz_cardiffnlp_twitter_xlm_roberta_base_sentiment_finetuned_de_3sentiment_2_exact_ep6_cp185874 import SUPPORTED_MODEL as DBAUMARTZ_CARDIFFNLP_TXRBSDF3E_2E6
#from .models.dbaumartz_cardiffnlp_twitter_xlm_roberta_base_sentiment_finetuned_de_3sentiment_2_exact_ep7_cp216853 import SUPPORTED_MODEL as DBAUMARTZ_CARDIFFNLP_TXRBSDF3E_2E7
#from .models.dbaumartz_cardiffnlp_twitter_xlm_roberta_base_sentiment_finetuned_de_3sentiment_2_exact_ep8_cp247832 import SUPPORTED_MODEL as DBAUMARTZ_CARDIFFNLP_TXRBSDF3E_2E8
#from .models.dbaumartz_cardiffnlp_twitter_xlm_roberta_base_sentiment_finetuned_de_3sentiment_2_exact_ep9_cp278811 import SUPPORTED_MODEL as DBAUMARTZ_CARDIFFNLP_TXRBSDF3E_2E9
#from .models.dbaumartz_cardiffnlp_twitter_xlm_roberta_base_sentiment_finetuned_de_3sentiment_2_exact_ep10_cp309790 import SUPPORTED_MODEL as DBAUMARTZ_CARDIFFNLP_TXRBSDF3E_2E10
#from .models.dbaumartz_cardiffnlp_twitter_xlm_roberta_base_sentiment_finetuned_de_3sentiment_2_exact_ep11_cp340769 import SUPPORTED_MODEL as DBAUMARTZ_CARDIFFNLP_TXRBSDF3E_2E11
#from .models.dbaumartz_cardiffnlp_twitter_xlm_roberta_base_sentiment_finetuned_de_3sentiment_2_exact_ep12_cp371748 import SUPPORTED_MODEL as DBAUMARTZ_CARDIFFNLP_TXRBSDF3E_2E12
#from .models.dbaumartz_cardiffnlp_twitter_xlm_roberta_base_sentiment_finetuned_de_3sentiment_2_exact_ep13_cp402727 import SUPPORTED_MODEL as DBAUMARTZ_CARDIFFNLP_TXRBSDF3E_2E13
#from .models.dbaumartz_cardiffnlp_twitter_xlm_roberta_base_sentiment_finetuned_de_3sentiment_2_exact_ep14_cp433706 import SUPPORTED_MODEL as DBAUMARTZ_CARDIFFNLP_TXRBSDF3E_2E14
#from .models.dbaumartz_cardiffnlp_twitter_xlm_roberta_base_sentiment_finetuned_de_3sentiment_2_exact_ep15_cp464685 import SUPPORTED_MODEL as DBAUMARTZ_CARDIFFNLP_TXRBSDF3E_2E15
#from .models.dbaumartz_cardiffnlp_twitter_xlm_roberta_base_sentiment_finetuned_de_3sentiment_2_exact_ep16_cp495664 import SUPPORTED_MODEL as DBAUMARTZ_CARDIFFNLP_TXRBSDF3E_2E16
#from .models.dbaumartz_cardiffnlp_twitter_xlm_roberta_base_sentiment_finetuned_de_3sentiment_2_exact_ep17_cp526643 import SUPPORTED_MODEL as DBAUMARTZ_CARDIFFNLP_TXRBSDF3E_2E17
#from .models.dbaumartz_cardiffnlp_twitter_xlm_roberta_base_sentiment_finetuned_de_3sentiment_2_exact_ep18_cp557622 import SUPPORTED_MODEL as DBAUMARTZ_CARDIFFNLP_TXRBSDF3E_2E18
#from .models.dbaumartz_cardiffnlp_twitter_xlm_roberta_base_sentiment_finetuned_de_3sentiment_2_exact_ep19_cp588601 import SUPPORTED_MODEL as DBAUMARTZ_CARDIFFNLP_TXRBSDF3E_2E19
#from .models.dbaumartz_cardiffnlp_twitter_xlm_roberta_base_sentiment_finetuned_de_3sentiment_2_exact_ep20_cp619580 import SUPPORTED_MODEL as DBAUMARTZ_CARDIFFNLP_TXRBSDF3E_2E20
#from .models.dbaumartz_mdraw_german_news_sentiment_bert_finetuned_de_3sentiment_2_ep1_cp35057 import SUPPORTED_MODEL as DBAUMARTZ_MDRAW_GNSBFD3_2E1
#from .models.dbaumartz_mdraw_german_news_sentiment_bert_finetuned_de_3sentiment_2_ep2_cp70114 import SUPPORTED_MODEL as DBAUMARTZ_MDRAW_GNSBFD3_2E2
#from .models.dbaumartz_mdraw_german_news_sentiment_bert_finetuned_de_3sentiment_2_ep3_cp105171 import SUPPORTED_MODEL as DBAUMARTZ_MDRAW_GNSBFD3_2E3
#from .models.dbaumartz_mdraw_german_news_sentiment_bert_finetuned_de_3sentiment_2_ep4_cp140228 import SUPPORTED_MODEL as DBAUMARTZ_MDRAW_GNSBFD3_2E4
#from .models.dbaumartz_mdraw_german_news_sentiment_bert_finetuned_de_3sentiment_2_ep5_cp175285 import SUPPORTED_MODEL as DBAUMARTZ_MDRAW_GNSBFD3_2E5
#from .models.dbaumartz_mdraw_german_news_sentiment_bert_finetuned_de_3sentiment_2_unseen_ep1_cp4193 import SUPPORTED_MODEL as DBAUMARTZ_MDRAW_GNSBFD3U_2E1
#from .models.dbaumartz_mdraw_german_news_sentiment_bert_finetuned_de_3sentiment_2_unseen_ep2_cp8386 import SUPPORTED_MODEL as DBAUMARTZ_MDRAW_GNSBFD3U_2E2
#from .models.dbaumartz_mdraw_german_news_sentiment_bert_finetuned_de_3sentiment_2_unseen_ep3_cp12579 import SUPPORTED_MODEL as DBAUMARTZ_MDRAW_GNSBFD3U_2E3
#from .models.dbaumartz_mdraw_german_news_sentiment_bert_finetuned_de_3sentiment_2_unseen_ep4_cp16772 import SUPPORTED_MODEL as DBAUMARTZ_MDRAW_GNSBFD3U_2E4
#from .models.dbaumartz_mdraw_german_news_sentiment_bert_finetuned_de_3sentiment_2_unseen_ep5_cp20965 import SUPPORTED_MODEL as DBAUMARTZ_MDRAW_GNSBFD3U_2E5
#from .models.dbaumartz_mdraw_german_news_sentiment_bert_finetuned_de_3sentiment_2_unseen_adapter_pfeiffer_ep2_cp25156 import SUPPORTED_MODEL as DBAUMARTZ_MDRAW_GNSBFD3UAP_2E2
#from .models.dbaumartz_mdraw_german_news_sentiment_bert_finetuned_de_3sentiment_2_unseen_adapter_pfeiffer_ep4_cp50312 import SUPPORTED_MODEL as DBAUMARTZ_MDRAW_GNSBFD3UAP_2E4
#from .models.dbaumartz_mdraw_german_news_sentiment_bert_finetuned_de_3sentiment_2_unseen_adapter_pfeiffer_ep6_cp75468 import SUPPORTED_MODEL as DBAUMARTZ_MDRAW_GNSBFD3UAP_2E6
#from .models.dbaumartz_mdraw_german_news_sentiment_bert_finetuned_de_3sentiment_2_unseen_adapter_pfeiffer_ep8_cp100624 import SUPPORTED_MODEL as DBAUMARTZ_MDRAW_GNSBFD3UAP_2E8
#from .models.dbaumartz_mdraw_german_news_sentiment_bert_finetuned_de_3sentiment_2_unseen_adapter_pfeiffer_ep10_cp125780 import SUPPORTED_MODEL as DBAUMARTZ_MDRAW_GNSBFD3UAP_2E10
#from .models.dbaumartz_mdraw_german_news_sentiment_bert_finetuned_de_3sentiment_2_exact_ep1_cp30979 import SUPPORTED_MODEL as DBAUMARTZ_MDRAW_GNSBFD3E_2E1
# from .models.dbaumartz_mdraw_german_news_sentiment_bert_finetuned_de_3sentiment_2_exact_ep6_cp185874 import SUPPORTED_MODEL as DBAUMARTZ_MDRAW_GNSBFD3E_2E6
# from .models.dbaumartz_mdraw_german_news_sentiment_bert_finetuned_de_3sentiment_2_exact_ep7_cp216853 import SUPPORTED_MODEL as DBAUMARTZ_MDRAW_GNSBFD3E_2E7
# from .models.dbaumartz_mdraw_german_news_sentiment_bert_finetuned_de_3sentiment_2_exact_ep8_cp247832 import SUPPORTED_MODEL as DBAUMARTZ_MDRAW_GNSBFD3E_2E8
# from .models.dbaumartz_mdraw_german_news_sentiment_bert_finetuned_de_3sentiment_2_exact_ep9_cp278811 import SUPPORTED_MODEL as DBAUMARTZ_MDRAW_GNSBFD3E_2E9
# from .models.dbaumartz_mdraw_german_news_sentiment_bert_finetuned_de_3sentiment_2_exact_ep10_cp309790 import SUPPORTED_MODEL as DBAUMARTZ_MDRAW_GNSBFD3E_2E10
# from .models.dbaumartz_mdraw_german_news_sentiment_bert_finetuned_de_3sentiment_2_exact_ep11_cp340769 import SUPPORTED_MODEL as DBAUMARTZ_MDRAW_GNSBFD3E_2E11
# from .models.dbaumartz_mdraw_german_news_sentiment_bert_finetuned_de_3sentiment_2_exact_ep12_cp371748 import SUPPORTED_MODEL as DBAUMARTZ_MDRAW_GNSBFD3E_2E12
# from .models.dbaumartz_mdraw_german_news_sentiment_bert_finetuned_de_3sentiment_2_exact_ep13_cp402727 import SUPPORTED_MODEL as DBAUMARTZ_MDRAW_GNSBFD3E_2E13
# from .models.dbaumartz_mdraw_german_news_sentiment_bert_finetuned_de_3sentiment_2_exact_ep14_cp433706 import SUPPORTED_MODEL as DBAUMARTZ_MDRAW_GNSBFD3E_2E14
# from .models.dbaumartz_mdraw_german_news_sentiment_bert_finetuned_de_3sentiment_2_exact_ep15_cp464685 import SUPPORTED_MODEL as DBAUMARTZ_MDRAW_GNSBFD3E_2E15
# from .models.dbaumartz_mdraw_german_news_sentiment_bert_finetuned_de_3sentiment_2_exact_ep16_cp495664 import SUPPORTED_MODEL as DBAUMARTZ_MDRAW_GNSBFD3E_2E16
# from .models.dbaumartz_mdraw_german_news_sentiment_bert_finetuned_de_3sentiment_2_exact_ep17_cp526643 import SUPPORTED_MODEL as DBAUMARTZ_MDRAW_GNSBFD3E_2E17
# from .models.dbaumartz_mdraw_german_news_sentiment_bert_finetuned_de_3sentiment_2_exact_ep18_cp557622 import SUPPORTED_MODEL as DBAUMARTZ_MDRAW_GNSBFD3E_2E18
# from .models.dbaumartz_mdraw_german_news_sentiment_bert_finetuned_de_3sentiment_2_exact_ep19_cp588601 import SUPPORTED_MODEL as DBAUMARTZ_MDRAW_GNSBFD3E_2E19
# from .models.dbaumartz_mdraw_german_news_sentiment_bert_finetuned_de_3sentiment_2_exact_ep20_cp619580 import SUPPORTED_MODEL as DBAUMARTZ_MDRAW_GNSBFD3E_2E20
# #from .models.dbaumartz_mdraw_german_news_sentiment_bert_finetuned_de_3sentiment_2_exact_ep2_cp61958 import SUPPORTED_MODEL as DBAUMARTZ_MDRAW_GNSBFD3E_2E2
#from .models.dbaumartz_mdraw_german_news_sentiment_bert_finetuned_de_3sentiment_2_exact_ep3_cp92937 import SUPPORTED_MODEL as DBAUMARTZ_MDRAW_GNSBFD3E_2E3
#from .models.dbaumartz_mdraw_german_news_sentiment_bert_finetuned_de_3sentiment_2_exact_ep4_cp123916 import SUPPORTED_MODEL as DBAUMARTZ_MDRAW_GNSBFD3E_2E4
#from .models.dbaumartz_mdraw_german_news_sentiment_bert_finetuned_de_3sentiment_2_exact_ep5_cp154895 import SUPPORTED_MODEL as DBAUMARTZ_MDRAW_GNSBFD3E_2E5
#from .models.dbaumartz_oliverguhr_german_sentiment_bert_finetuned_de_3sentiment_2_ep1_cp35057 import SUPPORTED_MODEL as DBAUMARTZ_OLIVERGUHR_GSBFD3_2E1
#from .models.dbaumartz_oliverguhr_german_sentiment_bert_finetuned_de_3sentiment_2_ep2_cp70114 import SUPPORTED_MODEL as DBAUMARTZ_OLIVERGUHR_GSBFD3_2E2
#from .models.dbaumartz_oliverguhr_german_sentiment_bert_finetuned_de_3sentiment_2_ep3_cp105171 import SUPPORTED_MODEL as DBAUMARTZ_OLIVERGUHR_GSBFD3_2E3
#from .models.dbaumartz_oliverguhr_german_sentiment_bert_finetuned_de_3sentiment_2_ep4_cp140228 import SUPPORTED_MODEL as DBAUMARTZ_OLIVERGUHR_GSBFD3_2E4
#from .models.dbaumartz_oliverguhr_german_sentiment_bert_finetuned_de_3sentiment_2_ep5_cp175285 import SUPPORTED_MODEL as DBAUMARTZ_OLIVERGUHR_GSBFD3_2E5
#from .models.dbaumartz_oliverguhr_german_sentiment_bert_finetuned_de_3sentiment_2_unseen_ep1_cp4224 import SUPPORTED_MODEL as DBAUMARTZ_OLIVERGUHR_GSBFD3U_2E1
#from .models.dbaumartz_oliverguhr_german_sentiment_bert_finetuned_de_3sentiment_2_unseen_ep2_cp8448 import SUPPORTED_MODEL as DBAUMARTZ_OLIVERGUHR_GSBFD3U_2E2
#from .models.dbaumartz_oliverguhr_german_sentiment_bert_finetuned_de_3sentiment_2_unseen_ep3_cp12672 import SUPPORTED_MODEL as DBAUMARTZ_OLIVERGUHR_GSBFD3U_2E3
#from .models.dbaumartz_oliverguhr_german_sentiment_bert_finetuned_de_3sentiment_2_unseen_ep4_cp16896 import SUPPORTED_MODEL as DBAUMARTZ_OLIVERGUHR_GSBFD3U_2E4
#from .models.dbaumartz_oliverguhr_german_sentiment_bert_finetuned_de_3sentiment_2_unseen_ep5_cp21120 import SUPPORTED_MODEL as DBAUMARTZ_OLIVERGUHR_GSBFD3U_2E5
#from .models.dbaumartz_oliverguhr_german_sentiment_bert_finetuned_de_3sentiment_2_unseen_adapter_pfeiffer_ep2_cp25342 import SUPPORTED_MODEL as DBAUMARTZ_OLIVERGUHR_GSBFD3UAP_2E2
#from .models.dbaumartz_oliverguhr_german_sentiment_bert_finetuned_de_3sentiment_2_unseen_adapter_pfeiffer_ep4_cp50684 import SUPPORTED_MODEL as DBAUMARTZ_OLIVERGUHR_GSBFD3UAP_2E4
#from .models.dbaumartz_oliverguhr_german_sentiment_bert_finetuned_de_3sentiment_2_unseen_adapter_pfeiffer_ep6_cp76026 import SUPPORTED_MODEL as DBAUMARTZ_OLIVERGUHR_GSBFD3UAP_2E6
#from .models.dbaumartz_oliverguhr_german_sentiment_bert_finetuned_de_3sentiment_2_unseen_adapter_pfeiffer_ep8_cp101368 import SUPPORTED_MODEL as DBAUMARTZ_OLIVERGUHR_GSBFD3UAP_2E8
#from .models.dbaumartz_oliverguhr_german_sentiment_bert_finetuned_de_3sentiment_2_unseen_adapter_pfeiffer_ep10_cp126710 import SUPPORTED_MODEL as DBAUMARTZ_OLIVERGUHR_GSBFD3UAP_2E10
#from .models.dbaumartz_oliverguhr_german_sentiment_bert_finetuned_de_3sentiment_2_exact_ep1_cp30979 import SUPPORTED_MODEL as DBAUMARTZ_OLIVERGUHR_GSBFD3E_2E1
#from .models.dbaumartz_oliverguhr_german_sentiment_bert_finetuned_de_3sentiment_2_exact_ep2_cp61958 import SUPPORTED_MODEL as DBAUMARTZ_OLIVERGUHR_GSBFD3E_2E2
#from .models.dbaumartz_oliverguhr_german_sentiment_bert_finetuned_de_3sentiment_2_exact_ep3_cp92937 import SUPPORTED_MODEL as DBAUMARTZ_OLIVERGUHR_GSBFD3E_2E3
#from .models.dbaumartz_oliverguhr_german_sentiment_bert_finetuned_de_3sentiment_2_exact_ep4_cp123916 import SUPPORTED_MODEL as DBAUMARTZ_OLIVERGUHR_GSBFD3E_2E4
#from .models.dbaumartz_oliverguhr_german_sentiment_bert_finetuned_de_3sentiment_2_exact_ep5_cp154895 import SUPPORTED_MODEL as DBAUMARTZ_OLIVERGUHR_GSBFD3E_2E5
# from .models.dbaumartz_oliverguhr_german_sentiment_bert_finetuned_de_3sentiment_2_exact_ep6_cp185874 import SUPPORTED_MODEL as DBAUMARTZ_OLIVERGUHR_GSBFD3E_2E6
# from .models.dbaumartz_oliverguhr_german_sentiment_bert_finetuned_de_3sentiment_2_exact_ep7_cp216853 import SUPPORTED_MODEL as DBAUMARTZ_OLIVERGUHR_GSBFD3E_2E7
# from .models.dbaumartz_oliverguhr_german_sentiment_bert_finetuned_de_3sentiment_2_exact_ep8_cp247832 import SUPPORTED_MODEL as DBAUMARTZ_OLIVERGUHR_GSBFD3E_2E8
# from .models.dbaumartz_oliverguhr_german_sentiment_bert_finetuned_de_3sentiment_2_exact_ep9_cp278811 import SUPPORTED_MODEL as DBAUMARTZ_OLIVERGUHR_GSBFD3E_2E9
# from .models.dbaumartz_oliverguhr_german_sentiment_bert_finetuned_de_3sentiment_2_exact_ep10_cp309790 import SUPPORTED_MODEL as DBAUMARTZ_OLIVERGUHR_GSBFD3E_2E10
# from .models.dbaumartz_oliverguhr_german_sentiment_bert_finetuned_de_3sentiment_2_exact_ep11_cp340769 import SUPPORTED_MODEL as DBAUMARTZ_OLIVERGUHR_GSBFD3E_2E11
# from .models.dbaumartz_oliverguhr_german_sentiment_bert_finetuned_de_3sentiment_2_exact_ep12_cp371748 import SUPPORTED_MODEL as DBAUMARTZ_OLIVERGUHR_GSBFD3E_2E12
# from .models.dbaumartz_oliverguhr_german_sentiment_bert_finetuned_de_3sentiment_2_exact_ep13_cp402727 import SUPPORTED_MODEL as DBAUMARTZ_OLIVERGUHR_GSBFD3E_2E13
# from .models.dbaumartz_oliverguhr_german_sentiment_bert_finetuned_de_3sentiment_2_exact_ep14_cp433706 import SUPPORTED_MODEL as DBAUMARTZ_OLIVERGUHR_GSBFD3E_2E14
# from .models.dbaumartz_oliverguhr_german_sentiment_bert_finetuned_de_3sentiment_2_exact_ep15_cp464685 import SUPPORTED_MODEL as DBAUMARTZ_OLIVERGUHR_GSBFD3E_2E15
# from .models.dbaumartz_oliverguhr_german_sentiment_bert_finetuned_de_3sentiment_2_exact_ep16_cp495664 import SUPPORTED_MODEL as DBAUMARTZ_OLIVERGUHR_GSBFD3E_2E16
# from .models.dbaumartz_oliverguhr_german_sentiment_bert_finetuned_de_3sentiment_2_exact_ep17_cp526643 import SUPPORTED_MODEL as DBAUMARTZ_OLIVERGUHR_GSBFD3E_2E17
# from .models.dbaumartz_oliverguhr_german_sentiment_bert_finetuned_de_3sentiment_2_exact_ep18_cp557622 import SUPPORTED_MODEL as DBAUMARTZ_OLIVERGUHR_GSBFD3E_2E18
# from .models.dbaumartz_oliverguhr_german_sentiment_bert_finetuned_de_3sentiment_2_exact_ep19_cp588601 import SUPPORTED_MODEL as DBAUMARTZ_OLIVERGUHR_GSBFD3E_2E19
# from .models.dbaumartz_oliverguhr_german_sentiment_bert_finetuned_de_3sentiment_2_exact_ep20_cp619580 import SUPPORTED_MODEL as DBAUMARTZ_OLIVERGUHR_GSBFD3E_2E20
#from .models.dbaumartz_philschmid_distilbert_base_multilingual_cased_sentiment_2_finetuned_de_3sentiment_2_ep1_cp35057 import SUPPORTED_MODEL as DBAUMARTZ_PHILSCHMID_DBMCS2FD3_2E1
#from .models.dbaumartz_philschmid_distilbert_base_multilingual_cased_sentiment_2_finetuned_de_3sentiment_2_ep2_cp70114 import SUPPORTED_MODEL as DBAUMARTZ_PHILSCHMID_DBMCS2FD3_2E2
#from .models.dbaumartz_philschmid_distilbert_base_multilingual_cased_sentiment_2_finetuned_de_3sentiment_2_ep3_cp105171 import SUPPORTED_MODEL as DBAUMARTZ_PHILSCHMID_DBMCS2FD3_2E3
#from .models.dbaumartz_philschmid_distilbert_base_multilingual_cased_sentiment_2_finetuned_de_3sentiment_2_ep4_cp140228 import SUPPORTED_MODEL as DBAUMARTZ_PHILSCHMID_DBMCS2FD3_2E4
#from .models.dbaumartz_philschmid_distilbert_base_multilingual_cased_sentiment_2_finetuned_de_3sentiment_2_ep5_cp175285 import SUPPORTED_MODEL as DBAUMARTZ_PHILSCHMID_DBMCS2FD3_2E5
#from .models.dbaumartz_philschmid_distilbert_base_multilingual_cased_sentiment_2_finetuned_de_3sentiment_2_unseen_ep1_cp30870 import SUPPORTED_MODEL as DBAUMARTZ_PHILSCHMID_DBMCS2FD3U_2E1
#from .models.dbaumartz_philschmid_distilbert_base_multilingual_cased_sentiment_2_finetuned_de_3sentiment_2_unseen_ep2_cp61740 import SUPPORTED_MODEL as DBAUMARTZ_PHILSCHMID_DBMCS2FD3U_2E2
#from .models.dbaumartz_philschmid_distilbert_base_multilingual_cased_sentiment_2_finetuned_de_3sentiment_2_unseen_ep3_cp92610 import SUPPORTED_MODEL as DBAUMARTZ_PHILSCHMID_DBMCS2FD3U_2E3
#from .models.dbaumartz_philschmid_distilbert_base_multilingual_cased_sentiment_2_finetuned_de_3sentiment_2_unseen_ep4_cp123480 import SUPPORTED_MODEL as DBAUMARTZ_PHILSCHMID_DBMCS2FD3U_2E4
#from .models.dbaumartz_philschmid_distilbert_base_multilingual_cased_sentiment_2_finetuned_de_3sentiment_2_unseen_ep5_cp154350 import SUPPORTED_MODEL as DBAUMARTZ_PHILSCHMID_DBMCS2FD3U_2E5
#from .models.dbaumartz_philschmid_distilbert_base_multilingual_cased_sentiment_2_finetuned_de_3sentiment_2_unseen_adapter_pfeiffer_ep2_cp185216 import SUPPORTED_MODEL as DBAUMARTZ_PHILSCHMID_DBMCS2FD3UAP_2E2
#from .models.dbaumartz_philschmid_distilbert_base_multilingual_cased_sentiment_2_finetuned_de_3sentiment_2_unseen_adapter_pfeiffer_ep4_cp370432 import SUPPORTED_MODEL as DBAUMARTZ_PHILSCHMID_DBMCS2FD3UAP_2E4
#from .models.dbaumartz_philschmid_distilbert_base_multilingual_cased_sentiment_2_finetuned_de_3sentiment_2_unseen_adapter_pfeiffer_ep6_cp555648 import SUPPORTED_MODEL as DBAUMARTZ_PHILSCHMID_DBMCS2FD3UAP_2E6
#from .models.dbaumartz_philschmid_distilbert_base_multilingual_cased_sentiment_2_finetuned_de_3sentiment_2_unseen_adapter_pfeiffer_ep8_cp740864 import SUPPORTED_MODEL as DBAUMARTZ_PHILSCHMID_DBMCS2FD3UAP_2E8
#from .models.dbaumartz_philschmid_distilbert_base_multilingual_cased_sentiment_2_finetuned_de_3sentiment_2_unseen_adapter_pfeiffer_ep10_cp926080 import SUPPORTED_MODEL as DBAUMARTZ_PHILSCHMID_DBMCS2FD3UAP_2E10
#from .models.dbaumartz_philschmid_distilbert_base_multilingual_cased_sentiment_2_finetuned_de_3sentiment_2_exact_ep1_cp30979 import SUPPORTED_MODEL as DBAUMARTZ_PHILSCHMID_DBMCS2FD3E_2E1
#from .models.dbaumartz_philschmid_distilbert_base_multilingual_cased_sentiment_2_finetuned_de_3sentiment_2_exact_ep2_cp61958 import SUPPORTED_MODEL as DBAUMARTZ_PHILSCHMID_DBMCS2FD3E_2E2
#from .models.dbaumartz_philschmid_distilbert_base_multilingual_cased_sentiment_2_finetuned_de_3sentiment_2_exact_ep3_cp92937 import SUPPORTED_MODEL as DBAUMARTZ_PHILSCHMID_DBMCS2FD3E_2E3
#from .models.dbaumartz_philschmid_distilbert_base_multilingual_cased_sentiment_2_finetuned_de_3sentiment_2_exact_ep4_cp123916 import SUPPORTED_MODEL as DBAUMARTZ_PHILSCHMID_DBMCS2FD3E_2E4
#from .models.dbaumartz_philschmid_distilbert_base_multilingual_cased_sentiment_2_finetuned_de_3sentiment_2_exact_ep5_cp154895 import SUPPORTED_MODEL as DBAUMARTZ_PHILSCHMID_DBMCS2FD3E_2E5
#from .models.dbaumartz_philschmid_distilbert_base_multilingual_cased_sentiment_2_finetuned_de_3sentiment_2_exact_ep6_cp185874 import SUPPORTED_MODEL as DBAUMARTZ_PHILSCHMID_DBMCS2FD3E_2E6
#from .models.dbaumartz_philschmid_distilbert_base_multilingual_cased_sentiment_2_finetuned_de_3sentiment_2_exact_ep7_cp216853 import SUPPORTED_MODEL as DBAUMARTZ_PHILSCHMID_DBMCS2FD3E_2E7
#from .models.dbaumartz_philschmid_distilbert_base_multilingual_cased_sentiment_2_finetuned_de_3sentiment_2_exact_ep8_cp247832 import SUPPORTED_MODEL as DBAUMARTZ_PHILSCHMID_DBMCS2FD3E_2E8
#from .models.dbaumartz_philschmid_distilbert_base_multilingual_cased_sentiment_2_finetuned_de_3sentiment_2_exact_ep9_cp278811 import SUPPORTED_MODEL as DBAUMARTZ_PHILSCHMID_DBMCS2FD3E_2E9
#from .models.dbaumartz_philschmid_distilbert_base_multilingual_cased_sentiment_2_finetuned_de_3sentiment_2_exact_ep10_cp309790 import SUPPORTED_MODEL as DBAUMARTZ_PHILSCHMID_DBMCS2FD3E_2E10
#from .models.dbaumartz_philschmid_distilbert_base_multilingual_cased_sentiment_2_finetuned_de_3sentiment_2_exact_ep11_cp340769 import SUPPORTED_MODEL as DBAUMARTZ_PHILSCHMID_DBMCS2FD3E_2E11
#from .models.dbaumartz_philschmid_distilbert_base_multilingual_cased_sentiment_2_finetuned_de_3sentiment_2_exact_ep12_cp371748 import SUPPORTED_MODEL as DBAUMARTZ_PHILSCHMID_DBMCS2FD3E_2E12
#from .models.dbaumartz_philschmid_distilbert_base_multilingual_cased_sentiment_2_finetuned_de_3sentiment_2_exact_ep13_cp402727 import SUPPORTED_MODEL as DBAUMARTZ_PHILSCHMID_DBMCS2FD3E_2E13
#from .models.dbaumartz_philschmid_distilbert_base_multilingual_cased_sentiment_2_finetuned_de_3sentiment_2_exact_ep14_cp433706 import SUPPORTED_MODEL as DBAUMARTZ_PHILSCHMID_DBMCS2FD3E_2E14
#from .models.dbaumartz_philschmid_distilbert_base_multilingual_cased_sentiment_2_finetuned_de_3sentiment_2_exact_ep15_cp464685 import SUPPORTED_MODEL as DBAUMARTZ_PHILSCHMID_DBMCS2FD3E_2E15
#from .models.dbaumartz_philschmid_distilbert_base_multilingual_cased_sentiment_2_finetuned_de_3sentiment_2_exact_ep16_cp495664 import SUPPORTED_MODEL as DBAUMARTZ_PHILSCHMID_DBMCS2FD3E_2E16
#from .models.dbaumartz_philschmid_distilbert_base_multilingual_cased_sentiment_2_finetuned_de_3sentiment_2_exact_ep17_cp526643 import SUPPORTED_MODEL as DBAUMARTZ_PHILSCHMID_DBMCS2FD3E_2E17
#from .models.dbaumartz_philschmid_distilbert_base_multilingual_cased_sentiment_2_finetuned_de_3sentiment_2_exact_ep18_cp557622 import SUPPORTED_MODEL as DBAUMARTZ_PHILSCHMID_DBMCS2FD3E_2E18
#from .models.dbaumartz_philschmid_distilbert_base_multilingual_cased_sentiment_2_finetuned_de_3sentiment_2_exact_ep19_cp588601 import SUPPORTED_MODEL as DBAUMARTZ_PHILSCHMID_DBMCS2FD3E_2E19
#from .models.dbaumartz_philschmid_distilbert_base_multilingual_cased_sentiment_2_finetuned_de_3sentiment_2_exact_ep20_cp619580 import SUPPORTED_MODEL as DBAUMARTZ_PHILSCHMID_DBMCS2FD3E_2E20
from .models.finiteautomata_bertweet_base_sentiment_analysis import SUPPORTED_MODEL as FINITEAUTOMATA_BBSA
from .models.j_hartmann_sentiment_roberta_large_english_3_classes import SUPPORTED_MODEL as JHARTMANN_SRLE3C
from .models.liyuan_amazon_review_sentiment_analysis import SUPPORTED_MODEL as LIYUAN_ARSA
from .models.mdraw_german_news_sentiment_bert import SUPPORTED_MODEL as MDRAW_GNSB
from .models.nlptown_bert_base_multilingual_uncased_sentiment import SUPPORTED_MODEL as NLPTOWN_BBMUS
from .models.oliverguhr_german_sentiment_bert import SUPPORTED_MODEL as OLIVERGUHR_GSB
from .models.philschmid_distilbert_base_multilingual_cased_sentiment_2 import SUPPORTED_MODEL as PHILSHMID_DBMCS2
from .models.siebert_sentiment_roberta_large_english import SUPPORTED_MODEL as SIEBERT_SRLE


SUPPORTED_MODELS = {
    **CARDIFFNLP_TRBS,
    **CARDIFFNLP_TRBSL,
    **CARDIFFNLP_TXRBS,
    **NLPTOWN_BBMUS,
    **FINITEAUTOMATA_BBSA,
    **SIEBERT_SRLE,
    **JHARTMANN_SRLE3C,
    **LIYUAN_ARSA,
    **PHILSHMID_DBMCS2,
    **CLAMPERT_MSC19,
    **OLIVERGUHR_GSB,
    **MDRAW_GNSB,
    **CMARKEA_DBS,
    #**DBAUMARTZ_CARDIFFNLP_TXRBSFD3_2E1,
    #**DBAUMARTZ_CARDIFFNLP_TXRBSFD3_2E2,
    #**DBAUMARTZ_CARDIFFNLP_TXRBSFD3_2E3,
    #**DBAUMARTZ_CARDIFFNLP_TXRBSFD3_2E4,
    #**DBAUMARTZ_CARDIFFNLP_TXRBSFD3_2E5,
    #**DBAUMARTZ_CARDIFFNLP_TXRBSDF3U_2E1,
    #**DBAUMARTZ_CARDIFFNLP_TXRBSDF3U_2E2,
    #**DBAUMARTZ_CARDIFFNLP_TXRBSDF3U_2E3,
    #**DBAUMARTZ_CARDIFFNLP_TXRBSDF3U_2E4,
    #**DBAUMARTZ_CARDIFFNLP_TXRBSDF3U_2E5,
    #**DBAUMARTZ_CARDIFFNLP_TXRBSDF3UAP_2E2,
    #**DBAUMARTZ_CARDIFFNLP_TXRBSDF3UAP_2E4,
    #**DBAUMARTZ_CARDIFFNLP_TXRBSDF3UAP_2E6,
    #**DBAUMARTZ_CARDIFFNLP_TXRBSDF3UAP_2E8,
    #**DBAUMARTZ_CARDIFFNLP_TXRBSDF3UAP_2E10,
    #**DBAUMARTZ_CARDIFFNLP_TXRBSDF3E_2E1,
    #**DBAUMARTZ_CARDIFFNLP_TXRBSDF3E_2E2,
    #**DBAUMARTZ_CARDIFFNLP_TXRBSDF3E_2E3,
    #**DBAUMARTZ_CARDIFFNLP_TXRBSDF3E_2E4,
    #**DBAUMARTZ_CARDIFFNLP_TXRBSDF3E_2E5,
    #**DBAUMARTZ_CARDIFFNLP_TXRBSDF3E_2E6,
    #**DBAUMARTZ_CARDIFFNLP_TXRBSDF3E_2E7,
    #**DBAUMARTZ_CARDIFFNLP_TXRBSDF3E_2E8,
    #**DBAUMARTZ_CARDIFFNLP_TXRBSDF3E_2E9,
    #**DBAUMARTZ_CARDIFFNLP_TXRBSDF3E_2E10,
    #**DBAUMARTZ_CARDIFFNLP_TXRBSDF3E_2E11,
    #**DBAUMARTZ_CARDIFFNLP_TXRBSDF3E_2E12,
    #**DBAUMARTZ_CARDIFFNLP_TXRBSDF3E_2E13,
    #**DBAUMARTZ_CARDIFFNLP_TXRBSDF3E_2E14,
    #**DBAUMARTZ_CARDIFFNLP_TXRBSDF3E_2E15,
    #**DBAUMARTZ_CARDIFFNLP_TXRBSDF3E_2E16,
    #**DBAUMARTZ_CARDIFFNLP_TXRBSDF3E_2E17,
    #**DBAUMARTZ_CARDIFFNLP_TXRBSDF3E_2E18,
    #**DBAUMARTZ_CARDIFFNLP_TXRBSDF3E_2E19,
    #**DBAUMARTZ_CARDIFFNLP_TXRBSDF3E_2E20,
    #**DBAUMARTZ_MDRAW_GNSBFD3_2E1,
    #**DBAUMARTZ_MDRAW_GNSBFD3_2E2,
    #**DBAUMARTZ_MDRAW_GNSBFD3_2E3,
    #**DBAUMARTZ_MDRAW_GNSBFD3_2E4,
    #**DBAUMARTZ_MDRAW_GNSBFD3_2E5,
    #**DBAUMARTZ_MDRAW_GNSBFD3U_2E1,
    #**DBAUMARTZ_MDRAW_GNSBFD3U_2E2,
    #**DBAUMARTZ_MDRAW_GNSBFD3U_2E3,
    #**DBAUMARTZ_MDRAW_GNSBFD3U_2E4,
    #**DBAUMARTZ_MDRAW_GNSBFD3U_2E5,
    #**DBAUMARTZ_MDRAW_GNSBFD3UAP_2E2,
    #**DBAUMARTZ_MDRAW_GNSBFD3UAP_2E4,
    #**DBAUMARTZ_MDRAW_GNSBFD3UAP_2E6,
    #**DBAUMARTZ_MDRAW_GNSBFD3UAP_2E8,
    #**DBAUMARTZ_MDRAW_GNSBFD3UAP_2E10,
    #**DBAUMARTZ_MDRAW_GNSBFD3E_2E1,
    #**DBAUMARTZ_MDRAW_GNSBFD3E_2E2,
    #**DBAUMARTZ_MDRAW_GNSBFD3E_2E3,
    #**DBAUMARTZ_MDRAW_GNSBFD3E_2E4,
    #**DBAUMARTZ_MDRAW_GNSBFD3E_2E5,
    # **DBAUMARTZ_MDRAW_GNSBFD3E_2E6,
    # **DBAUMARTZ_MDRAW_GNSBFD3E_2E7,
    # **DBAUMARTZ_MDRAW_GNSBFD3E_2E8,
    # **DBAUMARTZ_MDRAW_GNSBFD3E_2E9,
    # **DBAUMARTZ_MDRAW_GNSBFD3E_2E10,
    # **DBAUMARTZ_MDRAW_GNSBFD3E_2E11,
    # **DBAUMARTZ_MDRAW_GNSBFD3E_2E12,
    # **DBAUMARTZ_MDRAW_GNSBFD3E_2E13,
    # **DBAUMARTZ_MDRAW_GNSBFD3E_2E14,
    # **DBAUMARTZ_MDRAW_GNSBFD3E_2E15,
    # **DBAUMARTZ_MDRAW_GNSBFD3E_2E16,
    # **DBAUMARTZ_MDRAW_GNSBFD3E_2E17,
    # **DBAUMARTZ_MDRAW_GNSBFD3E_2E18,
    # **DBAUMARTZ_MDRAW_GNSBFD3E_2E19,
    # **DBAUMARTZ_MDRAW_GNSBFD3E_2E20,
    #**DBAUMARTZ_OLIVERGUHR_GSBFD3_2E1,
    #**DBAUMARTZ_OLIVERGUHR_GSBFD3_2E2,
    #**DBAUMARTZ_OLIVERGUHR_GSBFD3_2E3,
    #**DBAUMARTZ_OLIVERGUHR_GSBFD3_2E4,
    #**DBAUMARTZ_OLIVERGUHR_GSBFD3_2E5,
    #**DBAUMARTZ_OLIVERGUHR_GSBFD3U_2E1,
    #**DBAUMARTZ_OLIVERGUHR_GSBFD3U_2E2,
    #**DBAUMARTZ_OLIVERGUHR_GSBFD3U_2E3,
    #**DBAUMARTZ_OLIVERGUHR_GSBFD3U_2E4,
    #**DBAUMARTZ_OLIVERGUHR_GSBFD3U_2E5,
    #**DBAUMARTZ_OLIVERGUHR_GSBFD3UAP_2E2,
    #**DBAUMARTZ_OLIVERGUHR_GSBFD3UAP_2E4,
    #**DBAUMARTZ_OLIVERGUHR_GSBFD3UAP_2E6,
    #**DBAUMARTZ_OLIVERGUHR_GSBFD3UAP_2E8,
    #**DBAUMARTZ_OLIVERGUHR_GSBFD3UAP_2E10,
    #**DBAUMARTZ_OLIVERGUHR_GSBFD3E_2E1,
    #**DBAUMARTZ_OLIVERGUHR_GSBFD3E_2E2,
    #**DBAUMARTZ_OLIVERGUHR_GSBFD3E_2E3,
    #**DBAUMARTZ_OLIVERGUHR_GSBFD3E_2E4,
    #**DBAUMARTZ_OLIVERGUHR_GSBFD3E_2E5,
    # **DBAUMARTZ_OLIVERGUHR_GSBFD3E_2E6,
    # **DBAUMARTZ_OLIVERGUHR_GSBFD3E_2E7,
    # **DBAUMARTZ_OLIVERGUHR_GSBFD3E_2E8,
    # **DBAUMARTZ_OLIVERGUHR_GSBFD3E_2E9,
    # **DBAUMARTZ_OLIVERGUHR_GSBFD3E_2E10,
    # **DBAUMARTZ_OLIVERGUHR_GSBFD3E_2E11,
    # **DBAUMARTZ_OLIVERGUHR_GSBFD3E_2E12,
    # **DBAUMARTZ_OLIVERGUHR_GSBFD3E_2E13,
    # **DBAUMARTZ_OLIVERGUHR_GSBFD3E_2E14,
    # **DBAUMARTZ_OLIVERGUHR_GSBFD3E_2E15,
    # **DBAUMARTZ_OLIVERGUHR_GSBFD3E_2E16,
    # **DBAUMARTZ_OLIVERGUHR_GSBFD3E_2E17,
    # **DBAUMARTZ_OLIVERGUHR_GSBFD3E_2E18,
    # **DBAUMARTZ_OLIVERGUHR_GSBFD3E_2E19,
    # **DBAUMARTZ_OLIVERGUHR_GSBFD3E_2E20,
    #**DBAUMARTZ_PHILSCHMID_DBMCS2FD3_2E1,
    #**DBAUMARTZ_PHILSCHMID_DBMCS2FD3_2E2,
    #**DBAUMARTZ_PHILSCHMID_DBMCS2FD3_2E3,
    #**DBAUMARTZ_PHILSCHMID_DBMCS2FD3_2E4,
    #**DBAUMARTZ_PHILSCHMID_DBMCS2FD3_2E5,
    #**DBAUMARTZ_PHILSCHMID_DBMCS2FD3U_2E1,
    #**DBAUMARTZ_PHILSCHMID_DBMCS2FD3U_2E2,
    #**DBAUMARTZ_PHILSCHMID_DBMCS2FD3U_2E3,
    #**DBAUMARTZ_PHILSCHMID_DBMCS2FD3U_2E4,
    #**DBAUMARTZ_PHILSCHMID_DBMCS2FD3U_2E5,
    #**DBAUMARTZ_PHILSCHMID_DBMCS2FD3UAP_2E2,
    #**DBAUMARTZ_PHILSCHMID_DBMCS2FD3UAP_2E4,
    #**DBAUMARTZ_PHILSCHMID_DBMCS2FD3UAP_2E6,
    #**DBAUMARTZ_PHILSCHMID_DBMCS2FD3UAP_2E8,
    #**DBAUMARTZ_PHILSCHMID_DBMCS2FD3UAP_2E10,
    #**DBAUMARTZ_PHILSCHMID_DBMCS2FD3E_2E1,
    #**DBAUMARTZ_PHILSCHMID_DBMCS2FD3E_2E2,
    #**DBAUMARTZ_PHILSCHMID_DBMCS2FD3E_2E3,
    #**DBAUMARTZ_PHILSCHMID_DBMCS2FD3E_2E4,
    #**DBAUMARTZ_PHILSCHMID_DBMCS2FD3E_2E5,
    #**DBAUMARTZ_PHILSCHMID_DBMCS2FD3E_2E6,
    #**DBAUMARTZ_PHILSCHMID_DBMCS2FD3E_2E7,
    #**DBAUMARTZ_PHILSCHMID_DBMCS2FD3E_2E8,
    #**DBAUMARTZ_PHILSCHMID_DBMCS2FD3E_2E9,
    #**DBAUMARTZ_PHILSCHMID_DBMCS2FD3E_2E10,
    #**DBAUMARTZ_PHILSCHMID_DBMCS2FD3E_2E11,
    #**DBAUMARTZ_PHILSCHMID_DBMCS2FD3E_2E12,
    #**DBAUMARTZ_PHILSCHMID_DBMCS2FD3E_2E13,
    #**DBAUMARTZ_PHILSCHMID_DBMCS2FD3E_2E14,
    #**DBAUMARTZ_PHILSCHMID_DBMCS2FD3E_2E15,
    #**DBAUMARTZ_PHILSCHMID_DBMCS2FD3E_2E16,
    #**DBAUMARTZ_PHILSCHMID_DBMCS2FD3E_2E17,
    #**DBAUMARTZ_PHILSCHMID_DBMCS2FD3E_2E18,
    #**DBAUMARTZ_PHILSCHMID_DBMCS2FD3E_2E19,
    #**DBAUMARTZ_PHILSCHMID_DBMCS2FD3E_2E20,
}

settings = Settings()
supported_languages = sorted(list(set(chain(*[m["languages"] for m in SUPPORTED_MODELS.values()]))))
lru_cache_with_size = lru_cache(maxsize=settings.textimager_duui_transformers_sentiment_model_cache_size)
model_lock = Lock()

logging.basicConfig(level=settings.textimager_duui_transformers_sentiment_log_level)
logger = logging.getLogger(__name__)
logger.info("TTLab TextImager DUUI Transformers Sentiment")
logger.info("Name: %s", settings.textimager_duui_transformers_sentiment_annotator_name)
logger.info("Version: %s", settings.textimager_duui_transformers_sentiment_annotator_version)

device = 0 if torch.cuda.is_available() else -1
logger.info(f'USING {device}')

typesystem_filename = 'src/main/resources/TypeSystemSentiment.xml'
logger.info("Loading typesystem from \"%s\"", typesystem_filename)
with open(typesystem_filename, 'rb') as f:
    typesystem = load_typesystem(f)
    logger.debug("Base typesystem:")
    logger.debug(typesystem.to_xml())

lua_communication_script_filename = "src/main/lua/textimager_duui_transformers_sentiment.lua"
logger.info("Loading Lua communication script from \"%s\"", lua_communication_script_filename)
with open(lua_communication_script_filename, 'rb') as f:
    lua_communication_script = f.read().decode("utf-8")
    logger.debug("Lua communication script:")
    logger.debug(lua_communication_script)

app = FastAPI(
    openapi_url="/openapi.json",
    docs_url="/api",
    redoc_url=None,
    title=settings.textimager_duui_transformers_sentiment_annotator_name,
    description="Transformers-based sentiment analysis for TTLab TextImager DUUI",
    version=settings.textimager_duui_transformers_sentiment_annotator_version,
    terms_of_service="https://www.texttechnologylab.org/legal_notice/",
    contact={
        "name": "TTLab Team",
        "url": "https://texttechnologylab.org",
        "email": "baumartz@stud.uni-frankfurt.de",
    },
    license_info={
        "name": "AGPL",
        "url": "http://www.gnu.org/licenses/agpl-3.0.en.html",
    },
)


@app.get("/v1/communication_layer", response_class=PlainTextResponse)
def get_communication_layer() -> str:
    return lua_communication_script


@app.get("/v1/documentation")
def get_documentation() -> TextImagerDocumentation:
    capabilities = TextImagerCapability(
        supported_languages=supported_languages,
        reproducible=True
    )

    documentation = TextImagerDocumentation(
        annotator_name=settings.textimager_duui_transformers_sentiment_annotator_name,
        version=settings.textimager_duui_transformers_sentiment_annotator_version,
        implementation_lang="Python",
        meta={
            "python_version": python_version(),
            "python_version_full": sys_version,
            "transformers_version": transformers_version,
            "torch_version": torch.__version__,
        },
        docker_container_id="[TODO]",
        parameters={
            "model_name": SUPPORTED_MODELS,
        },
        capability=capabilities,
        implementation_specific=None,
    )

    return documentation


@app.get("/v1/typesystem")
def get_typesystem() -> Response:
    xml = typesystem.to_xml()
    xml_content = xml.encode("utf-8")

    return Response(
        content=xml_content,
        media_type="application/xml"
    )


def clean_cuda_cache():
    if device >= 0:
        logger.info('emptying cuda cache')
        torch.cuda.empty_cache()
        logger.info('cuda cache empty')


@app.post("/v1/process")
def post_process(request: TextImagerRequest) -> TextImagerResponse:
    processed_selections = []
    meta = None
    modification_meta = None

    clean_cuda_cache()

    dt = datetime.now()

    try:
        modification_timestamp_seconds = int(time())

        logger.debug("Received:")
        logger.debug(request)

        if request.model_name not in SUPPORTED_MODELS:
            raise Exception(f"Model \"{request.model_name}\" is not supported!")

        if request.lang not in SUPPORTED_MODELS[request.model_name]["languages"]:
            raise Exception(f"Document language \"{request.lang}\" is not supported by model \"{request.model_name}\"!")

        logger.info("Using model: \"%s\"", request.model_name)
        model_data = SUPPORTED_MODELS[request.model_name]
        logger.debug(model_data)

        for selection in request.selections:
            processed_sentences = process_selection(request.model_name, model_data, selection, request.doc_len)

            processed_selections.append(
                SentimentSelection(
                    selection=selection.selection,
                    sentences=processed_sentences
                )
            )

        meta = UimaAnnotationMeta(
            name=settings.textimager_duui_transformers_sentiment_annotator_name,
            version=settings.textimager_duui_transformers_sentiment_annotator_version,
            modelName=request.model_name,
            modelVersion=model_data["version"],
        )

        modification_meta_comment = f"{settings.textimager_duui_transformers_sentiment_annotator_name} ({settings.textimager_duui_transformers_sentiment_annotator_version})"
        modification_meta = UimaDocumentModification(
            user="TextImager",
            timestamp=modification_timestamp_seconds,
            comment=modification_meta_comment
        )

    except Exception as ex:
        logger.exception(ex)

    #logger.debug(processed_selections)
    for ps in processed_selections:
        for s in ps.sentences:
            logger.debug(s)

    dte = datetime.now()
    print(dte, 'Finished processing', flush=True)
    print('Time elapsed', f'{dte-dt}', flush=True)

    clean_cuda_cache()

    return TextImagerResponse(
        selections=processed_selections,
        meta=meta,
        modification_meta=modification_meta
    )


@lru_cache_with_size
def load_model(model_name, model_version, labels_count, adapter_path=None):
    mo = model_name
    to = model_name

    # manually load model if:
    # model is local path, not on huggingface hub
    # or adapter is used
    if model_version is None or adapter_path is not None:
        if adapter_path is None:
            from transformers import AutoModelForSequenceClassification
            mo = AutoModelForSequenceClassification.from_pretrained(model_name, revision=model_version, local_files_only=True)
        else:
            from transformers import AutoAdapterModel
            mo = AutoAdapterModel.from_pretrained(model_name, revision=model_version, local_files_only=True)
            adapter_name = mo.load_adapter(adapter_path)
            mo.set_active_adapters(adapter_name)

        to = AutoTokenizer.from_pretrained(model_name, local_files_only=True)

    return pipeline(
        "sentiment-analysis",
        model=mo,
        tokenizer=to,
        revision=model_version,
        top_k=labels_count,
        device=device
    )


def map_sentiment(sentiment_result: List[Dict[str, Union[str, float]]], sentiment_mapping: Dict[str, float], sentiment_polarity: Dict[str, List[str]], sentence: UimaSentence) -> SentimentSentence:
    # get label from top result and map to sentiment values -1, 0 or 1
    sentiment_value = 0.0
    top_result = sentiment_result[0]
    if top_result["label"] in sentiment_mapping:
        sentiment_value = sentiment_mapping[top_result["label"]]

    # get scores of all labels
    details = {
        s["label"]: s["score"]
        for s in sentiment_result
    }

    # calculate polarity: pos-neg
    polarities = {
        "pos": 0,
        "neu": 0,
        "neg": 0
    }
    for p in polarities:
        for l in sentiment_polarity[p]:
            for s in sentiment_result:
                if s["label"] == l:
                    polarities[p] += s["score"]

    polarity = polarities["pos"] - polarities["neg"]

    return SentimentSentence(
        sentence=sentence,
        sentiment=sentiment_value,
        score=top_result["score"],
        details=details,
        polarity=polarity,
        **polarities
    )


def fix_unicode_problems(text):
    # fix emoji in python string and prevent json error on response
    # File "/usr/local/lib/python3.8/site-packages/starlette/responses.py", line 190, in render
    # UnicodeEncodeError: 'utf-8' codec can't encode characters in position xx-yy: surrogates not allowed
    clean_text = text.encode('utf-16', 'surrogatepass').decode('utf-16', 'surrogateescape')
    return clean_text


def process_selection(model_name, model_data, selection, doc_len):
    for s in selection.sentences:
        s.text = fix_unicode_problems(s.text)

    texts = [
        model_data["preprocess"](s.text)
        for s in selection.sentences
    ]
    logger.debug("Preprocessed texts:")
    logger.debug(texts)

    with model_lock:
        model_type = "huggingface" if not "type" in model_data else model_data["type"]
        if model_type == "local":
            sentiment_analysis = load_model(model_data["path"], None, len(model_data["mapping"]))
        elif model_type == "adapter":
            adapter_model_type = "huggingface" if not "type" in model_data else model_data["type"]
            adapter_path = model_data["adapter_path"]
            if adapter_model_type == "local":
                sentiment_analysis = load_model(model_data["model_path"], None, len(model_data["mapping"]), adapter_path)
            else:
                sentiment_analysis = load_model(model_data["model_name"], model_data["model_version"], len(model_data["mapping"]), adapter_path)
        else:
            sentiment_analysis = load_model(model_name, model_data["version"], len(model_data["mapping"]))

        results = sentiment_analysis(
            texts, truncation=True, padding=True, max_length=model_data["max_length"], batch_size=128
        )

    processed_sentences = [
        map_sentiment(r, model_data["mapping"], model_data["3sentiment"], s)
        for s, r
        in zip(selection.sentences, results)
    ]

    if len(results) > 1:
        begin = 0
        end = doc_len

        sentiments = 0
        for sentence in processed_sentences:
            sentiments += sentence.sentiment
        sentiment = sentiments / len(processed_sentences)

        scores = 0
        for sentence in processed_sentences:
            scores += sentence.score
        score = scores / len(processed_sentences)

        details = {}
        for sentence in processed_sentences:
            for d in sentence.details:
                if d not in details:
                    details[d] = 0
                details[d] += sentence.details[d]
        for d in details:
            details[d] = details[d] / len(processed_sentences)

        polaritys = 0
        for sentence in processed_sentences:
            polaritys += sentence.polarity
        polarity = polaritys / len(processed_sentences)

        poss = 0
        for sentence in processed_sentences:
            poss += sentence.pos
        pos = poss / len(processed_sentences)

        neus = 0
        for sentence in processed_sentences:
            neus += sentence.neu
        neu = neus / len(processed_sentences)

        negs = 0
        for sentence in processed_sentences:
            negs += sentence.neg
        neg = negs / len(processed_sentences)

        processed_sentences.append(
            SentimentSentence(
                sentence=UimaSentence(
                    text="",
                    begin=begin,
                    end=end,
                ),
                sentiment=sentiment,
                score=score,
                details=details,
                polarity=polarity,
                pos=pos,
                neu=neu,
                neg=neg
            )
        )

    return processed_sentences
