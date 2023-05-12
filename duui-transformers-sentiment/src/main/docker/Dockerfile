FROM python:3.8

WORKDIR /usr/src/app

EXPOSE 9714

# dependencies
RUN pip install setuptools wheel

# enable different variants, eg for adapter dependencies
ARG TEXTIMAGER_DUUI_TRANSFORMERS_SENTIMENT_VARIANT=""
ENV TEXTIMAGER_DUUI_TRANSFORMERS_SENTIMENT_VARIANT=$TEXTIMAGER_DUUI_TRANSFORMERS_SENTIMENT_VARIANT

COPY ./requirements${TEXTIMAGER_DUUI_TRANSFORMERS_SENTIMENT_VARIANT}.txt ./requirements.txt
RUN pip install --no-deps -r requirements.txt

# preload models during docker build
COPY ./src/main/python/models/preload_models.py ./src/main/python/models/preload_models.py

COPY ./src/main/python/models/cardiffnlp_twitter_roberta_base_sentiment.py ./src/main/python/models/cardiffnlp_twitter_roberta_base_sentiment.py
RUN ["python3", "./src/main/python/models/cardiffnlp_twitter_roberta_base_sentiment.py"]

COPY ./src/main/python/models/cardiffnlp_twitter_roberta_base_sentiment_latest.py ./src/main/python/models/cardiffnlp_twitter_roberta_base_sentiment_latest.py
RUN ["python3", "./src/main/python/models/cardiffnlp_twitter_roberta_base_sentiment_latest.py"]

COPY ./src/main/python/models/cardiffnlp_twitter_xlm_roberta_base_sentiment.py ./src/main/python/models/cardiffnlp_twitter_xlm_roberta_base_sentiment.py
RUN ["python3", "./src/main/python/models/cardiffnlp_twitter_xlm_roberta_base_sentiment.py"]

COPY ./src/main/python/models/oliverguhr_german_sentiment_bert.py ./src/main/python/models/oliverguhr_german_sentiment_bert.py
RUN ["python3", "./src/main/python/models/oliverguhr_german_sentiment_bert.py"]

COPY ./src/main/python/models/mdraw_german_news_sentiment_bert.py ./src/main/python/models/mdraw_german_news_sentiment_bert.py
RUN ["python3", "./src/main/python/models/mdraw_german_news_sentiment_bert.py"]

COPY ./src/main/python/models/nlptown_bert_base_multilingual_uncased_sentiment.py ./src/main/python/models/nlptown_bert_base_multilingual_uncased_sentiment.py
RUN ["python3", "./src/main/python/models/nlptown_bert_base_multilingual_uncased_sentiment.py"]

COPY ./src/main/python/models/finiteautomata_bertweet_base_sentiment_analysis.py ./src/main/python/models/finiteautomata_bertweet_base_sentiment_analysis.py
RUN ["python3", "./src/main/python/models/finiteautomata_bertweet_base_sentiment_analysis.py"]

COPY ./src/main/python/models/siebert_sentiment_roberta_large_english.py ./src/main/python/models/siebert_sentiment_roberta_large_english.py
RUN ["python3", "./src/main/python/models/siebert_sentiment_roberta_large_english.py"]

COPY ./src/main/python/models/j_hartmann_sentiment_roberta_large_english_3_classes.py ./src/main/python/models/j_hartmann_sentiment_roberta_large_english_3_classes.py
RUN ["python3", "./src/main/python/models/j_hartmann_sentiment_roberta_large_english_3_classes.py"]

COPY ./src/main/python/models/liyuan_amazon_review_sentiment_analysis.py ./src/main/python/models/liyuan_amazon_review_sentiment_analysis.py
RUN ["python3", "./src/main/python/models/liyuan_amazon_review_sentiment_analysis.py"]

COPY ./src/main/python/models/philschmid_distilbert_base_multilingual_cased_sentiment_2.py ./src/main/python/models/philschmid_distilbert_base_multilingual_cased_sentiment_2.py
RUN ["python3", "./src/main/python/models/philschmid_distilbert_base_multilingual_cased_sentiment_2.py"]

COPY ./src/main/python/models/cmarkea_distilcamembert_base_sentiment.py ./src/main/python/models/cmarkea_distilcamembert_base_sentiment.py
RUN ["python3", "./src/main/python/models/cmarkea_distilcamembert_base_sentiment.py"]

COPY ./src/main/python/models/clampert_multilingual_sentiment_covid19.py ./src/main/python/models/clampert_multilingual_sentiment_covid19.py
RUN ["python3", "./src/main/python/models/clampert_multilingual_sentiment_covid19.py"]

#COPY ./src/main/python/models/dbaumartz_cardiffnlp_twitter_xlm_roberta_base_sentiment_finetuned_de_3sentiment_2_ep1_cp35057.py ./src/main/python/models/dbaumartz_cardiffnlp_twitter_xlm_roberta_base_sentiment_finetuned_de_3sentiment_2_ep1_cp35057.py
#COPY ./models/cardiffnlp_twitter-xlm-roberta-base-sentiment-finetuned-de-3sentiment/checkpoint-35057 /models/dbaumartz/cardiffnlp_twitter-xlm-roberta-base-sentiment-finetuned-de-3sentiment/checkpoint-35057/
#COPY ./src/main/python/models/dbaumartz_cardiffnlp_twitter_xlm_roberta_base_sentiment_finetuned_de_3sentiment_2_ep2_cp70114.py ./src/main/python/models/dbaumartz_cardiffnlp_twitter_xlm_roberta_base_sentiment_finetuned_de_3sentiment_2_ep2_cp70114.py
#COPY ./models/cardiffnlp_twitter-xlm-roberta-base-sentiment-finetuned-de-3sentiment/checkpoint-70114 /models/dbaumartz/cardiffnlp_twitter-xlm-roberta-base-sentiment-finetuned-de-3sentiment/checkpoint-70114/
#COPY ./src/main/python/models/dbaumartz_cardiffnlp_twitter_xlm_roberta_base_sentiment_finetuned_de_3sentiment_2_ep3_cp105171.py ./src/main/python/models/dbaumartz_cardiffnlp_twitter_xlm_roberta_base_sentiment_finetuned_de_3sentiment_2_ep3_cp105171.py
#COPY ./models/cardiffnlp_twitter-xlm-roberta-base-sentiment-finetuned-de-3sentiment/checkpoint-105171 /models/dbaumartz/cardiffnlp_twitter-xlm-roberta-base-sentiment-finetuned-de-3sentiment/checkpoint-105171/
#COPY ./src/main/python/models/dbaumartz_cardiffnlp_twitter_xlm_roberta_base_sentiment_finetuned_de_3sentiment_2_ep4_cp140228.py ./src/main/python/models/dbaumartz_cardiffnlp_twitter_xlm_roberta_base_sentiment_finetuned_de_3sentiment_2_ep4_cp140228.py
#COPY ./models/cardiffnlp_twitter-xlm-roberta-base-sentiment-finetuned-de-3sentiment/checkpoint-140228 /models/dbaumartz/cardiffnlp_twitter-xlm-roberta-base-sentiment-finetuned-de-3sentiment/checkpoint-140228/
#COPY ./src/main/python/models/dbaumartz_cardiffnlp_twitter_xlm_roberta_base_sentiment_finetuned_de_3sentiment_2_ep5_cp175285.py ./src/main/python/models/dbaumartz_cardiffnlp_twitter_xlm_roberta_base_sentiment_finetuned_de_3sentiment_2_ep5_cp175285.py
#COPY ./models/cardiffnlp_twitter-xlm-roberta-base-sentiment-finetuned-de-3sentiment/checkpoint-175285 /models/dbaumartz/cardiffnlp_twitter-xlm-roberta-base-sentiment-finetuned-de-3sentiment/checkpoint-175285/

#COPY ./src/main/python/models/dbaumartz_mdraw_german_news_sentiment_bert_finetuned_de_3sentiment_2_ep1_cp35057.py ./src/main/python/models/dbaumartz_mdraw_german_news_sentiment_bert_finetuned_de_3sentiment_2_ep1_cp35057.py
#COPY ./models/mdraw_german-news-sentiment-bert-finetuned-de-3sentiment/checkpoint-35057 /models/dbaumartz/mdraw_german-news-sentiment-bert-finetuned-de-3sentiment/checkpoint-35057/
#COPY ./src/main/python/models/dbaumartz_mdraw_german_news_sentiment_bert_finetuned_de_3sentiment_2_ep2_cp70114.py ./src/main/python/models/dbaumartz_mdraw_german_news_sentiment_bert_finetuned_de_3sentiment_2_ep2_cp70114.py
#COPY ./models/mdraw_german-news-sentiment-bert-finetuned-de-3sentiment/checkpoint-70114 /models/dbaumartz/mdraw_german-news-sentiment-bert-finetuned-de-3sentiment/checkpoint-70114/
#COPY ./src/main/python/models/dbaumartz_mdraw_german_news_sentiment_bert_finetuned_de_3sentiment_2_ep3_cp105171.py ./src/main/python/models/dbaumartz_mdraw_german_news_sentiment_bert_finetuned_de_3sentiment_2_ep3_cp105171.py
#COPY ./models/mdraw_german-news-sentiment-bert-finetuned-de-3sentiment/checkpoint-105171 /models/dbaumartz/mdraw_german-news-sentiment-bert-finetuned-de-3sentiment/checkpoint-105171/
#COPY ./src/main/python/models/dbaumartz_mdraw_german_news_sentiment_bert_finetuned_de_3sentiment_2_ep4_cp140228.py ./src/main/python/models/dbaumartz_mdraw_german_news_sentiment_bert_finetuned_de_3sentiment_2_ep4_cp140228.py
#COPY ./models/mdraw_german-news-sentiment-bert-finetuned-de-3sentiment/checkpoint-140228 /models/dbaumartz/mdraw_german-news-sentiment-bert-finetuned-de-3sentiment/checkpoint-140228/
#COPY ./src/main/python/models/dbaumartz_mdraw_german_news_sentiment_bert_finetuned_de_3sentiment_2_ep5_cp175285.py ./src/main/python/models/dbaumartz_mdraw_german_news_sentiment_bert_finetuned_de_3sentiment_2_ep5_cp175285.py
#COPY ./models/mdraw_german-news-sentiment-bert-finetuned-de-3sentiment/checkpoint-175285 /models/dbaumartz/mdraw_german-news-sentiment-bert-finetuned-de-3sentiment/checkpoint-175285/

#COPY ./src/main/python/models/dbaumartz_oliverguhr_german_sentiment_bert_finetuned_de_3sentiment_2_ep1_cp35057.py ./src/main/python/models/dbaumartz_oliverguhr_german_sentiment_bert_finetuned_de_3sentiment_2_ep1_cp35057.py
#COPY ./models/oliverguhr_german-sentiment-bert-finetuned-de-3sentiment/checkpoint-35057 /models/dbaumartz/oliverguhr_german-sentiment-bert-finetuned-de-3sentiment/checkpoint-35057/
#COPY ./src/main/python/models/dbaumartz_oliverguhr_german_sentiment_bert_finetuned_de_3sentiment_2_ep2_cp70114.py ./src/main/python/models/dbaumartz_oliverguhr_german_sentiment_bert_finetuned_de_3sentiment_2_ep2_cp70114.py
#COPY ./models/oliverguhr_german-sentiment-bert-finetuned-de-3sentiment/checkpoint-70114 /models/dbaumartz/oliverguhr_german-sentiment-bert-finetuned-de-3sentiment/checkpoint-70114/
#COPY ./src/main/python/models/dbaumartz_oliverguhr_german_sentiment_bert_finetuned_de_3sentiment_2_ep3_cp105171.py ./src/main/python/models/dbaumartz_oliverguhr_german_sentiment_bert_finetuned_de_3sentiment_2_ep3_cp105171.py
#COPY ./models/oliverguhr_german-sentiment-bert-finetuned-de-3sentiment/checkpoint-105171 /models/dbaumartz/oliverguhr_german-sentiment-bert-finetuned-de-3sentiment/checkpoint-105171/
#COPY ./src/main/python/models/dbaumartz_oliverguhr_german_sentiment_bert_finetuned_de_3sentiment_2_ep4_cp140228.py ./src/main/python/models/dbaumartz_oliverguhr_german_sentiment_bert_finetuned_de_3sentiment_2_ep4_cp140228.py
#COPY ./models/oliverguhr_german-sentiment-bert-finetuned-de-3sentiment/checkpoint-140228 /models/dbaumartz/oliverguhr_german-sentiment-bert-finetuned-de-3sentiment/checkpoint-140228/
#COPY ./src/main/python/models/dbaumartz_oliverguhr_german_sentiment_bert_finetuned_de_3sentiment_2_ep5_cp175285.py ./src/main/python/models/dbaumartz_oliverguhr_german_sentiment_bert_finetuned_de_3sentiment_2_ep5_cp175285.py
#COPY ./models/oliverguhr_german-sentiment-bert-finetuned-de-3sentiment/checkpoint-175285 /models/dbaumartz/oliverguhr_german-sentiment-bert-finetuned-de-3sentiment/checkpoint-175285/

#COPY ./src/main/python/models/dbaumartz_philschmid_distilbert_base_multilingual_cased_sentiment_2_finetuned_de_3sentiment_2_ep1_cp35057.py ./src/main/python/models/dbaumartz_philschmid_distilbert_base_multilingual_cased_sentiment_2_finetuned_de_3sentiment_2_ep1_cp35057.py
#COPY ./models/philschmid_distilbert-base-multilingual-cased-sentiment-2-finetuned-de-3sentiment/checkpoint-35057 /models/dbaumartz/philschmid_distilbert-base-multilingual-cased-sentiment-2-finetuned-de-3sentiment/checkpoint-35057/
#COPY ./src/main/python/models/dbaumartz_philschmid_distilbert_base_multilingual_cased_sentiment_2_finetuned_de_3sentiment_2_ep2_cp70114.py ./src/main/python/models/dbaumartz_philschmid_distilbert_base_multilingual_cased_sentiment_2_finetuned_de_3sentiment_2_ep2_cp70114.py
#COPY ./models/philschmid_distilbert-base-multilingual-cased-sentiment-2-finetuned-de-3sentiment/checkpoint-70114 /models/dbaumartz/philschmid_distilbert-base-multilingual-cased-sentiment-2-finetuned-de-3sentiment/checkpoint-70114/
#COPY ./src/main/python/models/dbaumartz_philschmid_distilbert_base_multilingual_cased_sentiment_2_finetuned_de_3sentiment_2_ep3_cp105171.py ./src/main/python/models/dbaumartz_philschmid_distilbert_base_multilingual_cased_sentiment_2_finetuned_de_3sentiment_2_ep3_cp105171.py
#COPY ./models/philschmid_distilbert-base-multilingual-cased-sentiment-2-finetuned-de-3sentiment/checkpoint-105171 /models/dbaumartz/philschmid_distilbert-base-multilingual-cased-sentiment-2-finetuned-de-3sentiment/checkpoint-105171/
#COPY ./src/main/python/models/dbaumartz_philschmid_distilbert_base_multilingual_cased_sentiment_2_finetuned_de_3sentiment_2_ep4_cp140228.py ./src/main/python/models/dbaumartz_philschmid_distilbert_base_multilingual_cased_sentiment_2_finetuned_de_3sentiment_2_ep4_cp140228.py
#COPY ./models/philschmid_distilbert-base-multilingual-cased-sentiment-2-finetuned-de-3sentiment/checkpoint-140228 /models/dbaumartz/philschmid_distilbert-base-multilingual-cased-sentiment-2-finetuned-de-3sentiment/checkpoint-140228/
#COPY ./src/main/python/models/dbaumartz_philschmid_distilbert_base_multilingual_cased_sentiment_2_finetuned_de_3sentiment_2_ep5_cp175285.py ./src/main/python/models/dbaumartz_philschmid_distilbert_base_multilingual_cased_sentiment_2_finetuned_de_3sentiment_2_ep5_cp175285.py
#COPY ./models/philschmid_distilbert-base-multilingual-cased-sentiment-2-finetuned-de-3sentiment/checkpoint-175285 /models/dbaumartz/philschmid_distilbert-base-multilingual-cased-sentiment-2-finetuned-de-3sentiment/checkpoint-175285/

#COPY ./src/main/python/models/dbaumartz_cardiffnlp_twitter_xlm_roberta_base_sentiment_finetuned_de_3sentiment_2_unseen_ep1_cp35010.py ./src/main/python/models/dbaumartz_cardiffnlp_twitter_xlm_roberta_base_sentiment_finetuned_de_3sentiment_2_unseen_ep1_cp35010.py
#COPY ./models/cardiffnlp_twitter-xlm-roberta-base-sentiment-finetuned-de-3sentiment-unseen/checkpoint-35010 /models/dbaumartz/cardiffnlp_twitter-xlm-roberta-base-sentiment-finetuned-de-3sentiment-unseen/checkpoint-35010/
#COPY ./src/main/python/models/dbaumartz_cardiffnlp_twitter_xlm_roberta_base_sentiment_finetuned_de_3sentiment_2_unseen_ep2_cp70020.py ./src/main/python/models/dbaumartz_cardiffnlp_twitter_xlm_roberta_base_sentiment_finetuned_de_3sentiment_2_unseen_ep2_cp70020.py
#COPY ./models/cardiffnlp_twitter-xlm-roberta-base-sentiment-finetuned-de-3sentiment-unseen/checkpoint-70020 /models/dbaumartz/cardiffnlp_twitter-xlm-roberta-base-sentiment-finetuned-de-3sentiment-unseen/checkpoint-70020/
#COPY ./src/main/python/models/dbaumartz_cardiffnlp_twitter_xlm_roberta_base_sentiment_finetuned_de_3sentiment_2_unseen_ep3_cp105030.py ./src/main/python/models/dbaumartz_cardiffnlp_twitter_xlm_roberta_base_sentiment_finetuned_de_3sentiment_2_unseen_ep3_cp105030.py
#COPY ./models/cardiffnlp_twitter-xlm-roberta-base-sentiment-finetuned-de-3sentiment-unseen/checkpoint-105030 /models/dbaumartz/cardiffnlp_twitter-xlm-roberta-base-sentiment-finetuned-de-3sentiment-unseen/checkpoint-105030/
#COPY ./src/main/python/models/dbaumartz_cardiffnlp_twitter_xlm_roberta_base_sentiment_finetuned_de_3sentiment_2_unseen_ep4_cp140040.py ./src/main/python/models/dbaumartz_cardiffnlp_twitter_xlm_roberta_base_sentiment_finetuned_de_3sentiment_2_unseen_ep4_cp140040.py
#COPY ./models/cardiffnlp_twitter-xlm-roberta-base-sentiment-finetuned-de-3sentiment-unseen/checkpoint-140040 /models/dbaumartz/cardiffnlp_twitter-xlm-roberta-base-sentiment-finetuned-de-3sentiment-unseen/checkpoint-140040/
#COPY ./src/main/python/models/dbaumartz_cardiffnlp_twitter_xlm_roberta_base_sentiment_finetuned_de_3sentiment_2_unseen_ep5_cp175050.py ./src/main/python/models/dbaumartz_cardiffnlp_twitter_xlm_roberta_base_sentiment_finetuned_de_3sentiment_2_unseen_ep5_cp175050.py
#COPY ./models/cardiffnlp_twitter-xlm-roberta-base-sentiment-finetuned-de-3sentiment-unseen/checkpoint-175050 /models/dbaumartz/cardiffnlp_twitter-xlm-roberta-base-sentiment-finetuned-de-3sentiment-unseen/checkpoint-175050/
#
#COPY ./src/main/python/models/dbaumartz_mdraw_german_news_sentiment_bert_finetuned_de_3sentiment_2_unseen_ep1_cp4193.py ./src/main/python/models/dbaumartz_mdraw_german_news_sentiment_bert_finetuned_de_3sentiment_2_unseen_ep1_cp4193.py
#COPY ./models/mdraw_german-news-sentiment-bert-finetuned-de-3sentiment-unseen/checkpoint-4193 /models/dbaumartz/mdraw_german-news-sentiment-bert-finetuned-de-3sentiment-unseen/checkpoint-4193/
#COPY ./src/main/python/models/dbaumartz_mdraw_german_news_sentiment_bert_finetuned_de_3sentiment_2_unseen_ep2_cp8386.py ./src/main/python/models/dbaumartz_mdraw_german_news_sentiment_bert_finetuned_de_3sentiment_2_unseen_ep2_cp8386.py
#COPY ./models/mdraw_german-news-sentiment-bert-finetuned-de-3sentiment-unseen/checkpoint-8386 /models/dbaumartz/mdraw_german-news-sentiment-bert-finetuned-de-3sentiment-unseen/checkpoint-8386/
#COPY ./src/main/python/models/dbaumartz_mdraw_german_news_sentiment_bert_finetuned_de_3sentiment_2_unseen_ep3_cp12579.py ./src/main/python/models/dbaumartz_mdraw_german_news_sentiment_bert_finetuned_de_3sentiment_2_unseen_ep3_cp12579.py
#COPY ./models/mdraw_german-news-sentiment-bert-finetuned-de-3sentiment-unseen/checkpoint-12579 /models/dbaumartz/mdraw_german-news-sentiment-bert-finetuned-de-3sentiment-unseen/checkpoint-12579/
#COPY ./src/main/python/models/dbaumartz_mdraw_german_news_sentiment_bert_finetuned_de_3sentiment_2_unseen_ep4_cp16772.py ./src/main/python/models/dbaumartz_mdraw_german_news_sentiment_bert_finetuned_de_3sentiment_2_unseen_ep4_cp16772.py
#COPY ./models/mdraw_german-news-sentiment-bert-finetuned-de-3sentiment-unseen/checkpoint-16772 /models/dbaumartz/mdraw_german-news-sentiment-bert-finetuned-de-3sentiment-unseen/checkpoint-16772/
#COPY ./src/main/python/models/dbaumartz_mdraw_german_news_sentiment_bert_finetuned_de_3sentiment_2_unseen_ep5_cp20965.py ./src/main/python/models/dbaumartz_mdraw_german_news_sentiment_bert_finetuned_de_3sentiment_2_unseen_ep5_cp20965.py
#COPY ./models/mdraw_german-news-sentiment-bert-finetuned-de-3sentiment-unseen/checkpoint-20965 /models/dbaumartz/mdraw_german-news-sentiment-bert-finetuned-de-3sentiment-unseen/checkpoint-20965/
#
#COPY ./src/main/python/models/dbaumartz_oliverguhr_german_sentiment_bert_finetuned_de_3sentiment_2_unseen_ep1_cp4224.py ./src/main/python/models/dbaumartz_oliverguhr_german_sentiment_bert_finetuned_de_3sentiment_2_unseen_ep1_cp4224.py
#COPY ./models/oliverguhr_german-sentiment-bert-finetuned-de-3sentiment-unseen/checkpoint-4224 /models/dbaumartz/oliverguhr_german-sentiment-bert-finetuned-de-3sentiment-unseen/checkpoint-4224/
#COPY ./src/main/python/models/dbaumartz_oliverguhr_german_sentiment_bert_finetuned_de_3sentiment_2_unseen_ep2_cp8448.py ./src/main/python/models/dbaumartz_oliverguhr_german_sentiment_bert_finetuned_de_3sentiment_2_unseen_ep2_cp8448.py
#COPY ./models/oliverguhr_german-sentiment-bert-finetuned-de-3sentiment-unseen/checkpoint-8448 /models/dbaumartz/oliverguhr_german-sentiment-bert-finetuned-de-3sentiment-unseen/checkpoint-8448/
#COPY ./src/main/python/models/dbaumartz_oliverguhr_german_sentiment_bert_finetuned_de_3sentiment_2_unseen_ep3_cp12672.py ./src/main/python/models/dbaumartz_oliverguhr_german_sentiment_bert_finetuned_de_3sentiment_2_unseen_ep3_cp12672.py
#COPY ./models/oliverguhr_german-sentiment-bert-finetuned-de-3sentiment-unseen/checkpoint-12672 /models/dbaumartz/oliverguhr_german-sentiment-bert-finetuned-de-3sentiment-unseen/checkpoint-12672/
#COPY ./src/main/python/models/dbaumartz_oliverguhr_german_sentiment_bert_finetuned_de_3sentiment_2_unseen_ep4_cp16896.py ./src/main/python/models/dbaumartz_oliverguhr_german_sentiment_bert_finetuned_de_3sentiment_2_unseen_ep4_cp16896.py
#COPY ./models/oliverguhr_german-sentiment-bert-finetuned-de-3sentiment-unseen/checkpoint-16896 /models/dbaumartz/oliverguhr_german-sentiment-bert-finetuned-de-3sentiment-unseen/checkpoint-16896/
#COPY ./src/main/python/models/dbaumartz_oliverguhr_german_sentiment_bert_finetuned_de_3sentiment_2_unseen_ep5_cp21120.py ./src/main/python/models/dbaumartz_oliverguhr_german_sentiment_bert_finetuned_de_3sentiment_2_unseen_ep5_cp21120.py
#COPY ./models/oliverguhr_german-sentiment-bert-finetuned-de-3sentiment-unseen/checkpoint-21120 /models/dbaumartz/oliverguhr_german-sentiment-bert-finetuned-de-3sentiment-unseen/checkpoint-21120/
#
#COPY ./src/main/python/models/dbaumartz_philschmid_distilbert_base_multilingual_cased_sentiment_2_finetuned_de_3sentiment_2_unseen_ep1_cp30870.py ./src/main/python/models/dbaumartz_philschmid_distilbert_base_multilingual_cased_sentiment_2_finetuned_de_3sentiment_2_unseen_ep1_cp30870.py
#COPY ./models/philschmid_distilbert-base-multilingual-cased-sentiment-2-finetuned-de-3sentiment-unseen/checkpoint-30870 /models/dbaumartz/philschmid_distilbert-base-multilingual-cased-sentiment-2-finetuned-de-3sentiment-unseen/checkpoint-30870/
#COPY ./src/main/python/models/dbaumartz_philschmid_distilbert_base_multilingual_cased_sentiment_2_finetuned_de_3sentiment_2_unseen_ep2_cp61740.py ./src/main/python/models/dbaumartz_philschmid_distilbert_base_multilingual_cased_sentiment_2_finetuned_de_3sentiment_2_unseen_ep2_cp61740.py
#COPY ./models/philschmid_distilbert-base-multilingual-cased-sentiment-2-finetuned-de-3sentiment-unseen/checkpoint-61740 /models/dbaumartz/philschmid_distilbert-base-multilingual-cased-sentiment-2-finetuned-de-3sentiment-unseen/checkpoint-61740/
#COPY ./src/main/python/models/dbaumartz_philschmid_distilbert_base_multilingual_cased_sentiment_2_finetuned_de_3sentiment_2_unseen_ep3_cp92610.py ./src/main/python/models/dbaumartz_philschmid_distilbert_base_multilingual_cased_sentiment_2_finetuned_de_3sentiment_2_unseen_ep3_cp92610.py
#COPY ./models/philschmid_distilbert-base-multilingual-cased-sentiment-2-finetuned-de-3sentiment-unseen/checkpoint-92610 /models/dbaumartz/philschmid_distilbert-base-multilingual-cased-sentiment-2-finetuned-de-3sentiment-unseen/checkpoint-92610/
#COPY ./src/main/python/models/dbaumartz_philschmid_distilbert_base_multilingual_cased_sentiment_2_finetuned_de_3sentiment_2_unseen_ep4_cp123480.py ./src/main/python/models/dbaumartz_philschmid_distilbert_base_multilingual_cased_sentiment_2_finetuned_de_3sentiment_2_unseen_ep4_cp123480.py
#COPY ./models/philschmid_distilbert-base-multilingual-cased-sentiment-2-finetuned-de-3sentiment-unseen/checkpoint-123480 /models/dbaumartz/philschmid_distilbert-base-multilingual-cased-sentiment-2-finetuned-de-3sentiment-unseen/checkpoint-123480/
#COPY ./src/main/python/models/dbaumartz_philschmid_distilbert_base_multilingual_cased_sentiment_2_finetuned_de_3sentiment_2_unseen_ep5_cp154350.py ./src/main/python/models/dbaumartz_philschmid_distilbert_base_multilingual_cased_sentiment_2_finetuned_de_3sentiment_2_unseen_ep5_cp154350.py
#COPY ./models/philschmid_distilbert-base-multilingual-cased-sentiment-2-finetuned-de-3sentiment-unseen/checkpoint-154350 /models/dbaumartz/philschmid_distilbert-base-multilingual-cased-sentiment-2-finetuned-de-3sentiment-unseen/checkpoint-154350/

#COPY ./src/main/python/models/dbaumartz_cardiffnlp_twitter_xlm_roberta_base_sentiment_finetuned_de_3sentiment_2_unseen_adapter_pfeiffer_ep2_cp210060.py ./src/main/python/models/dbaumartz_cardiffnlp_twitter_xlm_roberta_base_sentiment_finetuned_de_3sentiment_2_unseen_adapter_pfeiffer_ep2_cp210060.py
#COPY ./models/cardiffnlp_twitter-xlm-roberta-base-sentiment-finetuned-de-3sentiment-unseen-adapter-pfeiffer/checkpoint-210060 /models/dbaumartz/cardiffnlp_twitter-xlm-roberta-base-sentiment-finetuned-de-3sentiment-unseen-adapter-pfeiffer/checkpoint-210060/
#COPY ./src/main/python/models/dbaumartz_cardiffnlp_twitter_xlm_roberta_base_sentiment_finetuned_de_3sentiment_2_unseen_adapter_pfeiffer_ep4_cp420120.py ./src/main/python/models/dbaumartz_cardiffnlp_twitter_xlm_roberta_base_sentiment_finetuned_de_3sentiment_2_unseen_adapter_pfeiffer_ep4_cp420120.py
#COPY ./models/cardiffnlp_twitter-xlm-roberta-base-sentiment-finetuned-de-3sentiment-unseen-adapter-pfeiffer/checkpoint-420120 /models/dbaumartz/cardiffnlp_twitter-xlm-roberta-base-sentiment-finetuned-de-3sentiment-unseen-adapter-pfeiffer/checkpoint-420120/
#COPY ./src/main/python/models/dbaumartz_cardiffnlp_twitter_xlm_roberta_base_sentiment_finetuned_de_3sentiment_2_unseen_adapter_pfeiffer_ep6_cp630180.py ./src/main/python/models/dbaumartz_cardiffnlp_twitter_xlm_roberta_base_sentiment_finetuned_de_3sentiment_2_unseen_adapter_pfeiffer_ep6_cp630180.py
#COPY ./models/cardiffnlp_twitter-xlm-roberta-base-sentiment-finetuned-de-3sentiment-unseen-adapter-pfeiffer/checkpoint-630180 /models/dbaumartz/cardiffnlp_twitter-xlm-roberta-base-sentiment-finetuned-de-3sentiment-unseen-adapter-pfeiffer/checkpoint-630180/
#COPY ./src/main/python/models/dbaumartz_cardiffnlp_twitter_xlm_roberta_base_sentiment_finetuned_de_3sentiment_2_unseen_adapter_pfeiffer_ep8_cp840240.py ./src/main/python/models/dbaumartz_cardiffnlp_twitter_xlm_roberta_base_sentiment_finetuned_de_3sentiment_2_unseen_adapter_pfeiffer_ep8_cp840240.py
#COPY ./models/cardiffnlp_twitter-xlm-roberta-base-sentiment-finetuned-de-3sentiment-unseen-adapter-pfeiffer/checkpoint-840240 /models/dbaumartz/cardiffnlp_twitter-xlm-roberta-base-sentiment-finetuned-de-3sentiment-unseen-adapter-pfeiffer/checkpoint-840240/
#COPY ./src/main/python/models/dbaumartz_cardiffnlp_twitter_xlm_roberta_base_sentiment_finetuned_de_3sentiment_2_unseen_adapter_pfeiffer_ep10_cp1050300.py ./src/main/python/models/dbaumartz_cardiffnlp_twitter_xlm_roberta_base_sentiment_finetuned_de_3sentiment_2_unseen_adapter_pfeiffer_ep10_cp1050300.py
#COPY ./models/cardiffnlp_twitter-xlm-roberta-base-sentiment-finetuned-de-3sentiment-unseen-adapter-pfeiffer/checkpoint-1050300 /models/dbaumartz/cardiffnlp_twitter-xlm-roberta-base-sentiment-finetuned-de-3sentiment-unseen-adapter-pfeiffer/checkpoint-1050300/
#
#COPY ./src/main/python/models/dbaumartz_mdraw_german_news_sentiment_bert_finetuned_de_3sentiment_2_unseen_adapter_pfeiffer_ep2_cp25156.py ./src/main/python/models/dbaumartz_mdraw_german_news_sentiment_bert_finetuned_de_3sentiment_2_unseen_adapter_pfeiffer_ep2_cp25156.py
#COPY ./models/mdraw_german-news-sentiment-bert-finetuned-de-3sentiment-unseen-adapter-pfeiffer/checkpoint-25156 /models/dbaumartz/mdraw_german-news-sentiment-bert-finetuned-de-3sentiment-unseen-adapter-pfeiffer/checkpoint-25156/
#COPY ./src/main/python/models/dbaumartz_mdraw_german_news_sentiment_bert_finetuned_de_3sentiment_2_unseen_adapter_pfeiffer_ep4_cp50312.py ./src/main/python/models/dbaumartz_mdraw_german_news_sentiment_bert_finetuned_de_3sentiment_2_unseen_adapter_pfeiffer_ep4_cp50312.py
#COPY ./models/mdraw_german-news-sentiment-bert-finetuned-de-3sentiment-unseen-adapter-pfeiffer/checkpoint-50312 /models/dbaumartz/mdraw_german-news-sentiment-bert-finetuned-de-3sentiment-unseen-adapter-pfeiffer/checkpoint-50312/
#COPY ./src/main/python/models/dbaumartz_mdraw_german_news_sentiment_bert_finetuned_de_3sentiment_2_unseen_adapter_pfeiffer_ep6_cp75468.py ./src/main/python/models/dbaumartz_mdraw_german_news_sentiment_bert_finetuned_de_3sentiment_2_unseen_adapter_pfeiffer_ep6_cp75468.py
#COPY ./models/mdraw_german-news-sentiment-bert-finetuned-de-3sentiment-unseen-adapter-pfeiffer/checkpoint-75468 /models/dbaumartz/mdraw_german-news-sentiment-bert-finetuned-de-3sentiment-unseen-adapter-pfeiffer/checkpoint-75468/
#COPY ./src/main/python/models/dbaumartz_mdraw_german_news_sentiment_bert_finetuned_de_3sentiment_2_unseen_adapter_pfeiffer_ep8_cp100624.py ./src/main/python/models/dbaumartz_mdraw_german_news_sentiment_bert_finetuned_de_3sentiment_2_unseen_adapter_pfeiffer_ep8_cp100624.py
#COPY ./models/mdraw_german-news-sentiment-bert-finetuned-de-3sentiment-unseen-adapter-pfeiffer/checkpoint-100624 /models/dbaumartz/mdraw_german-news-sentiment-bert-finetuned-de-3sentiment-unseen-adapter-pfeiffer/checkpoint-100624/
#COPY ./src/main/python/models/dbaumartz_mdraw_german_news_sentiment_bert_finetuned_de_3sentiment_2_unseen_adapter_pfeiffer_ep10_cp125780.py ./src/main/python/models/dbaumartz_mdraw_german_news_sentiment_bert_finetuned_de_3sentiment_2_unseen_adapter_pfeiffer_ep10_cp125780.py
#COPY ./models/mdraw_german-news-sentiment-bert-finetuned-de-3sentiment-unseen-adapter-pfeiffer/checkpoint-125780 /models/dbaumartz/mdraw_german-news-sentiment-bert-finetuned-de-3sentiment-unseen-adapter-pfeiffer/checkpoint-125780/
#
#COPY ./src/main/python/models/dbaumartz_oliverguhr_german_sentiment_bert_finetuned_de_3sentiment_2_unseen_adapter_pfeiffer_ep2_cp25342.py ./src/main/python/models/dbaumartz_oliverguhr_german_sentiment_bert_finetuned_de_3sentiment_2_unseen_adapter_pfeiffer_ep2_cp25342.py
#COPY ./models/oliverguhr_german-sentiment-bert-finetuned-de-3sentiment-unseen-adapter-pfeiffer/checkpoint-25342 /models/dbaumartz/oliverguhr_german-sentiment-bert-finetuned-de-3sentiment-unseen-adapter-pfeiffer/checkpoint-25342/
#COPY ./src/main/python/models/dbaumartz_oliverguhr_german_sentiment_bert_finetuned_de_3sentiment_2_unseen_adapter_pfeiffer_ep4_cp50684.py ./src/main/python/models/dbaumartz_oliverguhr_german_sentiment_bert_finetuned_de_3sentiment_2_unseen_adapter_pfeiffer_ep4_cp50684.py
#COPY ./models/oliverguhr_german-sentiment-bert-finetuned-de-3sentiment-unseen-adapter-pfeiffer/checkpoint-50684 /models/dbaumartz/oliverguhr_german-sentiment-bert-finetuned-de-3sentiment-unseen-adapter-pfeiffer/checkpoint-50684/
#COPY ./src/main/python/models/dbaumartz_oliverguhr_german_sentiment_bert_finetuned_de_3sentiment_2_unseen_adapter_pfeiffer_ep6_cp76026.py ./src/main/python/models/dbaumartz_oliverguhr_german_sentiment_bert_finetuned_de_3sentiment_2_unseen_adapter_pfeiffer_ep6_cp76026.py
#COPY ./models/oliverguhr_german-sentiment-bert-finetuned-de-3sentiment-unseen-adapter-pfeiffer/checkpoint-76026 /models/dbaumartz/oliverguhr_german-sentiment-bert-finetuned-de-3sentiment-unseen-adapter-pfeiffer/checkpoint-76026/
#COPY ./src/main/python/models/dbaumartz_oliverguhr_german_sentiment_bert_finetuned_de_3sentiment_2_unseen_adapter_pfeiffer_ep8_cp101368.py ./src/main/python/models/dbaumartz_oliverguhr_german_sentiment_bert_finetuned_de_3sentiment_2_unseen_adapter_pfeiffer_ep8_cp101368.py
#COPY ./models/oliverguhr_german-sentiment-bert-finetuned-de-3sentiment-unseen-adapter-pfeiffer/checkpoint-101368 /models/dbaumartz/oliverguhr_german-sentiment-bert-finetuned-de-3sentiment-unseen-adapter-pfeiffer/checkpoint-101368/
#COPY ./src/main/python/models/dbaumartz_oliverguhr_german_sentiment_bert_finetuned_de_3sentiment_2_unseen_adapter_pfeiffer_ep10_cp126710.py ./src/main/python/models/dbaumartz_oliverguhr_german_sentiment_bert_finetuned_de_3sentiment_2_unseen_adapter_pfeiffer_ep10_cp126710.py
#COPY ./models/oliverguhr_german-sentiment-bert-finetuned-de-3sentiment-unseen-adapter-pfeiffer/checkpoint-126710 /models/dbaumartz/oliverguhr_german-sentiment-bert-finetuned-de-3sentiment-unseen-adapter-pfeiffer/checkpoint-126710/
#
#COPY ./src/main/python/models/dbaumartz_philschmid_distilbert_base_multilingual_cased_sentiment_2_finetuned_de_3sentiment_2_unseen_adapter_pfeiffer_ep2_cp185216.py ./src/main/python/models/dbaumartz_philschmid_distilbert_base_multilingual_cased_sentiment_2_finetuned_de_3sentiment_2_unseen_adapter_pfeiffer_ep2_cp185216.py
#COPY ./models/philschmid_distilbert-base-multilingual-cased-sentiment-2-finetuned-de-3sentiment-unseen-adapter-pfeiffer/checkpoint-185216 /models/dbaumartz/philschmid_distilbert-base-multilingual-cased-sentiment-2-finetuned-de-3sentiment-unseen-adapter-pfeiffer/checkpoint-185216/
#COPY ./src/main/python/models/dbaumartz_philschmid_distilbert_base_multilingual_cased_sentiment_2_finetuned_de_3sentiment_2_unseen_adapter_pfeiffer_ep4_cp370432.py ./src/main/python/models/dbaumartz_philschmid_distilbert_base_multilingual_cased_sentiment_2_finetuned_de_3sentiment_2_unseen_adapter_pfeiffer_ep4_cp370432.py
#COPY ./models/philschmid_distilbert-base-multilingual-cased-sentiment-2-finetuned-de-3sentiment-unseen-adapter-pfeiffer/checkpoint-370432 /models/dbaumartz/philschmid_distilbert-base-multilingual-cased-sentiment-2-finetuned-de-3sentiment-unseen-adapter-pfeiffer/checkpoint-370432/
#COPY ./src/main/python/models/dbaumartz_philschmid_distilbert_base_multilingual_cased_sentiment_2_finetuned_de_3sentiment_2_unseen_adapter_pfeiffer_ep6_cp555648.py ./src/main/python/models/dbaumartz_philschmid_distilbert_base_multilingual_cased_sentiment_2_finetuned_de_3sentiment_2_unseen_adapter_pfeiffer_ep6_cp555648.py
#COPY ./models/philschmid_distilbert-base-multilingual-cased-sentiment-2-finetuned-de-3sentiment-unseen-adapter-pfeiffer/checkpoint-555648 /models/dbaumartz/philschmid_distilbert-base-multilingual-cased-sentiment-2-finetuned-de-3sentiment-unseen-adapter-pfeiffer/checkpoint-555648/
#COPY ./src/main/python/models/dbaumartz_philschmid_distilbert_base_multilingual_cased_sentiment_2_finetuned_de_3sentiment_2_unseen_adapter_pfeiffer_ep8_cp740864.py ./src/main/python/models/dbaumartz_philschmid_distilbert_base_multilingual_cased_sentiment_2_finetuned_de_3sentiment_2_unseen_adapter_pfeiffer_ep8_cp740864.py
#COPY ./models/philschmid_distilbert-base-multilingual-cased-sentiment-2-finetuned-de-3sentiment-unseen-adapter-pfeiffer/checkpoint-740864 /models/dbaumartz/philschmid_distilbert-base-multilingual-cased-sentiment-2-finetuned-de-3sentiment-unseen-adapter-pfeiffer/checkpoint-740864/
#COPY ./src/main/python/models/dbaumartz_philschmid_distilbert_base_multilingual_cased_sentiment_2_finetuned_de_3sentiment_2_unseen_adapter_pfeiffer_ep10_cp926080.py ./src/main/python/models/dbaumartz_philschmid_distilbert_base_multilingual_cased_sentiment_2_finetuned_de_3sentiment_2_unseen_adapter_pfeiffer_ep10_cp926080.py
#COPY ./models/philschmid_distilbert-base-multilingual-cased-sentiment-2-finetuned-de-3sentiment-unseen-adapter-pfeiffer/checkpoint-926080 /models/dbaumartz/philschmid_distilbert-base-multilingual-cased-sentiment-2-finetuned-de-3sentiment-unseen-adapter-pfeiffer/checkpoint-926080/

#COPY ./src/main/python/models/dbaumartz_oliverguhr_german_sentiment_bert_finetuned_de_3sentiment_2_exact_ep1_cp30979.py ./src/main/python/models/dbaumartz_oliverguhr_german_sentiment_bert_finetuned_de_3sentiment_2_exact_ep1_cp30979.py
#COPY ./models/oliverguhr_german-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-30979 /models/dbaumartz/oliverguhr_german-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-30979/
#COPY ./src/main/python/models/dbaumartz_oliverguhr_german_sentiment_bert_finetuned_de_3sentiment_2_exact_ep2_cp61958.py ./src/main/python/models/dbaumartz_oliverguhr_german_sentiment_bert_finetuned_de_3sentiment_2_exact_ep2_cp61958.py
#COPY ./models/oliverguhr_german-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-61958 /models/dbaumartz/oliverguhr_german-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-61958/
#COPY ./src/main/python/models/dbaumartz_oliverguhr_german_sentiment_bert_finetuned_de_3sentiment_2_exact_ep3_cp92937.py ./src/main/python/models/dbaumartz_oliverguhr_german_sentiment_bert_finetuned_de_3sentiment_2_exact_ep3_cp92937.py
#COPY ./models/oliverguhr_german-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-92937 /models/dbaumartz/oliverguhr_german-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-92937/
#COPY ./src/main/python/models/dbaumartz_oliverguhr_german_sentiment_bert_finetuned_de_3sentiment_2_exact_ep4_cp123916.py ./src/main/python/models/dbaumartz_oliverguhr_german_sentiment_bert_finetuned_de_3sentiment_2_exact_ep4_cp123916.py
#COPY ./models/oliverguhr_german-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-123916 /models/dbaumartz/oliverguhr_german-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-123916/
#COPY ./src/main/python/models/dbaumartz_oliverguhr_german_sentiment_bert_finetuned_de_3sentiment_2_exact_ep5_cp154895.py ./src/main/python/models/dbaumartz_oliverguhr_german_sentiment_bert_finetuned_de_3sentiment_2_exact_ep5_cp154895.py
#COPY ./models/oliverguhr_german-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-154895 /models/dbaumartz/oliverguhr_german-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-154895/

#COPY ./src/main/python/models/dbaumartz_philschmid_distilbert_base_multilingual_cased_sentiment_2_finetuned_de_3sentiment_2_exact_ep1_cp30979.py ./src/main/python/models/dbaumartz_philschmid_distilbert_base_multilingual_cased_sentiment_2_finetuned_de_3sentiment_2_exact_ep1_cp30979.py
#COPY ./models/philschmid_distilbert-base-multilingual-cased-sentiment-2-finetuned-de-3sentiment-exact/checkpoint-30979 /models/dbaumartz/philschmid_distilbert-base-multilingual-cased-sentiment-2-finetuned-de-3sentiment-exact/checkpoint-30979/
#COPY ./src/main/python/models/dbaumartz_philschmid_distilbert_base_multilingual_cased_sentiment_2_finetuned_de_3sentiment_2_exact_ep2_cp61958.py ./src/main/python/models/dbaumartz_philschmid_distilbert_base_multilingual_cased_sentiment_2_finetuned_de_3sentiment_2_exact_ep2_cp61958.py
#COPY ./models/philschmid_distilbert-base-multilingual-cased-sentiment-2-finetuned-de-3sentiment-exact/checkpoint-61958 /models/dbaumartz/philschmid_distilbert-base-multilingual-cased-sentiment-2-finetuned-de-3sentiment-exact/checkpoint-61958/
#COPY ./src/main/python/models/dbaumartz_philschmid_distilbert_base_multilingual_cased_sentiment_2_finetuned_de_3sentiment_2_exact_ep3_cp92937.py ./src/main/python/models/dbaumartz_philschmid_distilbert_base_multilingual_cased_sentiment_2_finetuned_de_3sentiment_2_exact_ep3_cp92937.py
#COPY ./models/philschmid_distilbert-base-multilingual-cased-sentiment-2-finetuned-de-3sentiment-exact/checkpoint-92937 /models/dbaumartz/philschmid_distilbert-base-multilingual-cased-sentiment-2-finetuned-de-3sentiment-exact/checkpoint-92937/
#COPY ./src/main/python/models/dbaumartz_philschmid_distilbert_base_multilingual_cased_sentiment_2_finetuned_de_3sentiment_2_exact_ep4_cp123916.py ./src/main/python/models/dbaumartz_philschmid_distilbert_base_multilingual_cased_sentiment_2_finetuned_de_3sentiment_2_exact_ep4_cp123916.py
#COPY ./models/philschmid_distilbert-base-multilingual-cased-sentiment-2-finetuned-de-3sentiment-exact/checkpoint-123916 /models/dbaumartz/philschmid_distilbert-base-multilingual-cased-sentiment-2-finetuned-de-3sentiment-exact/checkpoint-123916/
#COPY ./src/main/python/models/dbaumartz_philschmid_distilbert_base_multilingual_cased_sentiment_2_finetuned_de_3sentiment_2_exact_ep5_cp154895.py ./src/main/python/models/dbaumartz_philschmid_distilbert_base_multilingual_cased_sentiment_2_finetuned_de_3sentiment_2_exact_ep5_cp154895.py
#COPY ./models/philschmid_distilbert-base-multilingual-cased-sentiment-2-finetuned-de-3sentiment-exact/checkpoint-154895 /models/dbaumartz/philschmid_distilbert-base-multilingual-cased-sentiment-2-finetuned-de-3sentiment-exact/checkpoint-154895/

#COPY ./src/main/python/models/dbaumartz_cardiffnlp_twitter_xlm_roberta_base_sentiment_finetuned_de_3sentiment_2_exact_ep1_cp30979.py ./src/main/python/models/dbaumartz_cardiffnlp_twitter_xlm_roberta_base_sentiment_finetuned_de_3sentiment_2_exact_ep1_cp30979.py
#COPY ./models/cardiffnlp_twitter-xlm-roberta-base-sentiment-finetuned-de-3sentiment-exact/checkpoint-30979 /models/dbaumartz/cardiffnlp_twitter-xlm-roberta-base-sentiment-finetuned-de-3sentiment-exact/checkpoint-30979/
#COPY ./src/main/python/models/dbaumartz_cardiffnlp_twitter_xlm_roberta_base_sentiment_finetuned_de_3sentiment_2_exact_ep2_cp61958.py ./src/main/python/models/dbaumartz_cardiffnlp_twitter_xlm_roberta_base_sentiment_finetuned_de_3sentiment_2_exact_ep2_cp61958.py
#COPY ./models/cardiffnlp_twitter-xlm-roberta-base-sentiment-finetuned-de-3sentiment-exact/checkpoint-61958 /models/dbaumartz/cardiffnlp_twitter-xlm-roberta-base-sentiment-finetuned-de-3sentiment-exact/checkpoint-61958/
#COPY ./src/main/python/models/dbaumartz_cardiffnlp_twitter_xlm_roberta_base_sentiment_finetuned_de_3sentiment_2_exact_ep3_cp92937.py ./src/main/python/models/dbaumartz_cardiffnlp_twitter_xlm_roberta_base_sentiment_finetuned_de_3sentiment_2_exact_ep3_cp92937.py
#COPY ./models/cardiffnlp_twitter-xlm-roberta-base-sentiment-finetuned-de-3sentiment-exact/checkpoint-92937 /models/dbaumartz/cardiffnlp_twitter-xlm-roberta-base-sentiment-finetuned-de-3sentiment-exact/checkpoint-92937/
#COPY ./src/main/python/models/dbaumartz_cardiffnlp_twitter_xlm_roberta_base_sentiment_finetuned_de_3sentiment_2_exact_ep4_cp123916.py ./src/main/python/models/dbaumartz_cardiffnlp_twitter_xlm_roberta_base_sentiment_finetuned_de_3sentiment_2_exact_ep4_cp123916.py
#COPY ./models/cardiffnlp_twitter-xlm-roberta-base-sentiment-finetuned-de-3sentiment-exact/checkpoint-123916 /models/dbaumartz/cardiffnlp_twitter-xlm-roberta-base-sentiment-finetuned-de-3sentiment-exact/checkpoint-123916/
#COPY ./src/main/python/models/dbaumartz_cardiffnlp_twitter_xlm_roberta_base_sentiment_finetuned_de_3sentiment_2_exact_ep5_cp154895.py ./src/main/python/models/dbaumartz_cardiffnlp_twitter_xlm_roberta_base_sentiment_finetuned_de_3sentiment_2_exact_ep5_cp154895.py
#COPY ./models/cardiffnlp_twitter-xlm-roberta-base-sentiment-finetuned-de-3sentiment-exact/checkpoint-154895 /models/dbaumartz/cardiffnlp_twitter-xlm-roberta-base-sentiment-finetuned-de-3sentiment-exact/checkpoint-154895/

#COPY ./src/main/python/models/dbaumartz_mdraw_german_news_sentiment_bert_finetuned_de_3sentiment_2_exact_ep1_cp30979.py ./src/main/python/models/dbaumartz_mdraw_german_news_sentiment_bert_finetuned_de_3sentiment_2_exact_ep1_cp30979.py
#COPY ./models/mdraw_german-news-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-30979 /models/dbaumartz/mdraw_german-news-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-30979/
#COPY ./src/main/python/models/dbaumartz_mdraw_german_news_sentiment_bert_finetuned_de_3sentiment_2_exact_ep2_cp61958.py ./src/main/python/models/dbaumartz_mdraw_german_news_sentiment_bert_finetuned_de_3sentiment_2_exact_ep2_cp61958.py
#COPY ./models/mdraw_german-news-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-61958 /models/dbaumartz/mdraw_german-news-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-61958/
#COPY ./src/main/python/models/dbaumartz_mdraw_german_news_sentiment_bert_finetuned_de_3sentiment_2_exact_ep3_cp92937.py ./src/main/python/models/dbaumartz_mdraw_german_news_sentiment_bert_finetuned_de_3sentiment_2_exact_ep3_cp92937.py
#COPY ./models/mdraw_german-news-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-92937 /models/dbaumartz/mdraw_german-news-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-92937/
#COPY ./src/main/python/models/dbaumartz_mdraw_german_news_sentiment_bert_finetuned_de_3sentiment_2_exact_ep4_cp123916.py ./src/main/python/models/dbaumartz_mdraw_german_news_sentiment_bert_finetuned_de_3sentiment_2_exact_ep4_cp123916.py
#COPY ./models/mdraw_german-news-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-123916 /models/dbaumartz/mdraw_german-news-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-123916/
#COPY ./src/main/python/models/dbaumartz_mdraw_german_news_sentiment_bert_finetuned_de_3sentiment_2_exact_ep5_cp154895.py ./src/main/python/models/dbaumartz_mdraw_german_news_sentiment_bert_finetuned_de_3sentiment_2_exact_ep5_cp154895.py
#COPY ./models/mdraw_german-news-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-154895 /models/dbaumartz/mdraw_german-news-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-154895/

#COPY ./src/main/python/models/dbaumartz_cardiffnlp_twitter_xlm_roberta_base_sentiment_finetuned_de_3sentiment_2_exact_ep6_cp185874.py ./src/main/python/models/dbaumartz_cardiffnlp_twitter_xlm_roberta_base_sentiment_finetuned_de_3sentiment_2_exact_ep6_cp185874.py
#COPY ./models/cardiffnlp_twitter-xlm-roberta-base-sentiment-finetuned-de-3sentiment-exact/checkpoint-185874 /models/dbaumartz/cardiffnlp_twitter-xlm-roberta-base-sentiment-finetuned-de-3sentiment-exact/checkpoint-185874/
#COPY ./src/main/python/models/dbaumartz_cardiffnlp_twitter_xlm_roberta_base_sentiment_finetuned_de_3sentiment_2_exact_ep7_cp216853.py ./src/main/python/models/dbaumartz_cardiffnlp_twitter_xlm_roberta_base_sentiment_finetuned_de_3sentiment_2_exact_ep7_cp216853.py
#COPY ./models/cardiffnlp_twitter-xlm-roberta-base-sentiment-finetuned-de-3sentiment-exact/checkpoint-216853 /models/dbaumartz/cardiffnlp_twitter-xlm-roberta-base-sentiment-finetuned-de-3sentiment-exact/checkpoint-216853/
#COPY ./src/main/python/models/dbaumartz_cardiffnlp_twitter_xlm_roberta_base_sentiment_finetuned_de_3sentiment_2_exact_ep8_cp247832.py ./src/main/python/models/dbaumartz_cardiffnlp_twitter_xlm_roberta_base_sentiment_finetuned_de_3sentiment_2_exact_ep8_cp247832.py
#COPY ./models/cardiffnlp_twitter-xlm-roberta-base-sentiment-finetuned-de-3sentiment-exact/checkpoint-247832 /models/dbaumartz/cardiffnlp_twitter-xlm-roberta-base-sentiment-finetuned-de-3sentiment-exact/checkpoint-247832/
#COPY ./src/main/python/models/dbaumartz_cardiffnlp_twitter_xlm_roberta_base_sentiment_finetuned_de_3sentiment_2_exact_ep9_cp278811.py ./src/main/python/models/dbaumartz_cardiffnlp_twitter_xlm_roberta_base_sentiment_finetuned_de_3sentiment_2_exact_ep9_cp278811.py
#COPY ./models/cardiffnlp_twitter-xlm-roberta-base-sentiment-finetuned-de-3sentiment-exact/checkpoint-278811 /models/dbaumartz/cardiffnlp_twitter-xlm-roberta-base-sentiment-finetuned-de-3sentiment-exact/checkpoint-278811/
#COPY ./src/main/python/models/dbaumartz_cardiffnlp_twitter_xlm_roberta_base_sentiment_finetuned_de_3sentiment_2_exact_ep10_cp309790.py ./src/main/python/models/dbaumartz_cardiffnlp_twitter_xlm_roberta_base_sentiment_finetuned_de_3sentiment_2_exact_ep10_cp309790.py
#COPY ./models/cardiffnlp_twitter-xlm-roberta-base-sentiment-finetuned-de-3sentiment-exact/checkpoint-309790 /models/dbaumartz/cardiffnlp_twitter-xlm-roberta-base-sentiment-finetuned-de-3sentiment-exact/checkpoint-309790/
#COPY ./src/main/python/models/dbaumartz_cardiffnlp_twitter_xlm_roberta_base_sentiment_finetuned_de_3sentiment_2_exact_ep11_cp340769.py ./src/main/python/models/dbaumartz_cardiffnlp_twitter_xlm_roberta_base_sentiment_finetuned_de_3sentiment_2_exact_ep11_cp340769.py
#COPY ./models/cardiffnlp_twitter-xlm-roberta-base-sentiment-finetuned-de-3sentiment-exact/checkpoint-340769 /models/dbaumartz/cardiffnlp_twitter-xlm-roberta-base-sentiment-finetuned-de-3sentiment-exact/checkpoint-340769/
#COPY ./src/main/python/models/dbaumartz_cardiffnlp_twitter_xlm_roberta_base_sentiment_finetuned_de_3sentiment_2_exact_ep12_cp371748.py ./src/main/python/models/dbaumartz_cardiffnlp_twitter_xlm_roberta_base_sentiment_finetuned_de_3sentiment_2_exact_ep12_cp371748.py
#COPY ./models/cardiffnlp_twitter-xlm-roberta-base-sentiment-finetuned-de-3sentiment-exact/checkpoint-371748 /models/dbaumartz/cardiffnlp_twitter-xlm-roberta-base-sentiment-finetuned-de-3sentiment-exact/checkpoint-371748/
#COPY ./src/main/python/models/dbaumartz_cardiffnlp_twitter_xlm_roberta_base_sentiment_finetuned_de_3sentiment_2_exact_ep13_cp402727.py ./src/main/python/models/dbaumartz_cardiffnlp_twitter_xlm_roberta_base_sentiment_finetuned_de_3sentiment_2_exact_ep13_cp402727.py
#COPY ./models/cardiffnlp_twitter-xlm-roberta-base-sentiment-finetuned-de-3sentiment-exact/checkpoint-402727 /models/dbaumartz/cardiffnlp_twitter-xlm-roberta-base-sentiment-finetuned-de-3sentiment-exact/checkpoint-402727/
#COPY ./src/main/python/models/dbaumartz_cardiffnlp_twitter_xlm_roberta_base_sentiment_finetuned_de_3sentiment_2_exact_ep14_cp433706.py ./src/main/python/models/dbaumartz_cardiffnlp_twitter_xlm_roberta_base_sentiment_finetuned_de_3sentiment_2_exact_ep14_cp433706.py
#COPY ./models/cardiffnlp_twitter-xlm-roberta-base-sentiment-finetuned-de-3sentiment-exact/checkpoint-433706 /models/dbaumartz/cardiffnlp_twitter-xlm-roberta-base-sentiment-finetuned-de-3sentiment-exact/checkpoint-433706/
#COPY ./src/main/python/models/dbaumartz_cardiffnlp_twitter_xlm_roberta_base_sentiment_finetuned_de_3sentiment_2_exact_ep15_cp464685.py ./src/main/python/models/dbaumartz_cardiffnlp_twitter_xlm_roberta_base_sentiment_finetuned_de_3sentiment_2_exact_ep15_cp464685.py
#COPY ./models/cardiffnlp_twitter-xlm-roberta-base-sentiment-finetuned-de-3sentiment-exact/checkpoint-464685 /models/dbaumartz/cardiffnlp_twitter-xlm-roberta-base-sentiment-finetuned-de-3sentiment-exact/checkpoint-464685/
#COPY ./src/main/python/models/dbaumartz_cardiffnlp_twitter_xlm_roberta_base_sentiment_finetuned_de_3sentiment_2_exact_ep16_cp495664.py ./src/main/python/models/dbaumartz_cardiffnlp_twitter_xlm_roberta_base_sentiment_finetuned_de_3sentiment_2_exact_ep16_cp495664.py
#COPY ./models/cardiffnlp_twitter-xlm-roberta-base-sentiment-finetuned-de-3sentiment-exact/checkpoint-495664 /models/dbaumartz/cardiffnlp_twitter-xlm-roberta-base-sentiment-finetuned-de-3sentiment-exact/checkpoint-495664/
#COPY ./src/main/python/models/dbaumartz_cardiffnlp_twitter_xlm_roberta_base_sentiment_finetuned_de_3sentiment_2_exact_ep17_cp526643.py ./src/main/python/models/dbaumartz_cardiffnlp_twitter_xlm_roberta_base_sentiment_finetuned_de_3sentiment_2_exact_ep17_cp526643.py
#COPY ./models/cardiffnlp_twitter-xlm-roberta-base-sentiment-finetuned-de-3sentiment-exact/checkpoint-526643 /models/dbaumartz/cardiffnlp_twitter-xlm-roberta-base-sentiment-finetuned-de-3sentiment-exact/checkpoint-526643/
#COPY ./src/main/python/models/dbaumartz_cardiffnlp_twitter_xlm_roberta_base_sentiment_finetuned_de_3sentiment_2_exact_ep18_cp557622.py ./src/main/python/models/dbaumartz_cardiffnlp_twitter_xlm_roberta_base_sentiment_finetuned_de_3sentiment_2_exact_ep18_cp557622.py
#COPY ./models/cardiffnlp_twitter-xlm-roberta-base-sentiment-finetuned-de-3sentiment-exact/checkpoint-557622 /models/dbaumartz/cardiffnlp_twitter-xlm-roberta-base-sentiment-finetuned-de-3sentiment-exact/checkpoint-557622/
#COPY ./src/main/python/models/dbaumartz_cardiffnlp_twitter_xlm_roberta_base_sentiment_finetuned_de_3sentiment_2_exact_ep19_cp588601.py ./src/main/python/models/dbaumartz_cardiffnlp_twitter_xlm_roberta_base_sentiment_finetuned_de_3sentiment_2_exact_ep19_cp588601.py
#COPY ./models/cardiffnlp_twitter-xlm-roberta-base-sentiment-finetuned-de-3sentiment-exact/checkpoint-588601 /models/dbaumartz/cardiffnlp_twitter-xlm-roberta-base-sentiment-finetuned-de-3sentiment-exact/checkpoint-588601/
#COPY ./src/main/python/models/dbaumartz_cardiffnlp_twitter_xlm_roberta_base_sentiment_finetuned_de_3sentiment_2_exact_ep20_cp619580.py ./src/main/python/models/dbaumartz_cardiffnlp_twitter_xlm_roberta_base_sentiment_finetuned_de_3sentiment_2_exact_ep20_cp619580.py
#COPY ./models/cardiffnlp_twitter-xlm-roberta-base-sentiment-finetuned-de-3sentiment-exact/checkpoint-619580 /models/dbaumartz/cardiffnlp_twitter-xlm-roberta-base-sentiment-finetuned-de-3sentiment-exact/checkpoint-619580/

#COPY ./src/main/python/models/dbaumartz_mdraw_german_news_sentiment_bert_finetuned_de_3sentiment_2_exact_ep6_cp185874.py ./src/main/python/models/dbaumartz_mdraw_german_news_sentiment_bert_finetuned_de_3sentiment_2_exact_ep6_cp185874.py
#COPY ./models/mdraw_german-news-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-185874 /models/dbaumartz/mdraw_german-news-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-185874/
#COPY ./src/main/python/models/dbaumartz_mdraw_german_news_sentiment_bert_finetuned_de_3sentiment_2_exact_ep7_cp216853.py ./src/main/python/models/dbaumartz_mdraw_german_news_sentiment_bert_finetuned_de_3sentiment_2_exact_ep7_cp216853.py
#COPY ./models/mdraw_german-news-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-216853 /models/dbaumartz/mdraw_german-news-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-216853/
#COPY ./src/main/python/models/dbaumartz_mdraw_german_news_sentiment_bert_finetuned_de_3sentiment_2_exact_ep8_cp247832.py ./src/main/python/models/dbaumartz_mdraw_german_news_sentiment_bert_finetuned_de_3sentiment_2_exact_ep8_cp247832.py
#COPY ./models/mdraw_german-news-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-247832 /models/dbaumartz/mdraw_german-news-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-247832/
#COPY ./src/main/python/models/dbaumartz_mdraw_german_news_sentiment_bert_finetuned_de_3sentiment_2_exact_ep9_cp278811.py ./src/main/python/models/dbaumartz_mdraw_german_news_sentiment_bert_finetuned_de_3sentiment_2_exact_ep9_cp278811.py
#COPY ./models/mdraw_german-news-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-278811 /models/dbaumartz/mdraw_german-news-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-278811/
#COPY ./src/main/python/models/dbaumartz_mdraw_german_news_sentiment_bert_finetuned_de_3sentiment_2_exact_ep10_cp309790.py ./src/main/python/models/dbaumartz_mdraw_german_news_sentiment_bert_finetuned_de_3sentiment_2_exact_ep10_cp309790.py
#COPY ./models/mdraw_german-news-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-309790 /models/dbaumartz/mdraw_german-news-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-309790/
#COPY ./src/main/python/models/dbaumartz_mdraw_german_news_sentiment_bert_finetuned_de_3sentiment_2_exact_ep11_cp340769.py ./src/main/python/models/dbaumartz_mdraw_german_news_sentiment_bert_finetuned_de_3sentiment_2_exact_ep11_cp340769.py
#COPY ./models/mdraw_german-news-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-340769 /models/dbaumartz/mdraw_german-news-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-340769/
#COPY ./src/main/python/models/dbaumartz_mdraw_german_news_sentiment_bert_finetuned_de_3sentiment_2_exact_ep12_cp371748.py ./src/main/python/models/dbaumartz_mdraw_german_news_sentiment_bert_finetuned_de_3sentiment_2_exact_ep12_cp371748.py
#COPY ./models/mdraw_german-news-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-371748 /models/dbaumartz/mdraw_german-news-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-371748/
#COPY ./src/main/python/models/dbaumartz_mdraw_german_news_sentiment_bert_finetuned_de_3sentiment_2_exact_ep13_cp402727.py ./src/main/python/models/dbaumartz_mdraw_german_news_sentiment_bert_finetuned_de_3sentiment_2_exact_ep13_cp402727.py
#COPY ./models/mdraw_german-news-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-402727 /models/dbaumartz/mdraw_german-news-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-402727/
#COPY ./src/main/python/models/dbaumartz_mdraw_german_news_sentiment_bert_finetuned_de_3sentiment_2_exact_ep14_cp433706.py ./src/main/python/models/dbaumartz_mdraw_german_news_sentiment_bert_finetuned_de_3sentiment_2_exact_ep14_cp433706.py
#COPY ./models/mdraw_german-news-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-433706 /models/dbaumartz/mdraw_german-news-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-433706/
#COPY ./src/main/python/models/dbaumartz_mdraw_german_news_sentiment_bert_finetuned_de_3sentiment_2_exact_ep15_cp464685.py ./src/main/python/models/dbaumartz_mdraw_german_news_sentiment_bert_finetuned_de_3sentiment_2_exact_ep15_cp464685.py
#COPY ./models/mdraw_german-news-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-464685 /models/dbaumartz/mdraw_german-news-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-464685/
#COPY ./src/main/python/models/dbaumartz_mdraw_german_news_sentiment_bert_finetuned_de_3sentiment_2_exact_ep16_cp495664.py ./src/main/python/models/dbaumartz_mdraw_german_news_sentiment_bert_finetuned_de_3sentiment_2_exact_ep16_cp495664.py
#COPY ./models/mdraw_german-news-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-495664 /models/dbaumartz/mdraw_german-news-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-495664/
#COPY ./src/main/python/models/dbaumartz_mdraw_german_news_sentiment_bert_finetuned_de_3sentiment_2_exact_ep17_cp526643.py ./src/main/python/models/dbaumartz_mdraw_german_news_sentiment_bert_finetuned_de_3sentiment_2_exact_ep17_cp526643.py
#COPY ./models/mdraw_german-news-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-526643 /models/dbaumartz/mdraw_german-news-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-526643/
#COPY ./src/main/python/models/dbaumartz_mdraw_german_news_sentiment_bert_finetuned_de_3sentiment_2_exact_ep18_cp557622.py ./src/main/python/models/dbaumartz_mdraw_german_news_sentiment_bert_finetuned_de_3sentiment_2_exact_ep18_cp557622.py
#COPY ./models/mdraw_german-news-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-557622 /models/dbaumartz/mdraw_german-news-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-557622/
#COPY ./src/main/python/models/dbaumartz_mdraw_german_news_sentiment_bert_finetuned_de_3sentiment_2_exact_ep19_cp588601.py ./src/main/python/models/dbaumartz_mdraw_german_news_sentiment_bert_finetuned_de_3sentiment_2_exact_ep19_cp588601.py
#COPY ./models/mdraw_german-news-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-588601 /models/dbaumartz/mdraw_german-news-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-588601/
#COPY ./src/main/python/models/dbaumartz_mdraw_german_news_sentiment_bert_finetuned_de_3sentiment_2_exact_ep20_cp619580.py ./src/main/python/models/dbaumartz_mdraw_german_news_sentiment_bert_finetuned_de_3sentiment_2_exact_ep20_cp619580.py
#COPY ./models/mdraw_german-news-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-619580 /models/dbaumartz/mdraw_german-news-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-619580/

#COPY ./src/main/python/models/dbaumartz_oliverguhr_german_sentiment_bert_finetuned_de_3sentiment_2_exact_ep6_cp185874.py ./src/main/python/models/dbaumartz_oliverguhr_german_sentiment_bert_finetuned_de_3sentiment_2_exact_ep6_cp185874.py
#COPY ./models/oliverguhr_german-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-185874 /models/dbaumartz/oliverguhr_german-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-185874/
#COPY ./src/main/python/models/dbaumartz_oliverguhr_german_sentiment_bert_finetuned_de_3sentiment_2_exact_ep7_cp216853.py ./src/main/python/models/dbaumartz_oliverguhr_german_sentiment_bert_finetuned_de_3sentiment_2_exact_ep7_cp216853.py
#COPY ./models/oliverguhr_german-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-216853 /models/dbaumartz/oliverguhr_german-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-216853/
#COPY ./src/main/python/models/dbaumartz_oliverguhr_german_sentiment_bert_finetuned_de_3sentiment_2_exact_ep8_cp247832.py ./src/main/python/models/dbaumartz_oliverguhr_german_sentiment_bert_finetuned_de_3sentiment_2_exact_ep8_cp247832.py
#COPY ./models/oliverguhr_german-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-247832 /models/dbaumartz/oliverguhr_german-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-247832/
#COPY ./src/main/python/models/dbaumartz_oliverguhr_german_sentiment_bert_finetuned_de_3sentiment_2_exact_ep9_cp278811.py ./src/main/python/models/dbaumartz_oliverguhr_german_sentiment_bert_finetuned_de_3sentiment_2_exact_ep9_cp278811.py
#COPY ./models/oliverguhr_german-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-278811 /models/dbaumartz/oliverguhr_german-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-278811/
#COPY ./src/main/python/models/dbaumartz_oliverguhr_german_sentiment_bert_finetuned_de_3sentiment_2_exact_ep10_cp309790.py ./src/main/python/models/dbaumartz_oliverguhr_german_sentiment_bert_finetuned_de_3sentiment_2_exact_ep10_cp309790.py
#COPY ./models/oliverguhr_german-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-309790 /models/dbaumartz/oliverguhr_german-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-309790/
#COPY ./src/main/python/models/dbaumartz_oliverguhr_german_sentiment_bert_finetuned_de_3sentiment_2_exact_ep11_cp340769.py ./src/main/python/models/dbaumartz_oliverguhr_german_sentiment_bert_finetuned_de_3sentiment_2_exact_ep11_cp340769.py
#COPY ./models/oliverguhr_german-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-340769 /models/dbaumartz/oliverguhr_german-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-340769/
#COPY ./src/main/python/models/dbaumartz_oliverguhr_german_sentiment_bert_finetuned_de_3sentiment_2_exact_ep12_cp371748.py ./src/main/python/models/dbaumartz_oliverguhr_german_sentiment_bert_finetuned_de_3sentiment_2_exact_ep12_cp371748.py
#COPY ./models/oliverguhr_german-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-371748 /models/dbaumartz/oliverguhr_german-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-371748/
#COPY ./src/main/python/models/dbaumartz_oliverguhr_german_sentiment_bert_finetuned_de_3sentiment_2_exact_ep13_cp402727.py ./src/main/python/models/dbaumartz_oliverguhr_german_sentiment_bert_finetuned_de_3sentiment_2_exact_ep13_cp402727.py
#COPY ./models/oliverguhr_german-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-402727 /models/dbaumartz/oliverguhr_german-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-402727/
#COPY ./src/main/python/models/dbaumartz_oliverguhr_german_sentiment_bert_finetuned_de_3sentiment_2_exact_ep14_cp433706.py ./src/main/python/models/dbaumartz_oliverguhr_german_sentiment_bert_finetuned_de_3sentiment_2_exact_ep14_cp433706.py
#COPY ./models/oliverguhr_german-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-433706 /models/dbaumartz/oliverguhr_german-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-433706/
#COPY ./src/main/python/models/dbaumartz_oliverguhr_german_sentiment_bert_finetuned_de_3sentiment_2_exact_ep15_cp464685.py ./src/main/python/models/dbaumartz_oliverguhr_german_sentiment_bert_finetuned_de_3sentiment_2_exact_ep15_cp464685.py
#COPY ./models/oliverguhr_german-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-464685 /models/dbaumartz/oliverguhr_german-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-464685/
#COPY ./src/main/python/models/dbaumartz_oliverguhr_german_sentiment_bert_finetuned_de_3sentiment_2_exact_ep16_cp495664.py ./src/main/python/models/dbaumartz_oliverguhr_german_sentiment_bert_finetuned_de_3sentiment_2_exact_ep16_cp495664.py
#COPY ./models/oliverguhr_german-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-495664 /models/dbaumartz/oliverguhr_german-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-495664/
#COPY ./src/main/python/models/dbaumartz_oliverguhr_german_sentiment_bert_finetuned_de_3sentiment_2_exact_ep17_cp526643.py ./src/main/python/models/dbaumartz_oliverguhr_german_sentiment_bert_finetuned_de_3sentiment_2_exact_ep17_cp526643.py
#COPY ./models/oliverguhr_german-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-526643 /models/dbaumartz/oliverguhr_german-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-526643/
#COPY ./src/main/python/models/dbaumartz_oliverguhr_german_sentiment_bert_finetuned_de_3sentiment_2_exact_ep18_cp557622.py ./src/main/python/models/dbaumartz_oliverguhr_german_sentiment_bert_finetuned_de_3sentiment_2_exact_ep18_cp557622.py
#COPY ./models/oliverguhr_german-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-557622 /models/dbaumartz/oliverguhr_german-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-557622/
#COPY ./src/main/python/models/dbaumartz_oliverguhr_german_sentiment_bert_finetuned_de_3sentiment_2_exact_ep19_cp588601.py ./src/main/python/models/dbaumartz_oliverguhr_german_sentiment_bert_finetuned_de_3sentiment_2_exact_ep19_cp588601.py
#COPY ./models/oliverguhr_german-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-588601 /models/dbaumartz/oliverguhr_german-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-588601/
#COPY ./src/main/python/models/dbaumartz_oliverguhr_german_sentiment_bert_finetuned_de_3sentiment_2_exact_ep20_cp619580.py ./src/main/python/models/dbaumartz_oliverguhr_german_sentiment_bert_finetuned_de_3sentiment_2_exact_ep20_cp619580.py
#COPY ./models/oliverguhr_german-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-619580 /models/dbaumartz/oliverguhr_german-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-619580/

#COPY ./src/main/python/models/dbaumartz_philschmid_distilbert_base_multilingual_cased_sentiment_2_finetuned_de_3sentiment_2_exact_ep6_cp185874.py ./src/main/python/models/dbaumartz_philschmid_distilbert_base_multilingual_cased_sentiment_2_finetuned_de_3sentiment_2_exact_ep6_cp185874.py
#COPY ./models/philschmid_distilbert-base-multilingual-cased-sentiment-2-finetuned-de-3sentiment-exact/checkpoint-185874 /models/dbaumartz/philschmid_distilbert-base-multilingual-cased-sentiment-2-finetuned-de-3sentiment-exact/checkpoint-185874/
#COPY ./src/main/python/models/dbaumartz_philschmid_distilbert_base_multilingual_cased_sentiment_2_finetuned_de_3sentiment_2_exact_ep7_cp216853.py ./src/main/python/models/dbaumartz_philschmid_distilbert_base_multilingual_cased_sentiment_2_finetuned_de_3sentiment_2_exact_ep7_cp216853.py
#COPY ./models/philschmid_distilbert-base-multilingual-cased-sentiment-2-finetuned-de-3sentiment-exact/checkpoint-216853 /models/dbaumartz/philschmid_distilbert-base-multilingual-cased-sentiment-2-finetuned-de-3sentiment-exact/checkpoint-216853/
#COPY ./src/main/python/models/dbaumartz_philschmid_distilbert_base_multilingual_cased_sentiment_2_finetuned_de_3sentiment_2_exact_ep8_cp247832.py ./src/main/python/models/dbaumartz_philschmid_distilbert_base_multilingual_cased_sentiment_2_finetuned_de_3sentiment_2_exact_ep8_cp247832.py
#COPY ./models/philschmid_distilbert-base-multilingual-cased-sentiment-2-finetuned-de-3sentiment-exact/checkpoint-247832 /models/dbaumartz/philschmid_distilbert-base-multilingual-cased-sentiment-2-finetuned-de-3sentiment-exact/checkpoint-247832/
#COPY ./src/main/python/models/dbaumartz_philschmid_distilbert_base_multilingual_cased_sentiment_2_finetuned_de_3sentiment_2_exact_ep9_cp278811.py ./src/main/python/models/dbaumartz_philschmid_distilbert_base_multilingual_cased_sentiment_2_finetuned_de_3sentiment_2_exact_ep9_cp278811.py
#COPY ./models/philschmid_distilbert-base-multilingual-cased-sentiment-2-finetuned-de-3sentiment-exact/checkpoint-278811 /models/dbaumartz/philschmid_distilbert-base-multilingual-cased-sentiment-2-finetuned-de-3sentiment-exact/checkpoint-278811/
#COPY ./src/main/python/models/dbaumartz_philschmid_distilbert_base_multilingual_cased_sentiment_2_finetuned_de_3sentiment_2_exact_ep10_cp309790.py ./src/main/python/models/dbaumartz_philschmid_distilbert_base_multilingual_cased_sentiment_2_finetuned_de_3sentiment_2_exact_ep10_cp309790.py
#COPY ./models/philschmid_distilbert-base-multilingual-cased-sentiment-2-finetuned-de-3sentiment-exact/checkpoint-309790 /models/dbaumartz/philschmid_distilbert-base-multilingual-cased-sentiment-2-finetuned-de-3sentiment-exact/checkpoint-309790/
#COPY ./src/main/python/models/dbaumartz_philschmid_distilbert_base_multilingual_cased_sentiment_2_finetuned_de_3sentiment_2_exact_ep11_cp340769.py ./src/main/python/models/dbaumartz_philschmid_distilbert_base_multilingual_cased_sentiment_2_finetuned_de_3sentiment_2_exact_ep11_cp340769.py
#COPY ./models/philschmid_distilbert-base-multilingual-cased-sentiment-2-finetuned-de-3sentiment-exact/checkpoint-340769 /models/dbaumartz/philschmid_distilbert-base-multilingual-cased-sentiment-2-finetuned-de-3sentiment-exact/checkpoint-340769/
#COPY ./src/main/python/models/dbaumartz_philschmid_distilbert_base_multilingual_cased_sentiment_2_finetuned_de_3sentiment_2_exact_ep12_cp371748.py ./src/main/python/models/dbaumartz_philschmid_distilbert_base_multilingual_cased_sentiment_2_finetuned_de_3sentiment_2_exact_ep12_cp371748.py
#COPY ./models/philschmid_distilbert-base-multilingual-cased-sentiment-2-finetuned-de-3sentiment-exact/checkpoint-371748 /models/dbaumartz/philschmid_distilbert-base-multilingual-cased-sentiment-2-finetuned-de-3sentiment-exact/checkpoint-371748/
#COPY ./src/main/python/models/dbaumartz_philschmid_distilbert_base_multilingual_cased_sentiment_2_finetuned_de_3sentiment_2_exact_ep13_cp402727.py ./src/main/python/models/dbaumartz_philschmid_distilbert_base_multilingual_cased_sentiment_2_finetuned_de_3sentiment_2_exact_ep13_cp402727.py
#COPY ./models/philschmid_distilbert-base-multilingual-cased-sentiment-2-finetuned-de-3sentiment-exact/checkpoint-402727 /models/dbaumartz/philschmid_distilbert-base-multilingual-cased-sentiment-2-finetuned-de-3sentiment-exact/checkpoint-402727/
#COPY ./src/main/python/models/dbaumartz_philschmid_distilbert_base_multilingual_cased_sentiment_2_finetuned_de_3sentiment_2_exact_ep14_cp433706.py ./src/main/python/models/dbaumartz_philschmid_distilbert_base_multilingual_cased_sentiment_2_finetuned_de_3sentiment_2_exact_ep14_cp433706.py
#COPY ./models/philschmid_distilbert-base-multilingual-cased-sentiment-2-finetuned-de-3sentiment-exact/checkpoint-433706 /models/dbaumartz/philschmid_distilbert-base-multilingual-cased-sentiment-2-finetuned-de-3sentiment-exact/checkpoint-433706/
#COPY ./src/main/python/models/dbaumartz_philschmid_distilbert_base_multilingual_cased_sentiment_2_finetuned_de_3sentiment_2_exact_ep15_cp464685.py ./src/main/python/models/dbaumartz_philschmid_distilbert_base_multilingual_cased_sentiment_2_finetuned_de_3sentiment_2_exact_ep15_cp464685.py
#COPY ./models/philschmid_distilbert-base-multilingual-cased-sentiment-2-finetuned-de-3sentiment-exact/checkpoint-464685 /models/dbaumartz/philschmid_distilbert-base-multilingual-cased-sentiment-2-finetuned-de-3sentiment-exact/checkpoint-464685/
#COPY ./src/main/python/models/dbaumartz_philschmid_distilbert_base_multilingual_cased_sentiment_2_finetuned_de_3sentiment_2_exact_ep16_cp495664.py ./src/main/python/models/dbaumartz_philschmid_distilbert_base_multilingual_cased_sentiment_2_finetuned_de_3sentiment_2_exact_ep16_cp495664.py
#COPY ./models/philschmid_distilbert-base-multilingual-cased-sentiment-2-finetuned-de-3sentiment-exact/checkpoint-495664 /models/dbaumartz/philschmid_distilbert-base-multilingual-cased-sentiment-2-finetuned-de-3sentiment-exact/checkpoint-495664/
#COPY ./src/main/python/models/dbaumartz_philschmid_distilbert_base_multilingual_cased_sentiment_2_finetuned_de_3sentiment_2_exact_ep17_cp526643.py ./src/main/python/models/dbaumartz_philschmid_distilbert_base_multilingual_cased_sentiment_2_finetuned_de_3sentiment_2_exact_ep17_cp526643.py
#COPY ./models/philschmid_distilbert-base-multilingual-cased-sentiment-2-finetuned-de-3sentiment-exact/checkpoint-526643 /models/dbaumartz/philschmid_distilbert-base-multilingual-cased-sentiment-2-finetuned-de-3sentiment-exact/checkpoint-526643/
#COPY ./src/main/python/models/dbaumartz_philschmid_distilbert_base_multilingual_cased_sentiment_2_finetuned_de_3sentiment_2_exact_ep18_cp557622.py ./src/main/python/models/dbaumartz_philschmid_distilbert_base_multilingual_cased_sentiment_2_finetuned_de_3sentiment_2_exact_ep18_cp557622.py
#COPY ./models/philschmid_distilbert-base-multilingual-cased-sentiment-2-finetuned-de-3sentiment-exact/checkpoint-557622 /models/dbaumartz/philschmid_distilbert-base-multilingual-cased-sentiment-2-finetuned-de-3sentiment-exact/checkpoint-557622/
#COPY ./src/main/python/models/dbaumartz_philschmid_distilbert_base_multilingual_cased_sentiment_2_finetuned_de_3sentiment_2_exact_ep19_cp588601.py ./src/main/python/models/dbaumartz_philschmid_distilbert_base_multilingual_cased_sentiment_2_finetuned_de_3sentiment_2_exact_ep19_cp588601.py
#COPY ./models/philschmid_distilbert-base-multilingual-cased-sentiment-2-finetuned-de-3sentiment-exact/checkpoint-588601 /models/dbaumartz/philschmid_distilbert-base-multilingual-cased-sentiment-2-finetuned-de-3sentiment-exact/checkpoint-588601/
#COPY ./src/main/python/models/dbaumartz_philschmid_distilbert_base_multilingual_cased_sentiment_2_finetuned_de_3sentiment_2_exact_ep20_cp619580.py ./src/main/python/models/dbaumartz_philschmid_distilbert_base_multilingual_cased_sentiment_2_finetuned_de_3sentiment_2_exact_ep20_cp619580.py
#COPY ./models/philschmid_distilbert-base-multilingual-cased-sentiment-2-finetuned-de-3sentiment-exact/checkpoint-619580 /models/dbaumartz/philschmid_distilbert-base-multilingual-cased-sentiment-2-finetuned-de-3sentiment-exact/checkpoint-619580/

# log level
ARG TEXTIMAGER_DUUI_TRANSFORMERS_SENTIMENT_LOG_LEVEL="DEBUG"
ENV TEXTIMAGER_DUUI_TRANSFORMERS_SENTIMENT_LOG_LEVEL=$TEXTIMAGER_DUUI_TRANSFORMERS_SENTIMENT_LOG_LEVEL

# config
ARG TEXTIMAGER_DUUI_TRANSFORMERS_SENTIMENT_MODEL_CACHE_SIZE=1
ENV TEXTIMAGER_DUUI_TRANSFORMERS_SENTIMENT_MODEL_CACHE_SIZE=$TEXTIMAGER_DUUI_TRANSFORMERS_SENTIMENT_MODEL_CACHE_SIZE

# meta data
ARG TEXTIMAGER_DUUI_TRANSFORMERS_SENTIMENT_ANNOTATOR_NAME="textimager-duui-transformers-sentiment"
ENV TEXTIMAGER_DUUI_TRANSFORMERS_SENTIMENT_ANNOTATOR_NAME=$TEXTIMAGER_DUUI_TRANSFORMERS_SENTIMENT_ANNOTATOR_NAME
ARG TEXTIMAGER_DUUI_TRANSFORMERS_SENTIMENT_ANNOTATOR_VERSION="unset"
ENV TEXTIMAGER_DUUI_TRANSFORMERS_SENTIMENT_ANNOTATOR_VERSION=$TEXTIMAGER_DUUI_TRANSFORMERS_SENTIMENT_ANNOTATOR_VERSION

# offline mode for huggingface
ARG TEXTIMAGER_DUUI_TRANSFORMERS_SENTIMENT_TRANSFORMERS_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=$TEXTIMAGER_DUUI_TRANSFORMERS_SENTIMENT_TRANSFORMERS_OFFLINE

# copy scripts
COPY ./src/main/resources/TypeSystemSentiment.xml ./src/main/resources/TypeSystemSentiment.xml
COPY ./src/main/python/__init__.py ./src/main/python/__init__.py
COPY ./src/main/python/duui/ ./src/main/python/duui/
COPY ./src/main/python/textimager_duui_transformers_sentiment.py ./src/main/python/textimager_duui_transformers_sentiment.py
COPY ./src/main/lua/textimager_duui_transformers_sentiment.lua ./src/main/lua/textimager_duui_transformers_sentiment.lua

ENTRYPOINT ["uvicorn", "src.main.python.textimager_duui_transformers_sentiment:app", "--host", "0.0.0.0", "--port" ,"9714"]
CMD ["--workers", "1"]
