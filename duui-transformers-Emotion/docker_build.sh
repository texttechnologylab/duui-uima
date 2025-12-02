#!/usr/bin/env bash
set -euo pipefail

#export ANNOTATOR_CUDA=
export ANNOTATOR_CUDA="-cuda"

export ANNOTATOR_NAME=duui-transformers-emotion
export ANNOTATOR_VERSION=0.3.0
export LOG_LEVEL=DEBUG
export MODEL_CACHE_SIZE=3
export DOCKER_REGISTRY="docker.texttechnologylab.org/"

####---------------------------------------------------------------------
#export MODEL_NAME="02shanky/finetuned-twitter-xlm-roberta-base-emotion"
#export MODEL_SPECNAME="finetuned-twitter-xlm-roberta-base-emotion"
#export MODEL_VERSION="28e6d080e9f73171b574dd88ac768da9e6622c36"
#export MODEL_SOURCE="https://huggingface.co/02shanky/finetuned-twitter-xlm-roberta-base-emotion"
#export MODEL_LANG="Multi"
####--------------------------------------------------------------------

####---------------------------------------------------------------------
#export MODEL_NAME="DReAMy-lib/xlm-roberta-large-DreamBank-emotion-presence"
#export MODEL_SPECNAME="dreamy-xlm-roberta-emotion"
#export MODEL_VERSION="b3487623ec2dd4b9bd0644d8266291afb9956e9f"
#export MODEL_SOURCE="https://huggingface.co/DReAMy-lib/xlm-roberta-large-DreamBank-emotion-presence"
#export MODEL_LANG="Multi"
####--------------------------------------------------------------------

####---------------------------------------------------------------------
#export MODEL_NAME="MilaNLProc/xlm-emo-t"
#export MODEL_SPECNAME="xlm-emo-t"
#export MODEL_VERSION="a6ee7c9fad08d60204e7ae437d41d392381496f0"
#export MODEL_SOURCE="https://huggingface.co/MilaNLProc/xlm-emo-t"
#export MODEL_LANG="Multi"
####--------------------------------------------------------------------


####---------------------------------------------------------------------
#export MODEL_NAME="j-hartmann/emotion-english-distilroberta-base"
#export MODEL_SPECNAME="emotion-english-distilroberta-base"
#export MODEL_VERSION="0e1cd914e3d46199ed785853e12b57304e04178b"
#export MODEL_SOURCE="https://huggingface.co/j-hartmann/emotion-english-distilroberta-base"
#export MODEL_LANG="EN"
####--------------------------------------------------------------------


####---------------------------------------------------------------------
#export MODEL_NAME="michellejieli/emotion_text_classifier"
#export MODEL_SPECNAME="emotion_text_classifier"
#export MODEL_VERSION="dc4df5597fcda82589511c3900fedbe1c0ffec82"
#export MODEL_SOURCE="https://huggingface.co/michellejieli/emotion_text_classifier"
#export MODEL_LANG="EN"
####--------------------------------------------------------------------

####---------------------------------------------------------------------
#export MODEL_NAME="cardiffnlp/twitter-roberta-base-emotion"
#export MODEL_SPECNAME="cardiffnlp-twitter-roberta-base-emotion"
#export MODEL_VERSION="2848306ad936b7cd47c76c2c4e14d694a41e0f54"
#export MODEL_SOURCE="https://huggingface.co/cardiffnlp/twitter-roberta-base-emotion"
#export MODEL_LANG="EN"
####--------------------------------------------------------------------

####---------------------------------------------------------------------
#export MODEL_NAME="finiteautomata/bertweet-base-emotion-analysis"
#export MODEL_SPECNAME="bertweet-base-emotion-analysis"
#export MODEL_VERSION="c482c9e1750a29dcc393234816bcf468ff77cd2d	"
#export MODEL_SOURCE="https://huggingface.co/finiteautomata/bertweet-base-emotion-analysis"
#export MODEL_LANG="EN"
####--------------------------------------------------------------------

####---------------------------------------------------------------------
#export MODEL_NAME="ActivationAI/distilbert-base-uncased-finetuned-emotion"
#export MODEL_SPECNAME="distilbert-base-uncased-finetuned-emotion"
#export MODEL_VERSION="dbf4470880ff3b73f22975241cd309bdf8e2195f"
#export MODEL_SOURCE="https://huggingface.co/ActivationAI/distilbert-base-uncased-finetuned-emotion"
#export MODEL_LANG="EN"
####--------------------------------------------------------------------

###---------------------------------------------------------------------
export MODEL_NAME="SamLowe/roberta-base-go_emotions"
export MODEL_SPECNAME="roberta-base-go-emotions"
export MODEL_VERSION="58b6c5b44a7a12093f782442969019c7e2982299"
export MODEL_SOURCE="https://huggingface.co/SamLowe/roberta-base-go_emotions"
export MODEL_LANG="EN"
###--------------------------------------------------------------------
#
####---------------------------------------------------------------------
#export MODEL_NAME="mrm8488/t5-base-finetuned-emotion"
#export MODEL_SPECNAME="t5-base-finetuned-emotion"
#export MODEL_VERSION="e44a316825f11230724b36412fbf1899c76e82de"
#export MODEL_SOURCE="https://huggingface.co/mrm8488/t5-base-finetuned-emotion"
#export MODEL_LANG="EN"
####--------------------------------------------------------------------


####---------------------------------------------------------------------
#export MODEL_NAME="pysentimiento"
#export MODEL_SPECNAME="pysentimiento"
#export MODEL_VERSION="60822acfd805ad5d95437c695daa33c18dbda060"
#export MODEL_SOURCE="https://github.com/pysentimiento/pysentimiento/"
#export MODEL_LANG="EN, ES, IT, PT"
####--------------------------------------------------------------------


####---------------------------------------------------------------------
#export MODEL_NAME="EmoAtlas"
#export MODEL_SPECNAME="emoatlas"
#export MODEL_VERSION="adae44a80dd55c1d1c467c4e72bdb2d8cf63bf28"
#export MODEL_SOURCE="https://github.com/alfonsosemeraro/emoatlas"
#export MODEL_LANG="EN"
####--------------------------------------------------------------------

###---------------------------------------------------------------------
#export MODEL_NAME="pol_emo_mDeBERTa"
#export MODEL_SPECNAME="pol_emo_mdeberta"
#export MODEL_VERSION="523da7dc2523631787ef0712bad53bfe2ac46840"
#export MODEL_SOURCE="https://github.com/tweedmann/pol_emo_mDeBERTa2"
#export MODEL_LANG="Multi"
###--------------------------------------------------------------------

###---------------------------------------------------------------------
#export MODEL_NAME="pranaydeeps/EXALT-Baseline"
#export MODEL_SPECNAME="exalt-baseline"
#export MODEL_VERSION="4b5e2a38b4e72823c428891170aec8930f580bad"
#export MODEL_SOURCE="https://huggingface.co/pranaydeeps/EXALT-Baseline"
#export MODEL_LANG="Multi"
###--------------------------------------------------------------------

###---------------------------------------------------------------------
#export MODEL_NAME="boltuix/bert-emotion"
#export MODEL_SPECNAME="bert-emotion"
#export MODEL_VERSION="00b4ef11958dd607b2ede29f6ed6d02338782c94"
#export MODEL_SOURCE="https://huggingface.co/boltuix/bert-emotion"
#export MODEL_LANG="EN"
###--------------------------------------------------------------------


###---------------------------------------------------------------------
#export MODEL_NAME="MilaNLProc/feel-it-italian-emotion"
#export MODEL_SPECNAME="feel-it"
#export MODEL_VERSION="6efdabf62230414aeba764986b4ae317ce7c5c47"
#export MODEL_SOURCE="https://huggingface.co/MilaNLProc/feel-it-italian-emotion"
#export MODEL_LANG="IT"
###--------------------------------------------------------------------


###---------------------------------------------------------------------
#export MODEL_NAME="cardiffnlp/twitter-roberta-base-emotion-multilabel-latest"
#export MODEL_SPECNAME="cardiffnlp-multilabel"
#export MODEL_VERSION="30a56d88e47e493f08f93c786d49c526550b55b9"
#export MODEL_SOURCE="https://huggingface.co/cardiffnlp/twitter-roberta-base-emotion-multilabel-latest"
#export MODEL_LANG="EN"
###--------------------------------------------------------------------

###---------------------------------------------------------------------
#export MODEL_NAME="finiteautomata/beto-emotion-analysis"
#export MODEL_SPECNAME="beto-es-analysis"
#export MODEL_VERSION="9b628b0bd91471ad9bd709c10522c379ce09c32a"
#export MODEL_SOURCE="https://huggingface.co/finiteautomata/beto-emotion-analysis"
#export MODEL_LANG="ES"
###--------------------------------------------------------------------

###---------------------------------------------------------------------
#export MODEL_NAME="poltextlab/xlm-roberta-large-pooled-emotions6"
#export MODEL_SPECNAME="poltextlab-s6"
#export MODEL_VERSION="8b77f680ede4f60e5061bfa5a9663bd7fd87803c"
#export MODEL_SOURCE="https://huggingface.co/poltextlab/xlm-roberta-large-pooled-emotions6"
#export MODEL_LANG="Multi"
###--------------------------------------------------------------------


###---------------------------------------------------------------------
#export MODEL_NAME="daveni/twitter-xlm-roberta-emotion-es"
#export MODEL_SPECNAME="twitter-xlm-roberta"
#export MODEL_VERSION="ab57a1137b2eb1f6c90fc77b0a4c4ced7dbd4d60"
#export MODEL_SOURCE="https://huggingface.co/daveni/twitter-xlm-roberta-emotion-es"
#export MODEL_LANG="ES"
###--------------------------------------------------------------------

###---------------------------------------------------------------------
#export MODEL_NAME="ChrisLalk/German-Emotions"
#export MODEL_SPECNAME="german-emotions"
#export MODEL_VERSION="a60c4707ead45e083f67157edd7ff56621ee39e"
#export MODEL_SOURCE="https://huggingface.co/ChrisLalk/German-Emotions"
#export MODEL_LANG="DE"
###--------------------------------------------------------------------

###---------------------------------------------------------------------
#export MODEL_NAME="msgfrom96/xlm_emo_multi"
#export MODEL_SPECNAME="xlm-emo-multi"
#export MODEL_VERSION="56b4b493b8591381fbd309eb4de118cd0771aa4a"
#export MODEL_SOURCE="https://huggingface.co/msgfrom96/xlm_emo_multi"
#export MODEL_LANG="EN,AR"
###--------------------------------------------------------------------

###---------------------------------------------------------------------
#export MODEL_NAME="cointegrated/rubert-tiny2-cedr-emotion-detection"
#export MODEL_SPECNAME="rubert-cedr-emotion"
#export MODEL_VERSION="453ae93ca895c98cda29522c72b6fbc5a08067b9"
#export MODEL_SOURCE="https://huggingface.co/cointegrated/rubert-tiny2-cedr-emotion-detection"
#export MODEL_LANG="RU"
###--------------------------------------------------------------------

###---------------------------------------------------------------------
#export MODEL_NAME="Aniemore/rubert-tiny2-russian-emotion-detection"
#export MODEL_SPECNAME="rubert-tiny2-russian"
#export MODEL_VERSION="a7b5618de479a2f77637393ed2931d48b9618208"
#export MODEL_SOURCE="https://huggingface.co/Aniemore/rubert-tiny2-russian-emotion-detection"
#export MODEL_LANG="RU"
###--------------------------------------------------------------------

###---------------------------------------------------------------------
#export MODEL_NAME="Johnson8187/Chinese-Emotion-Small"
#export MODEL_SPECNAME="chinese-emotion-small"
#export MODEL_VERSION="2c04ce86de44d232f0fbe31413868eb31d791aea"
#export MODEL_SOURCE="https://huggingface.co/Johnson8187/Chinese-Emotion-Small"
#export MODEL_LANG="ZH"
###--------------------------------------------------------------------

###---------------------------------------------------------------------
#export MODEL_NAME="Johnson8187/Chinese-Emotion"
#export MODEL_SPECNAME="chinese-emotion"
#export MODEL_VERSION="76f94d57b9fdf2b801b9ff9ef2d2af16d2ddf27e"
#export MODEL_SOURCE="https://huggingface.co/Johnson8187/Chinese-Emotion"
#export MODEL_LANG="ZH"
###--------------------------------------------------------------------

###---------------------------------------------------------------------
#export MODEL_NAME="AnasAlokla/multilingual_go_emotions"
#export MODEL_SPECNAME="multi-go-emotions"
#export MODEL_VERSION="64610deb3453acfff12bb8289e3e7d0913ae91df"
#export MODEL_SOURCE="https://huggingface.co/AnasAlokla/multilingual_go_emotions"
#export MODEL_LANG="AR,EN,FR,ES,NL,TR"
###--------------------------------------------------------------------

###---------------------------------------------------------------------
#export MODEL_NAME="Zoopa/emotion-classification-model"
#export MODEL_SPECNAME="zoopa-carer-emotion"
#export MODEL_VERSION="829c7072f509941b33ccae5e9d3ea3ba33e07bc9"
#export MODEL_SOURCE="https://huggingface.co/Zoopa/emotion-classification-model"
#export MODEL_LANG="EN"
###--------------------------------------------------------------------

###---------------------------------------------------------------------
#export MODEL_NAME="esuriddick/distilbert-base-uncased-finetuned-emotion"
#export MODEL_SPECNAME="distilbert-carer-emotion"
#export MODEL_VERSION="7c0c353fc08ba81ff219ec9df9fabfd0575c07ef"
#export MODEL_SOURCE="https://huggingface.co/esuriddick/distilbert-base-uncased-finetuned-emotion"
#export MODEL_LANG="EN"
###--------------------------------------------------------------------

###---------------------------------------------------------------------
#export MODEL_NAME="Panda0116/emotion-classification-model"
#export MODEL_SPECNAME="corbett-carer-emotion"
#export MODEL_VERSION="7db05a776be17ff42971dc07c3156f3aae40d730"
#export MODEL_SOURCE="https://huggingface.co/Panda0116/emotion-classification-model"
#export MODEL_LANG="EN"
###--------------------------------------------------------------------

####---------------------------------------------------------------------
#export MODEL_NAME="lordtt13/emo-mobilebert"
#export MODEL_SPECNAME="emo-mobilebert"
#export MODEL_VERSION="26d8fcb41762ae83cc9fa03005cb63cde06ef340"
#export MODEL_SOURCE="https://huggingface.co/lordtt13/emo-mobilebert"
#export MODEL_LANG="EN"
###--------------------------------------------------------------------

###---------------------------------------------------------------------
#export MODEL_NAME="alex-shvets/roberta-large-emopillars-contextual-emocontext"
#export MODEL_SPECNAME="emopillars-emocontext"
#export MODEL_VERSION="94b11cf8e151fc33e114dd78f5a72a5ad7b874cd"
#export MODEL_SOURCE="https://huggingface.co/alex-shvets/roberta-large-emopillars-contextual-emocontext"
#export MODEL_LANG="EN"
###--------------------------------------------------------------------

###---------------------------------------------------------------------
#export MODEL_NAME="AdapterHub/bert-base-uncased-pf-emo"
#export MODEL_SPECNAME="adapterhub-emocontext"
#export MODEL_VERSION="0406b2f47dd4ad43b8ae08230735866eea3f85c1"
#export MODEL_SOURCE="https://huggingface.co/AdapterHub/bert-base-uncased-pf-emo"
#export MODEL_LANG="EN"
###--------------------------------------------------------------------

###---------------------------------------------------------------------
#export MODEL_NAME="UniversalJoy/models-multilingual/small-all.pt"
#export MODEL_SPECNAME="universal-joy-multi-small"
#export MODEL_VERSION="6ab01e98c8106e610247e5e8f0712af08c007b67"
#export MODEL_SOURCE="https://github.com/sotlampr/universal-joy"
#export MODEL_LANG="Multi"
###--------------------------------------------------------------------

###---------------------------------------------------------------------
#export MODEL_NAME="UniversalJoy/models-multilingual/large-all.pt"
#export MODEL_SPECNAME="universal-joy-multi-large"
#export MODEL_VERSION="6ab01e98c8106e610247e5e8f0712af08c007b67"
#export MODEL_SOURCE="https://github.com/sotlampr/universal-joy"
#export MODEL_LANG="Multi"
###--------------------------------------------------------------------

###---------------------------------------------------------------------
#export MODEL_NAME="UniversalJoy/models-multilingual/combi-all.pt"
#export MODEL_SPECNAME="universal-joy-multi-combi"
#export MODEL_VERSION="6ab01e98c8106e610247e5e8f0712af08c007b67"
#export MODEL_SOURCE="https://github.com/sotlampr/universal-joy"
#export MODEL_LANG="Multi"
###--------------------------------------------------------------------

###---------------------------------------------------------------------
#export MODEL_NAME="UniversalJoy/models-huge-english/huge-en.pt"
#export MODEL_SPECNAME="universal-joy-en-huge"
#export MODEL_VERSION="6ab01e98c8106e610247e5e8f0712af08c007b67"
#export MODEL_SOURCE="https://github.com/sotlampr/universal-joy"
#export MODEL_LANG="EN"
###--------------------------------------------------------------------

###---------------------------------------------------------------------
#export MODEL_NAME="UniversalJoy/models-large-monolingual/large-en.pt"
#export MODEL_SPECNAME="universal-joy-en-large"
#export MODEL_VERSION="6ab01e98c8106e610247e5e8f0712af08c007b67"
#export MODEL_SOURCE="https://github.com/sotlampr/universal-joy"
#export MODEL_LANG="EN"
###--------------------------------------------------------------------

###---------------------------------------------------------------------
#export MODEL_NAME="UniversalJoy/models-large-monolingual/large-es.pt"
#export MODEL_SPECNAME="universal-joy-es-large"
#export MODEL_VERSION="6ab01e98c8106e610247e5e8f0712af08c007b67"
#export MODEL_SOURCE="https://github.com/sotlampr/universal-joy"
#export MODEL_LANG="ES"
###--------------------------------------------------------------------

###---------------------------------------------------------------------
#export MODEL_NAME="UniversalJoy/models-large-monolingual/large-pt.pt"
#export MODEL_SPECNAME="universal-joy-pt-large"
#export MODEL_VERSION="6ab01e98c8106e610247e5e8f0712af08c007b67"
#export MODEL_SOURCE="https://github.com/sotlampr/universal-joy"
#export MODEL_LANG="PT"
###--------------------------------------------------------------------

###---------------------------------------------------------------------
#export MODEL_NAME="UniversalJoy/models-small-monolingual/small-en.pt"
#export MODEL_SPECNAME="universal-joy-en-small"
#export MODEL_VERSION="6ab01e98c8106e610247e5e8f0712af08c007b67"
#export MODEL_SOURCE="https://github.com/sotlampr/universal-joy"
#export MODEL_LANG="EN"
###--------------------------------------------------------------------

###---------------------------------------------------------------------
#export MODEL_NAME="UniversalJoy/models-small-monolingual/small-es.pt"
#export MODEL_SPECNAME="universal-joy-es-small"
#export MODEL_VERSION="6ab01e98c8106e610247e5e8f0712af08c007b67"
#export MODEL_SOURCE="https://github.com/sotlampr/universal-joy"
#export MODEL_LANG="ES"
###--------------------------------------------------------------------

###---------------------------------------------------------------------
#export MODEL_NAME="UniversalJoy/models-small-monolingual/small-pt.pt"
#export MODEL_SPECNAME="universal-joy-pt-small"
#export MODEL_VERSION="6ab01e98c8106e610247e5e8f0712af08c007b67"
#export MODEL_SOURCE="https://github.com/sotlampr/universal-joy"
#export MODEL_LANG="PT"
###--------------------------------------------------------------------

###---------------------------------------------------------------------
#export MODEL_NAME="UniversalJoy/models-small-monolingual/small-tl.pt"
#export MODEL_SPECNAME="universal-joy-tl-small"
#export MODEL_VERSION="6ab01e98c8106e610247e5e8f0712af08c007b67"
#export MODEL_SOURCE="https://github.com/sotlampr/universal-joy"
#export MODEL_LANG="TL"
###--------------------------------------------------------------------

####---------------------------------------------------------------------
#export MODEL_NAME="UniversalJoy/models-small-monolingual/small-zh.pt"
#export MODEL_SPECNAME="universal-joy-zh-small"
#export MODEL_VERSION="6ab01e98c8106e610247e5e8f0712af08c007b67"
#export MODEL_SOURCE="https://github.com/sotlampr/universal-joy"
#export MODEL_LANG="ZH"
####--------------------------------------------------------------------


docker build \
  --build-arg ANNOTATOR_NAME \
  --build-arg ANNOTATOR_VERSION \
  --build-arg LOG_LEVEL \
  --build-arg MODEL_CACHE_SIZE \
  --build-arg MODEL_NAME \
  --build-arg MODEL_VERSION \
  --build-arg MODEL_SOURCE \
  --build-arg MODEL_LANG \
  -t ${DOCKER_REGISTRY}${ANNOTATOR_NAME}"-"${MODEL_SPECNAME}:${ANNOTATOR_VERSION}${ANNOTATOR_CUDA} \
  -f src/main/docker/Dockerfile${ANNOTATOR_CUDA} \
  .

docker tag \
  ${DOCKER_REGISTRY}${ANNOTATOR_NAME}"-"${MODEL_SPECNAME}:${ANNOTATOR_VERSION}${ANNOTATOR_CUDA} \
  ${DOCKER_REGISTRY}${ANNOTATOR_NAME}"-"${MODEL_SPECNAME}:latest${ANNOTATOR_CUDA}