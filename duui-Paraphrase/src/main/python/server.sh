# meta data
# model names options:
# 1) "tuner007/pegasus_paraphrase" : PegasusBase
# 2) "humarin/chatgpt_paraphraser_on_T5_base" : T5Base
# 3) "eugenesiow/bart-paraphrase": BartBase
# 4) "prithivida/parrot_paraphraser_on_T5": ParrotBase
MODEL_NAME="humarin/chatgpt_paraphraser_on_T5_base" \
CUDA=1 \
GPU_ID=0 \
TEXTIMAGER_PARA_ANNOTATOR_NAME="textimager-duui-paraphraser" \
TEXTIMAGER_PARA_ANNOTATOR_VERSION="0.1" \
/home/leon/uni/MSwAI_Gruppe2/venv/bin/python duui_paraphraser.py
#python duui_biofid.py
# specific env