#!/usr/bin/env bash
set -euo pipefail

export TEXTIMAGER_DUUI_DDC_FASTTEXT_ANNOTATOR_NAME=textimager-duui-ddc-fasttext
export TEXTIMAGER_DUUI_DDC_FASTTEXT_ANNOTATOR_VERSION=2.3.3

export DOCKER_REGISTRY="docker.texttechnologylab.org/"

#export JAVA_HOME=/usr/lib64/jvm/java-11-openjdk-11/

./mvnw clean
./mvnw package -DskipTests

# copy external data to access in docker context
# TODO these should better be accessed via a server...
mkdir -p "./docker_data"
rsync -aP "/home/staff_homes/dbaumart/mnt/resources/nlp/models/categorization/am_posmap.txt" "./docker_data/"
rsync -aP "/home/staff_homes/dbaumart/mnt/resources/nlp/bin/categorization/fastText_original_for_ducc_annotators" "./docker_data/"
rsync -aP "/home/staff_homes/dbaumart/dev/git/fastText_modern_for_duui_annotators" "./docker_data/"
rsync -aP "/home/staff_homes/dbaumart/mnt/resources/nlp/models/categorization/ddc/ddc_2018/wikipedia_de.v8.lemma.nopunct.pos.no_functionwords_gnd_ddc.v4.with_categories-lr0.2-lrUR150-minC5-dim100-ep10000-vec_vec_token_lemmapos.vec.epoch5000.bin" "./docker_data/"
rsync -aP "/home/staff_homes/dbaumart/mnt/resources/nlp/models/categorization/ddc/ddc2_2018/wikipedia_en.v8.lemma.nopunct_gnd_ddc.v3.with_wikidata_model_dim100_pretreined-glove.6B.100d.txt_epoch100000.epoch10000.bin" "./docker_data/"
rsync -aP "/home/staff_homes/dbaumart/mnt/resources/nlp/models/categorization/ddc/ddc_2018/wikipedia_de.v8.lemma.nopunct.pos.no_functionswords_gnd_ddc_full.v5.with_categories-lr0.2-lrUR150-minC5-dim100-ep10000-vec_vec_token_lemmapos.vec.epoch5000.bin" "./docker_data/"
rsync -aP "/home/staff_homes/dbaumart/mnt/resources/nlp/models/categorization/ddc/ddc3_2018/wikipedia_en.v8.lemma.nopunct_gnd_ddc_full.v5.with_categories_dim300-ep10000-vec_wiki.en.vec.bin" "./docker_data/"
rsync -aP "/storage/projects/baumartz/text2ddc2023/models/wikipedia_lemma_nopunct_pos__en_withwikidata__de_nofunction_withcategories__ddc1__dim100_lr0.2_lrur150_mincount5_epoch100.bin" "./docker_data/"
rsync -aP "/storage/projects/baumartz/text2ddc2023/models/wikipedia_lemma_nopunct_pos__en_withwikidata__de_nofunction_withcategories__ddc1__dim300_lr0.2_lrur150_mincount5_epoch100.bin" "./docker_data/"
rsync -aP "/storage/projects/baumartz/text2ddc2023/models/wikipedia_lemma_nopunct_pos_withcategories__en__de_nofunction__ddc2__dim100_lr0.2_lrur150_mincount5_epoch100.bin" "./docker_data/"
rsync -aP "/storage/projects/baumartz/text2ddc2023/models/wikipedia_lemma_nopunct_pos_withcategories__en__de_nofunction__ddc2__dim300_lr0.2_lrur150_mincount5_epoch100.bin" "./docker_data/"
rsync -aP "/storage/projects/baumartz/text2ddc2023/models/wikipedia_lemma_nopunct_pos_withcategories__en__de_nofunction__ddc3__dim100_lr0.2_lrur150_mincount5_epoch100.bin" "./docker_data/"
rsync -aP "/storage/projects/baumartz/text2ddc2023/models/wikipedia_lemma_nopunct_pos_withcategories__en__de_nofunction__ddc3__dim300_lr0.2_lrur150_mincount5_epoch100.bin" "./docker_data/"
rsync -aP "/storage/projects/baumartz/text2ddc2023/models/wikipedia_de.v8.lemma.nopunct.pos.no_functionwords_gnd_ddc1.v4.with_categories_dim100_lr0.2_lrur150_mincount5_epoch100.bin" "./docker_data/"
rsync -aP "/storage/projects/baumartz/text2ddc2023/models/wikipedia_de.v8.lemma.nopunct.pos.no_functionwords_gnd_ddc1.v4.with_categories_dim300_lr0.2_lrur150_mincount5_epoch100.bin" "./docker_data/"
rsync -aP "/storage/projects/baumartz/text2ddc2023/models/wikipedia_en.v8.lemma.nopunct_gnd_ddc1.v3.with_wikidata_dim100_lr0.2_lrur150_mincount5_epoch100.bin" "./docker_data/"
rsync -aP "/storage/projects/baumartz/text2ddc2023/models/wikipedia_en.v8.lemma.nopunct_gnd_ddc1.v3.with_wikidata_dim300_lr0.2_lrur150_mincount5_epoch100.bin" "./docker_data/"
rsync -aP "/storage/projects/baumartz/text2ddc2023/models/wikipedia_de.v8.lemma.nopunct.pos.no_functionwords_gnd_ddc.v4.with_categories_dim300_lr0.2_lrur150_mincount5_epoch100.bin" "./docker_data/"
rsync -aP "/storage/projects/baumartz/text2ddc2023/models/wikipedia_en.v8.lemma.nopunct_gnd_ddc.v3.with_wikidata_dim300_lr0.2_lrur150_mincount5_epoch100.bin" "./docker_data/"
rsync -aP "/storage/projects/baumartz/text2ddc2023/models/wikipedia_de.v8.lemma.nopunct.pos.no_functionswords_gnd_ddc_full.v5.with_categories_dim300_lr0.2_lrur150_mincount5_epoch100.bin" "./docker_data/"
rsync -aP "/storage/projects/baumartz/text2ddc2023/models/wikipedia_en.v8.lemma.nopunct_gnd_ddc_full.v5.with_categories_dim100_lr0.2_lrur150_mincount5_epoch100.bin" "./docker_data/"

docker build \
  --build-arg TEXTIMAGER_DUUI_DDC_FASTTEXT_ANNOTATOR_NAME \
  --build-arg TEXTIMAGER_DUUI_DDC_FASTTEXT_ANNOTATOR_VERSION \
  -t ${DOCKER_REGISTRY}${TEXTIMAGER_DUUI_DDC_FASTTEXT_ANNOTATOR_NAME}:${TEXTIMAGER_DUUI_DDC_FASTTEXT_ANNOTATOR_VERSION} \
  -f "src/main/docker/Dockerfile.jvm" \
  .

docker tag \
  ${DOCKER_REGISTRY}${TEXTIMAGER_DUUI_DDC_FASTTEXT_ANNOTATOR_NAME}:${TEXTIMAGER_DUUI_DDC_FASTTEXT_ANNOTATOR_VERSION} \
  ${DOCKER_REGISTRY}${TEXTIMAGER_DUUI_DDC_FASTTEXT_ANNOTATOR_NAME}:latest
