package org.hucompute.textimager.uima.ddc.fasttext.service.service;

import io.quarkus.arc.Lock;

import javax.annotation.PreDestroy;
import javax.inject.Singleton;

@Lock
@Singleton
public class FastTextDDC3DEService extends FastTextDDC3Service {
    FastTextDDC3DEService() throws Exception {
        this(
                "",
                "",
                ""
        );
    }

    FastTextDDC3DEService(String disambigTag, String disambigLabelReplace, String disambigLabelReplaceWith) throws Exception {
        // TODO paths in container are absolute!
        // TODO WICHTIG die parameter weiter anpassen an die Daten aus der TI Config
        super(
                "/home/stud_homes/baumartz/mnt/resources/nlp/bin/categorization/fastText_original_for_ducc_annotators/fasttext",
                "de,/home/stud_homes/baumartz/mnt/resources/nlp/models/categorization/ddc/ddc_2018/wikipedia_de.v8.lemma.nopunct.pos.no_functionswords_gnd_ddc_full.v5.with_categories-lr0.2-lrUR150-minC5-dim100-ep10000-vec_vec_token_lemmapos.vec.epoch5000.bin,635",
                false,
                1,
                true,
                true,
                "/home/stud_homes/baumartz/mnt/resources/nlp/models/categorization/am_posmap.txt",
                true,
                true,
                false, // TODO was true originally
                false,
                1000,
                "ddc2",
                "ddc3",
                disambigTag,
                disambigLabelReplace,
                disambigLabelReplaceWith
        );
    }

    @PreDestroy
    void preDestroy() {
        this.exit();
    }
}
