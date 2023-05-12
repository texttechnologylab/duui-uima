package org.hucompute.textimager.uima.ddc.fasttext.service.service;

import io.quarkus.arc.Lock;

import javax.annotation.PreDestroy;
import javax.inject.Singleton;

@Lock
@Singleton
public class FastTextDDC2DEService extends FastTextDDC2Service {
    FastTextDDC2DEService() throws Exception {
        this(
                "",
                "",
                ""
        );
    }

    FastTextDDC2DEService(String disambigTag, String disambigLabelReplace, String disambigLabelReplaceWith) throws Exception {
        // TODO paths in container are absolute!
        // TODO WICHTIG die parameter weiter anpassen an die Daten aus der TI Config
        super(
                "/home/stud_homes/baumartz/mnt/resources/nlp/bin/categorization/fastText_original_for_ducc_annotators/fasttext",
                "de,/home/stud_homes/baumartz/mnt/resources/nlp/models/categorization/ddc/ddc_2018/wikipedia_de.v8.lemma.nopunct.pos.no_functionwords_gnd_ddc.v4.with_categories-lr0.2-lrUR150-minC5-dim100-ep10000-vec_vec_token_lemmapos.vec.epoch5000.bin,98",
                false,
                1,
                true,
                true,
                "/home/stud_homes/baumartz/mnt/resources/nlp/models/categorization/am_posmap.txt",
                true,
                true,
                false, // TODO was true originally
                false,
                100,
                "ddc2",
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
