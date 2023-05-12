package org.hucompute.textimager.uima.ddc.fasttext.service.service;

import io.quarkus.arc.Lock;

import javax.annotation.PreDestroy;
import javax.inject.Singleton;

@Lock
@Singleton
public class FastTextDDC3ENService extends FastTextDDC3Service {
    FastTextDDC3ENService() throws Exception {
        this(
                "",
                "",
                ""
        );
    }

    FastTextDDC3ENService(String disambigTag, String disambigLabelReplace, String disambigLabelReplaceWith) throws Exception {
        // TODO paths in container are absolute!
        // TODO WICHTIG die parameter weiter anpassen an die Daten aus der TI Config
        super(
                "/home/stud_homes/baumartz/mnt/resources/nlp/bin/categorization/fastText_original_for_ducc_annotators/fasttext",
                "en,/home/stud_homes/baumartz/mnt/resources/nlp/models/categorization/ddc/ddc3_2018/wikipedia_en.v8.lemma.nopunct_gnd_ddc_full.v5.with_categories_dim300-ep10000-vec_wiki.en.vec.bin,601",
                false,
                1,
                true,
                false,
                "/home/stud_homes/baumartz/mnt/resources/nlp/models/categorization/am_posmap.txt",
                true,
                false,
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
