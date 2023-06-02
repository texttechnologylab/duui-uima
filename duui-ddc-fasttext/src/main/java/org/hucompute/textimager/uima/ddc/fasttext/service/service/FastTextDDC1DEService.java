package org.hucompute.textimager.uima.ddc.fasttext.service.service;

import io.quarkus.arc.Lock;

import javax.annotation.PreDestroy;
import javax.inject.Singleton;

@Lock
@Singleton
public class FastTextDDC1DEService extends FastTextDDC2Service {
    FastTextDDC1DEService() throws Exception {
        this(
                "",
                "",
                ""
        );
    }

    FastTextDDC1DEService(String disambigTag, String disambigLabelReplace, String disambigLabelReplaceWith) throws Exception {
        // TODO paths in container are absolute!
        // TODO WICHTIG die parameter weiter anpassen an die Daten aus der TI Config
        super(
                "/home/stud_homes/baumartz/mnt/resources/nlp/bin/categorization/fastText_modern_for_duui_annotators/fasttext",
                "de,/home/stud_homes/baumartz/mnt/resources/nlp/models/categorization/ddc/ddc_2023/wikipedia_de.v8.lemma.nopunct.pos.no_functionwords_gnd_ddc1.v4.with_categories_dim300_lr0.2_lrur150_mincount5_epoch500.bin,10",
                false,
                1,
                true,
                true,
                "/home/stud_homes/baumartz/mnt/resources/nlp/models/categorization/am_posmap.txt",
                true,
                true,
                false, // TODO was true originally
                false,
                10,
                "ddc1",
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
