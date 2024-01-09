package org.hucompute.textimager.uima.ddc.fasttext.service.service.impl;

import io.quarkus.arc.Lock;
import org.hucompute.textimager.uima.ddc.fasttext.service.service.FastTextDDC2Service;

import javax.annotation.PreDestroy;
import javax.inject.Singleton;

@Lock
@Singleton
public class FastTextDDC1ENDim100Service extends FastTextDDC2Service {
    FastTextDDC1ENDim100Service() throws Exception {
        this(
                "",
                "",
                ""
        );
    }

    FastTextDDC1ENDim100Service(String disambigTag, String disambigLabelReplace, String disambigLabelReplaceWith) throws Exception {
        // TODO paths in container are absolute!
        // TODO WICHTIG die parameter weiter anpassen an die Daten aus der TI Config
        super(
                "/home/stud_homes/baumartz/mnt/resources/nlp/bin/categorization/fastText_modern_for_duui_annotators/fasttext",
                "en,/home/stud_homes/baumartz/mnt/resources/nlp/models/categorization/ddc/ddc_2023/wikipedia_en.v8.lemma.nopunct_gnd_ddc1.v3.with_wikidata_dim100_lr0.2_lrur150_mincount5_epoch100.bin,10",
                true,
                1,
                true,
                false,
                "/home/stud_homes/baumartz/mnt/resources/nlp/models/categorization/am_posmap.txt",
                true,
                false,
                false, // TODO was true originally
                false,
                10,
                "ddc1;dim100;ddc_2023/wikipedia_en.v8.lemma.nopunct_gnd_ddc1.v3.with_wikidata_dim100_lr0.2_lrur150_mincount5_epoch100.bin",
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
