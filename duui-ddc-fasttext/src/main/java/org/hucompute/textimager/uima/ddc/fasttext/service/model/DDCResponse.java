package org.hucompute.textimager.uima.ddc.fasttext.service.model;

import com.fasterxml.jackson.annotation.JsonProperty;
import org.hucompute.textimager.uima.ddc.fasttext.service.service.FastTextResult;

import java.util.ArrayList;

public class DDCResponse {
    @JsonProperty("fasttext_results")
    public ArrayList<FastTextResult> fastTextResults;

    public DDCResponse() {
        fastTextResults = new ArrayList<>();
    }
}
