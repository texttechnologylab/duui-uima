package org.hucompute.textimager.uima.ddc.fasttext.service.model;

import javax.validation.constraints.NotBlank;
import javax.validation.constraints.NotNull;

public class DDCRequest {
    @NotNull
    @NotBlank
    public String lang;

    @NotNull
    @NotBlank
    public String text;
}
