package org.hucompute.textimager.uima.ddc.fasttext.service.model.duui;

import io.smallrye.common.constraint.Nullable;

import java.util.Map;

public class DUUIDocumentation {
    public String annotator_name;
    public String version;

    @Nullable
    public String implementation_lang;

    @Nullable
    public Map<String, String> meta;

    @Nullable
    public String docker_container_id;

    @Nullable
    public Map<String, String> parameters;

    public DUUICapability capability;

    @Nullable
    public String implementation_specific;

}
