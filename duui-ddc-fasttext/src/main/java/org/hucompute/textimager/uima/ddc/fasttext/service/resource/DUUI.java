package org.hucompute.textimager.uima.ddc.fasttext.service.resource;

import de.tudarmstadt.ukp.dkpro.core.api.metadata.type.DocumentMetaData;
import org.apache.uima.UIMAException;
import org.apache.uima.cas.impl.XmiCasDeserializer;
import org.apache.uima.cas.impl.XmiCasSerializer;
import org.apache.uima.fit.factory.JCasFactory;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;
import org.apache.uima.util.XMLSerializer;
import org.eclipse.microprofile.config.ConfigProvider;
import org.eclipse.microprofile.metrics.MetricUnits;
import org.eclipse.microprofile.metrics.annotation.Counted;
import org.eclipse.microprofile.metrics.annotation.Timed;
import org.hucompute.textimager.uima.ddc.fasttext.service.model.duui.DUUICapability;
import org.hucompute.textimager.uima.ddc.fasttext.service.model.duui.DUUIDocumentation;
import org.hucompute.textimager.uima.ddc.fasttext.service.service.impl.*;
import org.texttechnologylab.annotation.AnnotationComment;
import org.xml.sax.SAXException;

import javax.inject.Inject;
import javax.ws.rs.*;
import javax.ws.rs.core.MediaType;
import javax.xml.transform.OutputKeys;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;

@Path("/v1")
public class DUUI {
    // ML
    @Inject
    FastTextDDC1MLDim100Service fastTextDDC1MLDim100Service;
    @Inject
    FastTextDDC1MLDim300Service fastTextDDC1MLDim300Service;
    @Inject
    FastTextDDC2MLDim100Service fastTextDDC2MLDim100Service;
    @Inject
    FastTextDDC2MLDim300Service fastTextDDC2MLDim300Service;
    @Inject
    FastTextDDC3MLDim100Service fastTextDDC3MLDim100Service;
    @Inject
    FastTextDDC3MLDim300Service fastTextDDC3MLDim300Service;

    // DE
    @Inject
    FastTextDDC1DEDim100Service fastTextDDC1DEDim100Service;
    @Inject
    FastTextDDC1DEDim300Service fastTextDDC1DEDim300Service;
    @Inject
    FastTextDDC2DEDim100Service fastTextDDC2DEDim100Service;
    @Inject
    FastTextDDC2DEDim300Service fastTextDDC2DEDim300Service;
    @Inject
    FastTextDDC3DEDim100Service fastTextDDC3DEDim100Service;
    @Inject
    FastTextDDC3DEDim300Service fastTextDDC3DEDim300Service;

    // EN
    @Inject
    FastTextDDC1ENDim100Service fastTextDDC1ENDim100Service;
    @Inject
    FastTextDDC1ENDim300Service fastTextDDC1ENDim300Service;
    @Inject
    FastTextDDC2ENDim100Service fastTextDDC2ENDim100Service;
    @Inject
    FastTextDDC2ENDim300Service fastTextDDC2ENDim300Service;
    @Inject
    FastTextDDC3ENDim100Service fastTextDDC3ENDim100Service;
    @Inject
    FastTextDDC3ENDim300Service fastTextDDC3ENDim300Service;

    @GET
    @Path("communication_layer")
    @Produces(MediaType.TEXT_PLAIN)
    @Counted(name = "duui_v1_communication_layer_count")
    @Timed(name = "duui_v1_communication_layer_timer", unit = MetricUnits.MILLISECONDS)
    public String communication_layer() throws IOException {
        return Files.readString(Paths.get("/home/stud_homes/baumartz/dev/git/TTLab-UIMA/textimager-uima-ddc-fasttext/src/main/resources/textimager_duui_ddc_fasttext.lua"), StandardCharsets.UTF_8);
    }

    @GET
    @Path("documentation")
    @Produces(MediaType.APPLICATION_JSON)
    @Counted(name = "duui_v1_documentation_count")
    @Timed(name = "duui_v1_documentation_timer", unit = MetricUnits.MILLISECONDS)
    public DUUIDocumentation documentation() {
        DUUIDocumentation documentation = new DUUIDocumentation();
        documentation.annotator_name = "textimager-uima-ddc-fasttext";
        documentation.version = ConfigProvider.getConfig().getValue("quarkus.application.version", String.class);
        documentation.implementation_lang = "Java";
        documentation.meta = new HashMap<>();
        documentation.meta.put("java_version", Runtime.version().toString());
        documentation.docker_container_id = "TODO";
        documentation.parameters = new HashMap<>();
        documentation.parameters.put("model_name", "TODO");
        documentation.capability = new DUUICapability();
        documentation.capability.supported_languages = new ArrayList<>();
        documentation.capability.supported_languages.add("TODO");
        documentation.capability.reproducible = false;
        documentation.implementation_specific = null;
        return documentation;
    }

    @GET
    @Path("typesystem")
    @Produces(MediaType.APPLICATION_XML)
    @Counted(name = "duui_v1_typesystem_count")
    @Timed(name = "duui_v1_typesystem_timer", unit = MetricUnits.MILLISECONDS)
    public String typesystem() throws IOException {
        return Files.readString(Paths.get("/home/stud_homes/baumartz/dev/git/TTLab-UIMA/textimager-uima-ddc-fasttext/src/main/resources/TypeSystemCategory.xml"), StandardCharsets.UTF_8);
    }

    @POST
    @Path("process")
    @Consumes(MediaType.WILDCARD)
    @Produces(MediaType.TEXT_PLAIN)
    @Counted(name = "duui_v1_process_count")
    @Timed(name = "duui_v1_process_timer", unit = MetricUnits.MILLISECONDS)
    public OutputStream process(InputStream body) throws UIMAException, IOException, SAXException {
        JCas jCas = JCasFactory.createJCas();
        XmiCasDeserializer.deserialize(body, jCas.getCas(), true);

        String documentLanguage = jCas.getDocumentLanguage();
        String ddcVariant = "__";
        String selection = "__";

        for (AnnotationComment parameter : JCasUtil.select(jCas, AnnotationComment.class)) {
            System.out.println("Possible Parameter:");
            System.out.println(parameter.getKey());
            System.out.println(parameter.getValue());
            if (!parameter.getKey().equals("__textimager_duui_configuration_parameter_ddc_fasttext__")) {
                continue;
            }

            String[] fields = parameter.getValue().split("¤", 2);
            if (fields.length != 2) {
                System.out.println("Parameter field \"" + parameter.getValue() + "\" is invalid!");
                continue;
            }

            String name = fields[0].strip();
            String value = fields[1].strip();

            switch (name) {
                case "ddc_variant":
                    ddcVariant = value;
                    break;
                case "selection":
                    selection = value;
                    break;
                default:
                    System.out.println("Unknown parameter \"" + parameter.getValue() + "\" received!");
            }
        }

        System.out.println("Lang: " + documentLanguage);
        System.out.println("DDC: " + ddcVariant);
        System.out.println("Selection: " + selection);

        try {
            DocumentMetaData meta = DocumentMetaData.get(jCas);
            System.out.println("Document: " + meta.getDocumentId());
        }
        catch (Exception e) {
            System.out.println("Document unknown");
        }

        // Specific "new" models
        switch (ddcVariant) {
            case "ddc1_dim100":
                if (documentLanguage.equals("de")) {
                    fastTextDDC1DEDim100Service.process(jCas, selection);
                } else if (documentLanguage.equals("en")) {
                    fastTextDDC1ENDim100Service.process(jCas, selection);
                } else {
                    throw new UIMAException(new Exception("Only EN and DE languages are supported for ddc1_dim100!"));
                }
                break;
            case "ddc1_dim300":
                if (documentLanguage.equals("de")) {
                    fastTextDDC1DEDim300Service.process(jCas, selection);
                } else if (documentLanguage.equals("en")) {
                    fastTextDDC1ENDim300Service.process(jCas, selection);
                } else {
                    throw new UIMAException(new Exception("Only EN and DE languages are supported for ddc1_dim300!"));
                }
                break;
            case "ddc1_dim100_ml":
                jCas.setDocumentLanguage("_ml");
                fastTextDDC1MLDim100Service.process(jCas, selection);
                jCas.setDocumentLanguage(documentLanguage);
                break;
            case "ddc1_dim300_ml":
                jCas.setDocumentLanguage("_ml");
                fastTextDDC1MLDim300Service.process(jCas, selection);
                jCas.setDocumentLanguage(documentLanguage);
                break;
            case "ddc2_dim100":
                if (documentLanguage.equals("de")) {
                    fastTextDDC2DEDim100Service.process(jCas, selection);
                } else if (documentLanguage.equals("en")) {
                    fastTextDDC2ENDim100Service.process(jCas, selection);
                } else {
                    throw new UIMAException(new Exception("Only EN and DE languages are supported for ddc2_dim100!"));
                }
                break;
            case "ddc2_dim300":
                if (documentLanguage.equals("de")) {
                    fastTextDDC2DEDim300Service.process(jCas, selection);
                } else if (documentLanguage.equals("en")) {
                    fastTextDDC2ENDim300Service.process(jCas, selection);
                } else {
                    throw new UIMAException(new Exception("Only EN and DE languages are supported for ddc2_dim300!"));
                }
                break;
            case "ddc2_dim100_ml":
                jCas.setDocumentLanguage("_ml");
                fastTextDDC2MLDim100Service.process(jCas, selection);
                jCas.setDocumentLanguage(documentLanguage);
                break;
            case "ddc2_dim300_ml":
                jCas.setDocumentLanguage("_ml");
                fastTextDDC2MLDim300Service.process(jCas, selection);
                jCas.setDocumentLanguage(documentLanguage);
                break;
            case "ddc3_dim100":
                if (documentLanguage.equals("de")) {
                    fastTextDDC2DEDim100Service.process(jCas, selection);
                    fastTextDDC3DEDim100Service.process(jCas, selection);
                } else if (documentLanguage.equals("en")) {
                    fastTextDDC3ENDim100Service.process(jCas, selection);
                } else {
                    throw new UIMAException(new Exception("Only EN and DE languages are supported for ddc3_dim100!"));
                }
                break;
            case "ddc3_dim300":
                if (documentLanguage.equals("de")) {
                    fastTextDDC3DEDim300Service.process(jCas, selection);
                } else if (documentLanguage.equals("en")) {
                    fastTextDDC2ENDim100Service.process(jCas, selection);
                    fastTextDDC3ENDim300Service.process(jCas, selection);
                } else {
                    throw new UIMAException(new Exception("Only EN and DE languages are supported for ddc3_dim300!"));
                }
                break;
            case "ddc3_dim100_ml":
                jCas.setDocumentLanguage("_ml");
                fastTextDDC3MLDim100Service.process(jCas, selection);
                jCas.setDocumentLanguage(documentLanguage);
                break;
            case "ddc3_dim300_ml":
                jCas.setDocumentLanguage("_ml");
                fastTextDDC3MLDim300Service.process(jCas, selection);
                jCas.setDocumentLanguage(documentLanguage);
                break;
            // Default "old" models for DDC1 and DDC2
            case "ddc2":
                if (documentLanguage.equals("de")) {
                    fastTextDDC2DEDim100Service.process(jCas, selection);
                } else if (documentLanguage.equals("en")) {
                    fastTextDDC2ENDim100Service.process(jCas, selection);
                } else {
                    throw new UIMAException(new Exception("Only EN and DE languages are supported for DDC2!"));
                }
                break;
            case "ddc3":
                // For DDC3 we also need to first run DDC2
                if (documentLanguage.equals("de")) {
                    fastTextDDC2DEDim100Service.process(jCas, selection);
                    fastTextDDC3DEDim100Service.process(jCas, selection);
                } else if (documentLanguage.equals("en")) {
                    fastTextDDC2ENDim100Service.process(jCas, selection);
                    fastTextDDC3ENDim300Service.process(jCas, selection);
                } else {
                    throw new UIMAException(new Exception("Only EN and DE languages are supported for DDC3!"));
                }
                break;
            default:
                throw new UIMAException(new Exception("Only DDC2 and DDC3 are supported!"));
        }

        OutputStream casStream = new ByteArrayOutputStream();

        XMLSerializer xmlSerializer = new XMLSerializer(casStream, true);
        xmlSerializer.setOutputProperty(OutputKeys.VERSION, "1.1");
        xmlSerializer.setOutputProperty(OutputKeys.ENCODING, StandardCharsets.UTF_8.toString());
        XmiCasSerializer xmiCasSerializer = new XmiCasSerializer(null);
        xmiCasSerializer.serialize(jCas.getCas(), xmlSerializer.getContentHandler());

        jCas.reset();
        jCas = null;

        return casStream;
    }
}
