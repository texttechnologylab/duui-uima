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
import org.hucompute.textimager.uima.ddc.fasttext.service.service.*;
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
    @Inject
    FastTextDDC1DEService fastTextDDC1DEService;

    @Inject
    FastTextDDC2DEService fastTextDDC2DEService;
    @Inject
    FastTextDDC3DEService fastTextDDC3DEService;

    @Inject
    FastTextDDC2ENService fastTextDDC2ENService;
    @Inject
    FastTextDDC3ENService fastTextDDC3ENService;

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
        String ddcVariant = "ddc2";
        String selection = "text";

        for (AnnotationComment parameter : JCasUtil.select(jCas, AnnotationComment.class)) {
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

        if (ddcVariant.equals("ddc1")) {
            if (documentLanguage.equals("de")) {
                fastTextDDC1DEService.process(jCas, selection);
            }
            else {
                throw new UIMAException(new Exception("Only DE languages are supported for DDC1!"));
            }
        }
        else if (ddcVariant.equals("ddc2")) {
            if (documentLanguage.equals("de")) {
                fastTextDDC2DEService.process(jCas, selection);
            }
            else if (documentLanguage.equals("en")) {
                fastTextDDC2ENService.process(jCas, selection);
            }
            else {
                throw new UIMAException(new Exception("Only EN and DE languages are supported for DDC2!"));
            }
        }
        else if (ddcVariant.equals("ddc3")) {
            // For DDC3 we also need to first run DDC2
            if (documentLanguage.equals("de")) {
                fastTextDDC2DEService.process(jCas, selection);
                fastTextDDC3DEService.process(jCas, selection);
            }
            else if (documentLanguage.equals("en")) {
                fastTextDDC2ENService.process(jCas, selection);
                fastTextDDC3ENService.process(jCas, selection);
            }
            else {
                throw new UIMAException(new Exception("Only EN and DE languages are supported for DDC3!"));
            }
        }
        else {
            throw new UIMAException(new Exception("Only DDC2 and DDC3 are supported!"));
        }

        OutputStream casStream = new ByteArrayOutputStream();
        //XmiCasSerializer.serialize(jCas.getCas(), casStream);

        XMLSerializer xmlSerializer = new XMLSerializer(casStream, true);
        xmlSerializer.setOutputProperty(OutputKeys.VERSION, "1.1");
        xmlSerializer.setOutputProperty(OutputKeys.ENCODING, StandardCharsets.UTF_8.toString());
        XmiCasSerializer xmiCasSerializer = new XmiCasSerializer(null);
        xmiCasSerializer.serialize(jCas.getCas(), xmlSerializer.getContentHandler());

        return casStream;
    }
}
