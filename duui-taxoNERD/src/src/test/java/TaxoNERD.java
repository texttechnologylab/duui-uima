import org.apache.uima.UIMAException;
import org.apache.uima.cas.CASException;
import org.apache.uima.fit.factory.JCasFactory;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;
import org.apache.uima.resource.ResourceInitializationException;
import org.json.JSONArray;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.texttechnologylab.DockerUnifiedUIMAInterface.DUUIComposer;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIDockerDriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIRemoteDriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIUIMADriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaContext;
import org.texttechnologylab.annotation.AnnotationComment;
import org.texttechnologylab.annotation.type.Taxon;
import org.xml.sax.SAXException;

import java.io.IOException;
import java.net.URISyntaxException;
import java.util.Arrays;

public class TaxoNERD {

    public static DUUIComposer pComposer = null;
    public static final int WORKERS = 1;

    @BeforeAll
    public static void init() throws IOException, UIMAException, SAXException, URISyntaxException {

        System.out.println("INIT...");

        DUUILuaContext ctx = new DUUILuaContext().withJsonLibrary();

        pComposer = new DUUIComposer()
                .withSkipVerification(true)
                .withLuaContext(ctx)
                .withWorkers(WORKERS);

        DUUIUIMADriver uima_driver = new DUUIUIMADriver();
        DUUIRemoteDriver remoteDriver = new DUUIRemoteDriver();
        DUUIDockerDriver dockerDriver = new DUUIDockerDriver();

        pComposer.addDriver(
                uima_driver,
                remoteDriver,
                dockerDriver
        );

        System.out.println("INIT done");

    }

    @ParameterizedTest
    @DisplayName("Taxa")
    @ValueSource(strings = {"gbif_backbone","taxref","ncbi_taxonomy"})
    public void taxa(String arg) throws Exception {
        pComposer.resetPipeline();

        pComposer.add(new DUUIRemoteDriver.Component("http://localhost:9715")
                .withScale(WORKERS)
//                    .withConstraintHost("rohan")
                .withParameter("model", "en_ner_eco_md")
                .withParameter("linking", arg)
                .withParameter("exclude", new JSONArray().put("tagger").put("parser").toString())
                .build().withTimeout(7200)
                );

        JCas pCas = generateCAS();

        pComposer.run(pCas);


        JCasUtil.select(pCas, Taxon.class).stream().forEach(t->{
            System.out.println(t.getCoveredText());
            JCasUtil.select(pCas, AnnotationComment.class).stream().filter(da->{
                return da.getReference().equals(t);
            }).forEach(da->{
                System.out.println(da.getKey()+" -> "+da.getValue());
            });
        });


    }

    @Test
    @DisplayName("Taxa")
    public void taxaSingle() throws Exception {
        pComposer.resetPipeline();
        String[] sInput = {"gbif", "taxref", "ncbi"};

        for (String s : sInput) {
            pComposer.add(new DUUIRemoteDriver.Component("http://localhost:9715/"+s)
                    .withWorkers(2)
                    .withTargetView("taxon")
                    .build().withTimeout(7200));
        }

        JCas pCas = generateCAS();

        pComposer.run(pCas);


        JCasUtil.select(pCas, Taxon.class).stream().forEach(t->{
            System.out.println(t.getCoveredText());
            JCasUtil.select(pCas, AnnotationComment.class).stream().filter(da->{
                return da.getReference().equals(t);
            }).forEach(da->{
                System.out.println(da.getKey()+" -> "+da.getValue());
            });
        });


    }

    public JCas generateCAS() throws ResourceInitializationException, CASException {

        JCas pCas = JCasFactory.createText("The common sunflower (Helianthus annuus) is a species of large annual forb of the daisy family Asteraceae. The common sunflower is harvested for its edible oily seeds, which are often eaten as a snack food. They are also used in the production of cooking oil, as food for livestock, as bird food, and as plantings in domestic gardens for aesthetics. Wild plants are known for their multiple flower heads, whereas the domestic sunflower often possesses a single large flower head atop an unbranched stem. The plant has an erect rough-hairy stem, reaching typical heights of 3 metres (10 feet). The tallest sunflower on record achieved 10.9 m (35 ft 9 in). Sunflower leaves are broad, coarsely toothed, rough and mostly alternate; those near the bottom are largest and commonly heart-shaped. The plant flowers in summer. What is often called the \"flower\" of the sunflower is actually a \"flower head\" (pseudanthium), 7.5–12.5 centimetres (3–5 in) wide, of numerous small individual five-petaled flowers (\"florets\"). The outer flowers, which resemble petals, are called ray flowers. Each \"petal\" consists of a ligule composed of fused petals of an asymmetrical ray flower. They are sexually sterile and may be yellow, red, orange, or other colors. The spirally arranged flowers in the center of the head are called disk flowers. These mature into fruit (sunflower \"seeds\"). The prairie sunflower (H. petiolaris) is similar in appearance to the wild common sunflower; the scales in its central disk are tipped by white hairs.", "en");

        return pCas;
    }

    @Test
    public void createDockerCompose(){

        int iScale = 3;



    }

}
