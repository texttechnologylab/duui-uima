package org.hucompute.textimager.uima;

import org.apache.uima.cas.impl.XmiCasSerializer;
import org.apache.uima.fit.factory.JCasFactory;
import org.apache.uima.fit.util.CasIOUtil;
import org.apache.uima.jcas.JCas;
import org.apache.uima.util.XMLSerializer;
import org.junit.jupiter.api.Test;
import org.texttechnologylab.DockerUnifiedUIMAInterface.DUUIComposer;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIRemoteDriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaContext;
//import org.texttechnologylab.annotation.GeoNamesEntity;
//import de.unihd.dbs.uima.annotator.heideltime.biofid.HeidelTimeBioFID;
import org.xml.sax.SAXException;

import javax.xml.transform.OutputKeys;
import java.io.File;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import static org.apache.uima.fit.factory.AnalysisEngineFactory.createEngineDescription;

public class SRLTest {
    @Test
    public void multiTestEn() throws Exception {

        JCas jCas = JCasFactory.createJCas();

//        CasIOUtil.readXmi(jCas.getCas(), new File(("/home/staff_homes/mkonca/tmp/45808_spacy_gaz_geo_hei.xmi")));
        CasIOUtil.readXmi(jCas.getCas(), new File(("/home/mkonca/tmp/tst.xmi")));
//        CasIOUtil.readXmi(jCas.getCas(), new File(("/home/mkonca/tmp/sentences_sampled_gutenberg.txt.xmi")));
//        CasIOUtil.readXmi(jCas.getCas(), new File(("/home/mkonca/tmp/1.xmi")));
//        CasIOUtil.readXmi(jCas.getCas(), new File(("/home/mkonca/tmp/tst.xmi")));
//        CasIOUtil.readXmi(jCas.getCas(), new File(("/home/mkonca/tmp/45808mini_spacy.xmi")));
//        CasIOUtil.readXmi(jCas.getCas(), new File(("/home/mkonca/tmp/ex/tnl_full/567_1061.txt.xmi")));

        jCas.setDocumentLanguage("de");
//        System.out.println(jCas.getSofa().toString().length());

        DUUIComposer composer = new DUUIComposer()
                .withLuaContext(
                        new DUUILuaContext()
                                .withJsonLibrary()
                ).withSkipVerification(true);
        DUUIRemoteDriver remote_driver = new DUUIRemoteDriver(10000);
        composer.addDriver(remote_driver);
//        DUUIDockerDriver docker_driver = new DUUIDockerDriver();
//        composer.addDriver(docker_driver);

//        composer.add(
//            new DUUIRemoteDriver.Component("http://127.0.0.1:9715")
//        );

//        composer.add(new DUUIDockerDriver.
//                Component("bfsrl:0.1.0")
//                Component("docker.texttechnologylab.org/bfsrl:latest")
//                .withScale(1)
//                .withImageFetching());

//        String model = "cardiffnlp/twitter-xlm-roberta-base-sentiment";
//        String model = "cardiffnlp/twitter-xlm-roberta-base-sentiment";
//
        String model = "chkla/parlbert-topic-german";
        composer.add(
                new DUUIRemoteDriver.Component("http://127.0.0.1:8501")
                        .withParameter("model_name", model)
                        .withParameter("selection", "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence")
//                        .withParameter("selection", "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence")
        );

//        composer.add(
//                new DUUIRemoteDriver.Component("http://127.0.0.1:8501")
//        );

        //String GeonamesDocker = "docker.texttechnologylab.org/gazetteer-rs/geonames:latest";
//        composer.add(
//            new DUUIRemoteDriver.Component("http://127.0.0.1:8502")
//        );

//        DUUIUIMADriver duuiuimaDriver = new DUUIUIMADriver();
//        composer.addDriver(duuiuimaDriver);
//        composer.add(new DUUIUIMADriver.Component(
//                createEngineDescription(HeidelTimeBioFID.class
//                )));
/*
        DUUIUIMADriver duuiuimaDriver = new DUUIUIMADriver();
        composer.addDriver(duuiuimaDriver);
        String sPOSMapFile = BFSRLTest.class.getClassLoader().getResource("am_posmap.txt").getPath();
        composer.add(new DUUIUIMADriver.Component(
                createEngineDescription(GeonamesGazetteer.class,
                        GeonamesGazetteer.PARAM_TAGGING_TYPE_NAME, GeoNamesEntity.class.getName(),
                        GeonamesGazetteer.PARAM_MAPPING_PROVIDER_LOCATION, sPOSMapFile,
                        GeonamesGazetteer.PARAM_USE_LOWERCASE, false,
                        GeonamesGazetteer.PARAM_USE_STRING_TREE, true,
                        GeonamesGazetteer.PARAM_USE_SENTECE_LEVEL_TAGGING, true,
                        GeonamesGazetteer.PARAM_USE_LEMMATA, true,
                        GeonamesGazetteer.PARAM_NO_SKIPGRAMS, true,
                        GeonamesGazetteer.PARAM_ADD_ABBREVIATED_TAXA, false,
                        GeonamesGazetteer.PARAM_GET_ALL_SKIPS, false,
                        GeonamesGazetteer.PARAM_MIN_LENGTH, 1,
                        GeonamesGazetteer.PARAM_SPLIT_HYPEN, false,
                        GeonamesGazetteer.PARAM_ANNOTATION_COMMENTS, new String[]{ "ttlab_model", "ttlab_geonames_v_1.0.1" })
                ));*/


        composer.run(jCas);


        composer.shutdown();

//        Path outputXmi = Paths.get("src/test/resources/test.xmi");
        Path outputXmi = Paths.get("/home/mkonca/tmp/out.xmi");
//        Path outputXmi = Paths.get("/home/staff_homes/mkonca/tmp/45808_spacy_gaz_geo_hei_bfsrl.xmi");
        try (OutputStream outputStream = Files.newOutputStream(outputXmi)) {
            XMLSerializer xmlSerializer = new XMLSerializer(outputStream, true);
            xmlSerializer.setOutputProperty(OutputKeys.VERSION, "1.1");
            xmlSerializer.setOutputProperty(OutputKeys.ENCODING, StandardCharsets.UTF_8.toString());
            XmiCasSerializer xmiCasSerializer = new XmiCasSerializer(null);
            xmiCasSerializer.serialize(jCas.getCas(), xmlSerializer.getContentHandler());
        } catch (SAXException e) {
            e.printStackTrace();
        }

//        jCas.reset();
//        JCas jCas1 = JCasFactory.createJCas();
//        CasIOUtil.readXmi(jCas1.getCas(), new File(("/home/mkonca/tmp/tst.xmi")));
//        jCas1.setDocumentLanguage("de");
//        composer.run(jCas1);

//        outputXmi = Paths.get("/home/mkonca/tmp/out1.xmi");
//        Path outputXmi = Paths.get("/home/staff_homes/mkonca/tmp/45808_spacy_gaz_geo_hei_bfsrl.xmi");
//        try (OutputStream outputStream = Files.newOutputStream(outputXmi)) {
//            XMLSerializer xmlSerializer = new XMLSerializer(outputStream, true);
//            xmlSerializer.setOutputProperty(OutputKeys.VERSION, "1.1");
//            xmlSerializer.setOutputProperty(OutputKeys.ENCODING, StandardCharsets.UTF_8.toString());
//            XmiCasSerializer xmiCasSerializer = new XmiCasSerializer(null);
//            xmiCasSerializer.serialize(jCas1.getCas(), xmlSerializer.getContentHandler());
//        } catch (SAXException e) {
//            e.printStackTrace();
//        }
//        jCas1.reset();
//
//        jCas = JCasFactory.createJCas();
//        CasIOUtil.readXmi(jCas.getCas(), new File(("/home/mkonca/tmp/tst.xmi")));
//        jCas.setDocumentLanguage("de");
//        composer.run(jCas);
//
//        outputXmi = Paths.get("/home/mkonca/tmp/out2.xmi");
//        Path outputXmi = Paths.get("/home/staff_homes/mkonca/tmp/45808_spacy_gaz_geo_hei_bfsrl.xmi");
//        try (OutputStream outputStream = Files.newOutputStream(outputXmi)) {
//            XMLSerializer xmlSerializer = new XMLSerializer(outputStream, true);
//            xmlSerializer.setOutputProperty(OutputKeys.VERSION, "1.1");
//            xmlSerializer.setOutputProperty(OutputKeys.ENCODING, StandardCharsets.UTF_8.toString());
//            XmiCasSerializer xmiCasSerializer = new XmiCasSerializer(null);
//            xmiCasSerializer.serialize(jCas.getCas(), xmlSerializer.getContentHandler());
//        } catch (SAXException e) {
//            e.printStackTrace();
//        }
    }
}
