package org.hucompute.textimager.uima;

import org.apache.commons.io.FileUtils;
import org.apache.uima.cas.impl.XmiCasSerializer;
import org.apache.uima.fit.factory.JCasFactory;
import org.apache.uima.fit.util.CasIOUtil;
import org.apache.uima.jcas.JCas;
import org.apache.uima.util.XMLSerializer;
import org.junit.jupiter.api.Test;
import org.texttechnologylab.DockerUnifiedUIMAInterface.DUUIComposer;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIRemoteDriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIUIMADriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaContext;
import org.texttechnologylab.annotation.GeoNamesEntity;
//import de.unihd.dbs.uima.annotator.heideltime.biofid.HeidelTimeBioFID;
import org.xml.sax.SAXException;

import javax.xml.transform.OutputKeys;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.stream.Stream;
import java.util.zip.GZIPOutputStream;

import static org.apache.uima.fit.factory.AnalysisEngineFactory.createEngineDescription;

public class BFSRLTestmulti {
    @Test
    public void multiTest() throws Exception {
//        try (Stream<Path> paths = Files.walk(Paths.get("/home/mkonca/tmp/ex/tnl"))) {
//            paths
//                    .filter(Files::isRegularFile)
//                    .forEach(System.out::println);
//        }
        File folder = new File("/home/mkonca/tmp/ex/tnl");
        for (File file : folder.listFiles()) {
            if (file.isFile() && file.getName().endsWith(".txt")) {
                System.out.println(file.toString());
                InputStream in = new FileInputStream(file);
                String content = FileUtils.readFileToString(file);
                System.out.println(content.length());
                System.out.println("**********************************");
                JCas jCas = JCasFactory.createJCas();


                DUUIComposer composer = new DUUIComposer()
                        .withLuaContext(
                                new DUUILuaContext()
                                        .withJsonLibrary()
                        ).withSkipVerification(true);
                DUUIRemoteDriver remote_driver = new DUUIRemoteDriver(10000);
                composer.addDriver(remote_driver);

                //GAZETEER
                composer.add(
                    new DUUIRemoteDriver.Component("http://127.0.0.1:8503")
                );

                //SPACY
                composer.add(
                        new DUUIRemoteDriver.Component("http://127.0.0.1:8504")
                );
                jCas.setDocumentText(content);
                //String GeonamesDocker = "docker.texttechnologylab.org/gazetteer-rs/geonames:latest";
        //        composer.add(
        //            new DUUIRemoteDriver.Component("http://127.0.0.1:8502")
        //        );

                DUUIUIMADriver duuiuimaDriver = new DUUIUIMADriver();
                composer.addDriver(duuiuimaDriver);
//                composer.add(new DUUIUIMADriver.Component(
//                        createEngineDescription(HeidelTimeBioFID.class
//                        )));
//                DUUIUIMADriver duuiuimaDriver = new DUUIUIMADriver();
//                composer.addDriver(duuiuimaDriver);
                jCas.setDocumentLanguage("de");
                composer.run(jCas);

                composer.shutdown();

                Boolean gzip = true;
                if (gzip == true) {
//                    Path outputXmi = Paths.get("/mnt/ssd/SRL/data/biofid_new/out/" + file.getName());
                    Path outputXmi = Paths.get("/home/mkonca/tmp/ex/tnl_chunks_xmi/" + file.getName() + ".xmi.gz");
                    try (OutputStream outputStream = (new GZIPOutputStream(Files.newOutputStream(outputXmi)))) {
                        XMLSerializer xmlSerializer = new XMLSerializer(outputStream, true);
                        xmlSerializer.setOutputProperty(OutputKeys.VERSION, "1.0");
                        xmlSerializer.setOutputProperty(OutputKeys.ENCODING, StandardCharsets.UTF_8.toString());
                        XmiCasSerializer xmiCasSerializer = new XmiCasSerializer(null);
                        xmiCasSerializer.serialize(jCas.getCas(), xmlSerializer.getContentHandler());
                    } catch (SAXException e) {
                        e.printStackTrace();
                    }
                }
                else{
                        Path outputXmi = Paths.get("/home/mkonca/tmp/ex/tnl_chunks_xmi/" + file.getName() + ".xmi");
                        try (OutputStream outputStream = Files.newOutputStream(outputXmi)) {
                            XMLSerializer xmlSerializer = new XMLSerializer(outputStream, true);
                            xmlSerializer.setOutputProperty(OutputKeys.VERSION, "1.0");
                            xmlSerializer.setOutputProperty(OutputKeys.ENCODING, StandardCharsets.UTF_8.toString());
                            XmiCasSerializer xmiCasSerializer = new XmiCasSerializer(null);
                            xmiCasSerializer.serialize(jCas.getCas(), xmlSerializer.getContentHandler());
                        } catch (SAXException e) {
                            e.printStackTrace();
                        }
                    }

            }
        }
    }
}
