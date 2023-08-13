import de.tudarmstadt.ukp.dkpro.core.api.metadata.type.DocumentMetaData;
import de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence;
import io.swagger.models.auth.In;
import org.apache.uima.fit.factory.JCasFactory;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;
import org.apache.uima.util.CasIOUtils;
import org.dkpro.core.io.xmi.XmiWriter;
import org.hucompute.textimager.uima.type.GerVaderSentiment;
import org.junit.jupiter.api.Test;
import org.luaj.vm2.ast.Str;
import org.texttechnologylab.DockerUnifiedUIMAInterface.DUUIComposer;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.*;
import org.texttechnologylab.DockerUnifiedUIMAInterface.io.AsyncCollectionReader;
import org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaContext;
import org.texttechnologylab.annotation.SentimentBert;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;

import static org.apache.uima.fit.factory.AnalysisEngineFactory.createEngineDescription;

public class GermanSentimentTest {

    @Test
    public void annotateSentencesForTestJCas() throws Exception {

        DUUILuaContext ctx = new DUUILuaContext().withJsonLibrary();

        // Instanziierung des Composers, mit einigen Parametern
        DUUIComposer composer = new DUUIComposer()
                .withSkipVerification(true)     // wir überspringen die Verifikation aller Componenten =)
                .withLuaContext(ctx)            // wir setzen den definierten Kontext
                .withWorkers(1);         // wir geben dem Composer eine Anzahl an Threads mit.

        DUUIDockerDriver docker_driver = new DUUIDockerDriver();
        DUUIUIMADriver uima_driver = new DUUIUIMADriver()
                .withDebug(true);

        composer.addDriver(docker_driver, uima_driver);

        composer.add(new DUUIDockerDriver.Component("docker.texttechnologylab.org/textimager-duui-spacy-single-de_core_news_sm:0.1.4")
                .withScale(1)
                .build());

        // Hinzufügen einer UIMA-Componente zum schreiben der Ergebnisse
        composer.add(new DUUIUIMADriver.Component(createEngineDescription(XmiWriter.class,
                XmiWriter.PARAM_TARGET_LOCATION, "./testdata/xmi",
                XmiWriter.PARAM_PRETTY_PRINT, true,
                XmiWriter.PARAM_OVERWRITE, true,
                XmiWriter.PARAM_VERSION, "1.1"
        )).build());


        JCas jc = JCasFactory.createText("Ich liebe Erdbeeren sehr. Ich hasse Oliven, da sie mir nicht schecken. Café");

        DocumentMetaData dmd = DocumentMetaData.create(jc);
        File pFile = new File( "/test_jCas");
        dmd.setDocumentId(pFile.getName());
        dmd.setDocumentTitle(pFile.getName());
        dmd.setDocumentUri(pFile.getAbsolutePath());
        dmd.addToIndexes();

        composer.run(jc, "test");
    }

    @Test
    public void test() throws Exception {

        int iWorkers = 1;
        DUUILuaContext ctx = new DUUILuaContext().withJsonLibrary();

        // Instanziierung des Composers, mit einigen Parametern
        DUUIComposer composer = new DUUIComposer()
                .withSkipVerification(true)     // wir überspringen die Verifikation aller Componenten =)
                .withLuaContext(ctx)            // wir setzen den definierten Kontext
                .withWorkers(iWorkers);         // wir geben dem Composer eine Anzahl an Threads mit.


        DUUIDockerDriver docker_driver = new DUUIDockerDriver();
        DUUIRemoteDriver remoteDriver = new DUUIRemoteDriver(1000);
        composer.addDriver(docker_driver, remoteDriver);


        composer.add(new DUUIDockerDriver.Component("docker.texttechnologylab.org/german-sentiment-bert:latest")
                .withScale(iWorkers)
                .build());

        /*composer.add(new DUUIRemoteDriver.Component("http://localhost:9716")
                .withScale(iWorkers)
                .build());*/

        /*JCas jCas = JCasFactory.createText("Dies ist ein schöner Test. Leider weiß ich nicht weiter.", "de");
        //CasIOUtils.load(new File("/home/gabrami/Downloads/xmiExample/18001.xmi").toURI().toURL(), jCas.getCas());

        new Sentence(jCas, 0, jCas.getDocumentText().indexOf(".")).addToIndexes();
        new Sentence(jCas, jCas.getDocumentText().indexOf("."), jCas.getDocumentText().lastIndexOf(".")).addToIndexes();
         System.out.println(JCasUtil.select(jCas, Sentence.class).size());*/


        JCas jCas = JCasFactory.createJCas();
        CasIOUtils.load(new File("testdata/xmi/test_jCas.xmi").toURI().toURL(), jCas.getCas());
        composer.run(jCas, "test");

        for(SentimentBert sentimentBert: JCasUtil.select(jCas, SentimentBert.class)){
            System.out.println(sentimentBert.getCoveredText() + ": " + sentimentBert.getSentiment());
        }

    }

    @Test
    public void testWeirdDocLength() throws Exception {

        int iWorkers = 1;
        DUUILuaContext ctx = new DUUILuaContext().withJsonLibrary();

        // Instanziierung des Composers, mit einigen Parametern
        DUUIComposer composer = new DUUIComposer()
                .withSkipVerification(true)     // wir überspringen die Verifikation aller Componenten =)
                .withLuaContext(ctx)            // wir setzen den definierten Kontext
                .withWorkers(iWorkers);         // wir geben dem Composer eine Anzahl an Threads mit.

        DUUIDockerDriver docker_driver = new DUUIDockerDriver();
        DUUIRemoteDriver remoteDriver = new DUUIRemoteDriver(1000);
        composer.addDriver(docker_driver, remoteDriver);

        composer.add(new DUUIRemoteDriver.Component("http://localhost:9716")
                .withScale(iWorkers)
                .build());

        JCas jCas = JCasFactory.createJCas();
        CasIOUtils.load(new File("testdata/xmi/1._Sitzung_30.01.1937.xmi.gz.xmi").toURI().toURL(), jCas.getCas());
        composer.run(jCas, "test");


        for(SentimentBert sentimentBert: JCasUtil.select(jCas, SentimentBert.class)){
            System.out.println(sentimentBert.getBegin() + " " + sentimentBert.getEnd());
            //System.out.println(sentimentBert.getCoveredText() + ": " + sentimentBert.getSentiment());
        }
        System.out.println(jCas.getDocumentText().length());

    }

    @Test
    public void testWithAsyncReaderAndXmiRider() throws Exception {

        // Input- und Output-Pfade
        String sInputPath = "./testdata/xmi";
        String sOutputPath = "./testdata/xmi_output";
        String sSuffix = "xmi";

        int workers = 1;

        // Asynchroner reader für die Input-Dateien
        AsyncCollectionReader pCorpusReader = new AsyncCollectionReader(sInputPath, sSuffix, 1, false);
        new File(sOutputPath).mkdir();

        DUUILuaContext ctx = new DUUILuaContext().withJsonLibrary();

        // Instanziierung des Composers, mit einigen Parametern
        DUUIComposer composer = new DUUIComposer()
                .withSkipVerification(true)     // wir überspringen die Verifikation aller Componenten =)
                .withLuaContext(ctx)            // wir setzen den definierten Kontext
                .withWorkers(workers);         // wir geben dem Composer eine Anzahl an Threads mit.

        DUUIDockerDriver docker_driver = new DUUIDockerDriver();
        DUUIRemoteDriver remoteDriver = new DUUIRemoteDriver();
        DUUIUIMADriver uima_driver = new DUUIUIMADriver()
                .withDebug(false);
        composer.addDriver(docker_driver, remoteDriver, uima_driver);

        composer.add(new DUUIDockerDriver.Component("docker.texttechnologylab.org/german-sentiment-bert:latest")
                .withScale(workers)
                .build());

        /*composer.add(new DUUIRemoteDriver.Component("http://localhost:9714")
                .withScale(1)
                .build());*/

        // Hinzufügen einer UIMA-Componente zum schreiben der Ergebnisse
        composer.add(new DUUIUIMADriver.Component(createEngineDescription(XmiWriter.class,
                XmiWriter.PARAM_TARGET_LOCATION, sOutputPath,
                XmiWriter.PARAM_PRETTY_PRINT, true,
                XmiWriter.PARAM_OVERWRITE, true,
                XmiWriter.PARAM_VERSION, "1.1"
        )).build());

        long startTime = System.nanoTime();
        composer.run(pCorpusReader, "test");
        long endTime = System.nanoTime();
        System.out.println("Time passed: " + formatNanoSeconds(endTime-startTime));

    }

    @Test
    public void analyseParlamentDataSentimentBert() throws Exception {

        // Input- und Output-Pfade
        String sInputPath = "/home/max/uni/testdata/input/ThirdReich";
        String sOutputPath = "/home/max/uni/testdata/output/ThirdReich/SentimentBert";
        String sSuffix = "xmi.gz";

        int workers = 1;

        // Asynchroner reader für die Input-Dateien
        AsyncCollectionReader pCorpusReader = new AsyncCollectionReader(sInputPath, sSuffix, 1, false);
        new File(sOutputPath).mkdir();

        DUUILuaContext ctx = new DUUILuaContext().withJsonLibrary();

        // Instanziierung des Composers, mit einigen Parametern
        DUUIComposer composer = new DUUIComposer()
                .withSkipVerification(true)     // wir überspringen die Verifikation aller Componenten =)
                .withLuaContext(ctx)            // wir setzen den definierten Kontext
                .withWorkers(workers);         // wir geben dem Composer eine Anzahl an Threads mit.

        DUUIDockerDriver docker_driver = new DUUIDockerDriver();
        DUUIUIMADriver uima_driver = new DUUIUIMADriver()
                .withDebug(false);
        DUUIRemoteDriver duuiRemoteDriver = new DUUIRemoteDriver();
        composer.addDriver(docker_driver, uima_driver, duuiRemoteDriver);

        composer.add(new DUUIRemoteDriver.Component("http://localhost:9714")
                .withScale(workers)
                .build());

        /*composer.add(new DUUIDockerDriver.Component("docker.texttechnologylab.org/german-sentiment-bert:latest")
                .withScale(workers)
                .build());*/

        // Hinzufügen einer UIMA-Componente zum schreiben der Ergebnisse
        composer.add(new DUUIUIMADriver.Component(createEngineDescription(XmiWriter.class,
                XmiWriter.PARAM_TARGET_LOCATION, sOutputPath,
                XmiWriter.PARAM_PRETTY_PRINT, true,
                XmiWriter.PARAM_OVERWRITE, true,
                XmiWriter.PARAM_VERSION, "1.1"
        )).build());

        long startTime = System.nanoTime();
        composer.run(pCorpusReader, "test");
        long endTime = System.nanoTime();
        System.out.println("Time passed: " + formatNanoSeconds(endTime-startTime));

    }

    @Test
    public void analyseParlamentDataGerVader() throws Exception {

        // Input- und Output-Pfade
        String sInputPath = "/home/max/uni/testdata/input/ThirdReich";
        String sOutputPath = "/home/max/uni/testdata/output/ThirdReich/GerVader";
        String sSuffix = "xmi.gz";

        int workers = 1;

        // Asynchroner reader für die Input-Dateien
        AsyncCollectionReader pCorpusReader = new AsyncCollectionReader(sInputPath, sSuffix, 1, false);
        new File(sOutputPath).mkdir();

        DUUILuaContext ctx = new DUUILuaContext().withJsonLibrary();

        // Instanziierung des Composers, mit einigen Parametern
        DUUIComposer composer = new DUUIComposer()
                .withSkipVerification(true)     // wir überspringen die Verifikation aller Componenten =)
                .withLuaContext(ctx)            // wir setzen den definierten Kontext
                .withWorkers(workers);         // wir geben dem Composer eine Anzahl an Threads mit.

        DUUIDockerDriver docker_driver = new DUUIDockerDriver();
        DUUIUIMADriver uima_driver = new DUUIUIMADriver()
                .withDebug(false);
        DUUIRemoteDriver duuiRemoteDriver = new DUUIRemoteDriver();
        composer.addDriver(docker_driver, uima_driver, duuiRemoteDriver);

        composer.add(new DUUIRemoteDriver.Component("http://127.0.0.1:9714")
                .withScale(workers)
                .build());


        /*composer.add(new DUUIDockerDriver.Component("docker.texttechnologylab.org/gervader_duui:latest")
                .withScale(workers)
                .build());*/

        // Hinzufügen einer UIMA-Componente zum schreiben der Ergebnisse
        composer.add(new DUUIUIMADriver.Component(createEngineDescription(XmiWriter.class,
                XmiWriter.PARAM_TARGET_LOCATION, sOutputPath,
                XmiWriter.PARAM_PRETTY_PRINT, true,
                XmiWriter.PARAM_OVERWRITE, true,
                XmiWriter.PARAM_VERSION, "1.1"
        )).build());

        long startTime = System.nanoTime();
        composer.run(pCorpusReader, "test");
        long endTime = System.nanoTime();
        System.out.println("Time passed: " + formatNanoSeconds(endTime-startTime));

    }

    @Test
    public void compareBertVsVader() throws Exception {
        String sOutputPath = "/home/max/uni/testdata/output/ThirdReich/";

        Map<String, Map<String, String>> fileModelOutputMap = new HashMap<>();

        for(String modelName: new String[]{"SentimentBert", "GerVader"}){
            Files.walk(Paths.get(sOutputPath + modelName)).forEach((Path path) -> {
                if (Files.isRegularFile(path)){
                    File file = path.toFile();
                    Map<String, String> modelNameOutputPathMap = fileModelOutputMap.get(file.getName());
                    if (modelNameOutputPathMap == null){
                        modelNameOutputPathMap = new HashMap<>();
                        fileModelOutputMap.put(file.getName(), modelNameOutputPathMap);
                    }
                    modelNameOutputPathMap.put(modelName, file.getAbsolutePath());
                }
            });
        }

        Map<String, Map<String, Double>> sentenceModelNameSentientMap = new HashMap<>();

        for (Map<String, String> modelNameOutputPathMap: fileModelOutputMap.values()) {

            for (String modelName: modelNameOutputPathMap.keySet()){
                File file = new File(modelNameOutputPathMap.get(modelName));
                System.out.println(file.getAbsolutePath());
                JCas jCas = JCasFactory.createJCas();
                CasIOUtils.load(file.toURI().toURL(), jCas.getCas());
                System.out.println(jCas.getDocumentText().length());
                System.out.println();

                if(modelName == "SentimentBert"){
                    for(SentimentBert sentimentBert: JCasUtil.select(jCas, SentimentBert.class)){
                        Map<String, Double> modelSentimentMap = sentenceModelNameSentientMap.get(sentimentBert.getCoveredText());
                        if (modelSentimentMap == null){
                            modelSentimentMap = new HashMap<>();
                            sentenceModelNameSentientMap.put(sentimentBert.getCoveredText(), modelSentimentMap);
                        }
                        modelSentimentMap.put(modelName, new Double(sentimentBert.getSentiment()));
                    }
                }
                else{
                    for(GerVaderSentiment sentimentVader: JCasUtil.select(jCas, GerVaderSentiment.class)){
                        Map<String, Double> modelSentimentMap = sentenceModelNameSentientMap.get(sentimentVader.getCoveredText());
                        if (modelSentimentMap == null){
                            modelSentimentMap = new HashMap<>();
                            sentenceModelNameSentientMap.put(sentimentVader.getCoveredText(), modelSentimentMap);
                        }
                        modelSentimentMap.put(modelName, sentimentVader.getSentiment());
                    }
                }

            }

        }


        int i = 0;
        HashMap<Double, HashMap<Double, Integer>> bert_vader_sentiment_count_map = new HashMap<>();
        bert_vader_sentiment_count_map.put(-1.0, new HashMap<>());
        bert_vader_sentiment_count_map.put(0.0, new HashMap<>());
        bert_vader_sentiment_count_map.put(1.0, new HashMap<>());

        for (String sentence: sentenceModelNameSentientMap.keySet()){
            Map<String, Double> modelSentimentMap = sentenceModelNameSentientMap.get(sentence);
            Double sentimentBert = modelSentimentMap.get("SentimentBert");
            Double sentimentVader = modelSentimentMap.get("GerVader");
            Double sentimentVaderRounded = 0.0;
            if(sentimentVader > 0){
                sentimentVaderRounded = 1.0;
            } else if (sentimentVader < 0) {
                sentimentVaderRounded = -1.0;
            }

            HashMap<Double, Integer> vader_sentiment_count_map = bert_vader_sentiment_count_map.get(sentimentBert);

            Integer currentCount = vader_sentiment_count_map.get(sentimentVaderRounded);
            if(currentCount == null){
                currentCount = 0;
            }
            vader_sentiment_count_map.put(sentimentVaderRounded, currentCount + 1);
        }


        for(Double sentimentBert: bert_vader_sentiment_count_map.keySet()){
            HashMap<Double, Integer> vader_sentiment_count_map = bert_vader_sentiment_count_map.get(sentimentBert);
            System.out.println(sentimentBert);
            for(Double sentimentVader: vader_sentiment_count_map.keySet()){
                int count = vader_sentiment_count_map.get(sentimentVader);
                System.out.println("\t" + sentimentVader + " " + count);
            }
            System.out.println();
        }


        /*J*/
    }

    public static String formatNanoSeconds(long nanoSeconds){
        double runTime = Math.round(((double) nanoSeconds) / 10_000_000.0) / 100.0;
        return runTime + "s";
    }


}
