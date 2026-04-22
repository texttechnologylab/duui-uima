import org.apache.commons.compress.compressors.CompressorException;
import org.apache.uima.UIMAException;
import org.apache.uima.cas.CASException;
import org.apache.uima.fit.factory.JCasFactory;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;
import org.apache.uima.resource.ResourceInitializationException;
import org.apache.uima.util.InvalidXMLException;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.texttechnologylab.DockerUnifiedUIMAInterface.DUUIComposer;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIDockerDriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIRemoteDriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIUIMADriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaContext;
import org.texttechnologylab.annotation.neglab.ConditionSentence;
import org.texttechnologylab.annotation.neglab.TokenSuprisal;
import org.texttechnologylab.utilities.helper.FileUtils;
import org.xml.sax.SAXException;

import java.io.File;
import java.io.IOException;
import java.net.URISyntaxException;
import java.nio.file.Files;

/**
 * This script calculates surprisal values at the sentence and token level. It also calculates the surprisal value of a target word given a context
 * @author Giuseppe Abrami
 */
public class Suprisal {

    private static DUUIComposer pComposer = null;
    private static int iWorkers = 1;

    /**
     * Initialization of DUUI for each test, saves lines of code.
     *
     * @throws IOException
     * @throws URISyntaxException
     * @throws UIMAException
     * @throws SAXException
     */
    @BeforeAll
    public static void init() throws IOException, URISyntaxException, UIMAException, SAXException {

        DUUILuaContext ctx = new DUUILuaContext().withJsonLibrary();

        pComposer = new DUUIComposer()
                .withSkipVerification(true)
                .withLuaContext(ctx)
                .withWorkers(iWorkers);

        DUUIUIMADriver uima_driver = new DUUIUIMADriver();
        DUUIRemoteDriver remoteDriver = new DUUIRemoteDriver();
        DUUIDockerDriver dockerDriver = new DUUIDockerDriver();

        pComposer.addDriver(uima_driver, remoteDriver, dockerDriver);

    }

    /**
     * Example Test case
     * @throws Exception
     */
    @Test
    public void example() throws Exception {

        /**
         * Input-Path's ..
         *  .. using Windows: C:\\directory\\to\\your\\inputdata\\data.csv
         *  .. using Linux / MacOS: /directory/to/your/inputdata/data.csv
         * data.csv can also be named differently.
         *
         * Output-Path's ..
         *  .. using Windows: C:\\directory\\to\\your\\output
         *  .. using Linux / MacOS: /directory/to/your/output
         *
         *  It is also possible to store the input-file in the resource path of the project, which can then be accessed with a real path: this.getClass().getResource("name_of_file.csv").getPath()
         */
//        executeSuprisal(this.getClass().getResource("input.csv").getPath(),"/tmp/suprisal", "en", "google/gemma-3-4b-it", "");
        executeSuprisal(this.getClass().getResource("input.csv").getPath(),"/tmp/suprisal", "en", "goldfish-models/spa_latn_1000mb", "");

    }

    /**
     * Execute Suprisal
     * @param sOutputPath
     * @param sModel
     * @param sToken
     * @throws Exception
     */
    private void executeSuprisal(String sOutputPath, String sModel, String sToken) throws Exception {
        executeSuprisal(new File(this.getClass().getResource("input.csv").getFile()).getPath(), sOutputPath, "en", sModel, sToken);
    }

    /**
     * Execute Suprisal
     * @param sOutputPath
     * @param sModel
     * @throws Exception
     */
    private void executeSuprisal(String sOutputPath, String sModel) throws Exception {
        executeSuprisal(sOutputPath, sModel, "");
    }


    /**
     * Execute Suprisal
     * @param sInputPath
     * @param sOutputPath
     * @param sLanguage
     * @param sModel
     * @param sToken
     * @throws Exception
     */
    private void executeSuprisal(String sInputPath, String sOutputPath, String sLanguage, String sModel, String sToken) throws Exception {
        executeSuprisal(sInputPath, sOutputPath, sLanguage, "http://suprisal.duui.neglab.de", sModel, sToken);
//        executeSuprisal(sInputPath, sOutputPath, sLanguage, "http://localhost:9715", sModel, sToken);
    }

    /**
     * Execute Suprisal
     * @param sInputPath
     * @param sOutputPath
     * @param sLanguage
     * @param sURL
     * @param sModel
     * @param sToken
     * @throws Exception
     */
    private void executeSuprisal(String sInputPath, String sOutputPath, String sLanguage, String sURL, String sModel, String sToken) throws Exception {

        JCas jCas = JCasFactory.createJCas();

        /**
         * Converting CSV to XMI
         */
        String sContent = FileUtils.getContentFromFile(new File(sInputPath));

        StringBuilder sb = new StringBuilder();

        String[] sSplit = sContent.split("\n");

        for (int i = 1; i < sSplit.length; i++) {
            String sValue = sSplit[i];

            if(sValue.length()>0) {
                String[] tSplit = sValue.split(",");

                if(sb.length()>0){
                    sb.append("\n");
                }
                int iStart = sb.length();
                sb.append(tSplit[2]);
                int iEnd = sb.length();

                // Create an annotation of type ConditionSentence for each sentence.
                ConditionSentence cs = new ConditionSentence(jCas);
                cs.setBegin(iStart);
                cs.setEnd(iEnd);
                cs.setCondition(tSplit[1]);
                cs.setOrder(Integer.parseInt(tSplit[0]));
                cs.setTarget(tSplit[3]);
                cs.addToIndexes();
            }
        }

        // Add the merged text to the cas.
        jCas.setDocumentText(sb.toString());
        jCas.setDocumentLanguage(sLanguage);


        if(sToken.length()>0 && sToken!=null){
            // or using (as far as it is online) the remote-variant
            pComposer.add(new DUUIRemoteDriver.Component(sURL)
                    .withParameter("model", sModel)
                    .withParameter("token_authentication", sToken)
            ).withWorkers(1);

        }
        else{
            // or using (as far as it is online) the remote-variant
            pComposer.add(new DUUIRemoteDriver.Component(sURL)
                    .withParameter("model", sModel)
            ).withWorkers(1);

        }

        // execute the pipeline
        pComposer.run(jCas);

        /**
         * Save results in CSV-files
         */
        writeSentenceResults(jCas, sOutputPath.endsWith("/") ? sOutputPath : sOutputPath+"/"+"export_Sentence.csv");
        writeTokenResults(jCas, sOutputPath.endsWith("/") ? sOutputPath : sOutputPath+"/"+"export_Token.csv");

    }

    private void writeSentenceResults(JCas jCas, String sOutPath) throws IOException {

        StringBuilder outputBuilder = new StringBuilder();
        outputBuilder.append("Item,Condition,Sentence,Target,Score,ScoreSum");

        JCasUtil.select(jCas, ConditionSentence.class).stream().forEach(sentence -> {
            if(outputBuilder.length()>0) {
                outputBuilder.append("\n");
            }
            outputBuilder.append(sentence.getOrder()+","+sentence.getCoveredText()+","+sentence.getCondition()+","+sentence.getTarget()+","+sentence.getValue()+","+sentence.getSequenceScore()+","+sentence.getSequenceScoreSum());

        });

        File pFile = new File(sOutPath);
        if(!pFile.getParentFile().exists()){
            Files.createDirectories(pFile.getParentFile().toPath());
        }

        // write into file
        FileUtils.writeContent(outputBuilder.toString(), new  File(sOutPath));

    }

    private void writeTokenResults(JCas jCas, String sOutPath) throws IOException {

        StringBuilder outputBuilder = new StringBuilder();
        outputBuilder.append("Token,Begin,End,Value,Sentence");

        JCasUtil.select(jCas, ConditionSentence.class).stream().forEach(sentence -> {
            JCasUtil.selectCovered(TokenSuprisal.class, sentence).stream().forEach(t->{
                if(outputBuilder.length()>0) {
                    outputBuilder.append("\n");
                }
                outputBuilder.append(t.getCoveredText()+","+t.getBegin()+","+t.getEnd()+","+t.getValue()+","+sentence.getCoveredText());
            });

        });

        // write into file
        FileUtils.writeContent(outputBuilder.toString(), new  File(sOutPath));

    }

}
