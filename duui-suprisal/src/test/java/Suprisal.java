import de.tudarmstadt.ukp.dkpro.core.api.metadata.type.DocumentMetaData;
import org.apache.uima.UIMAException;
import org.apache.uima.fit.factory.JCasFactory;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;
import org.dkpro.core.io.xmi.XmiWriter;
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

import static org.apache.uima.fit.factory.AnalysisEngineFactory.createEngineDescription;

/**
 * Test case for the execution of DUUI-Suprisal
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

        JCas jCas = JCasFactory.createJCas();

        /**
         * Converting CSV to XMI
         */
        String sContent = FileUtils.getContentFromFile(new File(getClass().getResource("test.csv").getFile()));

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
        jCas.setDocumentLanguage("es");

        // See all the existing Annotations
//        JCasUtil.select(jCas, ConditionSentence.class).stream().forEach(sentence -> {
//            System.out.println(sentence.getCoveredText()+"\t"+sentence.getCondition()+"\t"+sentence.getTarget());
//        });

        // use the docker-component
//        pComposer.add(new DUUIDockerDriver.Component("duui-suprisal:latest").build()
//                .withWorkers(1));

       /**
        * use the remote-component
        *
        * The component was started as a Docker container with port 9715.
        * ''docker run --rm -p 9715:9714 docker.texttechnologylab.org/duui-suprisal:latest''
        */
//        pComposer.add(new DUUIRemoteDriver.Component("http://localhost:9715")
////                .withParameter("model", "goldfish-models/spa_latn_1000mb")
//                .withParameter("model", "google/gemma-3-4b-pt")
//        ).withWorkers(1);

//        pComposer.add(new DUUIUIMADriver.Component(createEngineDescription(XmiWriter.class,
//                XmiWriter.PARAM_TARGET_LOCATION, "/tmp/",
//                XmiWriter.PARAM_OVERWRITE, true,
//                XmiWriter.PARAM_VERSION, "1.1",
//                XmiWriter.PARAM_PRETTY_PRINT, true
//
//        )).withScale(iWorkers)
//                .build());

        // or using (as far as it is online) the remote-variant
        pComposer.add(new DUUIRemoteDriver.Component("http://suprisal.duui.neglab.de")
                .withParameter("model", "goldfish-models/spa_latn_1000mb")
//                .withParameter("token", "myToken") // if nessecary
        ).withWorkers(1);

        // execute the pipeline
        pComposer.run(jCas);

        // Write as CSV
        writeSentenceResults(jCas, "/tmp/export_Sentence.csv");
        writeTokenResults(jCas, "/tmp/export_Token.csv");

    }

    public void writeSentenceResults(JCas jCas, String sOutPath) throws IOException {

        StringBuilder outputBuilder = new StringBuilder();
        outputBuilder.append("Item,Condition,Sentence,Target,Score,ScoreSum");

        JCasUtil.select(jCas, ConditionSentence.class).stream().forEach(sentence -> {
            if(outputBuilder.length()>0) {
                outputBuilder.append("\n");
            }
            outputBuilder.append(sentence.getOrder()+","+sentence.getCoveredText()+","+sentence.getCondition()+","+sentence.getTarget()+","+sentence.getValue()+","+sentence.getSequenceScore()+","+sentence.getSequenceScoreSum());

        });

        // write into file
        FileUtils.writeContent(outputBuilder.toString(), new  File(sOutPath));

    }

    public void writeTokenResults(JCas jCas, String sOutPath) throws IOException {

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
