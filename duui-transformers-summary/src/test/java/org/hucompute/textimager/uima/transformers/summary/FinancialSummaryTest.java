package org.hucompute.textimager.uima.transformers.summary;

import de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence;
import org.apache.uima.UIMAException;
import org.apache.uima.fit.factory.JCasFactory;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;
import org.apache.uima.jcas.tcas.Annotation;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.texttechnologylab.DockerUnifiedUIMAInterface.DUUIComposer;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIRemoteDriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaContext;
import org.texttechnologylab.annotation.Summary;

import java.io.IOException;
import java.net.URISyntaxException;
import java.net.UnknownHostException;
import java.util.Collection;

public class FinancialSummaryTest {
    static DUUIComposer composer;
    static String summary_url = "http://127.0.0.1:1000";
    static String model = "Pegasus Financial";

    @BeforeAll
    static void beforeAll() throws URISyntaxException, IOException, UIMAException {
        composer = new DUUIComposer()
                .withSkipVerification(true)
                .withLuaContext(new DUUILuaContext().withJsonLibrary());

        DUUIRemoteDriver remoteDriver = new DUUIRemoteDriver();
        composer.addDriver(remoteDriver);
    }

    @AfterAll
    static void afterAll() throws UnknownHostException {
        composer.shutdown();
    }

    @Test
    public void sentencesTest() throws Exception {
        composer.add(
                new DUUIRemoteDriver.Component(summary_url)
                        .withParameter("model_name", model)
                        .withParameter("summary_length", "60")
        );

        String Text = "Everyone is gloomy about America’s jobs market. Investors talk of a “K-shaped” economy, in which growth is buoyed by an exuberant stockmarket and artificial-intelligence investment, while ordinary Americans languish. Job creation and overall economic growth, which usually move in tandem, have diverged. The Federal Reserve has cut interest rates at its two most recent meetings. Jerome Powell, the central bank’s chair, calls the loosening “risk management”, or insurance against a deeper downturn. Christopher Waller, a contender to replace Mr Powell, is pushing for further and faster cuts, beginning at the next meeting on December 10th, to support a weakening labour market.";
        JCas cas = JCasFactory.createText(Text, "en");

        Annotation sentence1 = new Sentence(cas, 0, 250);
        sentence1.addToIndexes();
        Annotation sentence2 = new Sentence(cas, 251, Text.length());
        sentence2.addToIndexes();

        composer.run(cas);

        Collection<Summary> summaries = JCasUtil.select(cas, Summary.class);

        for (Summary summary : summaries) {
            System.out.println(summary.getSummary());
        }
        assert summaries.size() == 2;
    }
}