package org.hucompute.textimager.uima.factchecking;
import org.apache.uima.fit.factory.JCasFactory;
import org.apache.uima.fit.util.JCasUtil;
//import org.hucompute.textimager.uima.util.XmlFormatter;
import org.apache.uima.jcas.JCas;
import org.apache.uima.jcas.cas.FSArray;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.texttechnologylab.DockerUnifiedUIMAInterface.DUUIComposer;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIDockerDriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIRemoteDriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaContext;
import org.texttechnologylab.annotation.Claim;
import org.texttechnologylab.annotation.Fact;
import org.texttechnologylab.annotation.AnomlySpelling;
import org.texttechnologylab.annotation.FactChecking;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import static org.junit.Assert.assertTrue;


public class MultiTestFact {
    @Test
    public void multiTestEn() throws Exception {
        DUUIComposer composer = new DUUIComposer()
                .withSkipVerification(true)
                .withLuaContext(
                        new DUUILuaContext()
                                .withJsonLibrary()
                );

        DUUIRemoteDriver remote_driver = new DUUIRemoteDriver(10000);
        composer.addDriver(remote_driver);
        DUUIDockerDriver dockerDriver = new DUUIDockerDriver()
                .withTimeout(10000);
        composer.addDriver(dockerDriver);

        composer.add(
                new DUUIRemoteDriver.Component("http://127.0.0.1:9714")
        );

//        composer.add(new DUUIDockerDriver.Component("textimager_duui_spellcheck:0.1.3"));

        JCas cas = JCasFactory.createText("Justine Bateman is a producer. Justine Tanya Bateman (born February 19, 1966) is an American writer, producer, and actress . She is best known for her regular role as Mallory Keaton on the sitcom Family Ties (1982 -- 1989). Until recently, Bateman ran a production and consulting company, SECTION 5 . In the fall of 2012, she started studying computer science at UCLA. Jane is an engineer. Jane writes code for Huggingface.");
        cas.setDocumentLanguage("en");


        Claim c1 = new Claim(cas, 0, 30);
        Fact fact1 = new Fact(cas, 31, 368);
        Claim claim2 = new Claim(cas, 369, 389);
        Fact fact2 = new Fact(cas, 390, 423);
        AnomlySpelling anomlySpelling = new AnomlySpelling(cas);
        c1.setFacts(new FSArray(cas, 2));
        c1.setFacts(0, fact1);
        c1.setFacts(1, fact2);
        c1.addToIndexes();
        claim2.setFacts(new FSArray(cas, 2));
        claim2.setFacts(0, fact1);
        claim2.setFacts(1, fact2);
        claim2.addToIndexes();
        fact1.setClaims(new FSArray(cas, 2));
        fact1.setClaims(0, c1);
        fact1.setClaims(1, claim2);
        fact1.addToIndexes();
        fact2.setClaims(new FSArray(cas, 2));
        fact2.setClaims(0, c1);
        fact2.setClaims(1, claim2);
        fact2.addToIndexes();
        JCasUtil.selectCovered(Fact.class, c1);
        JCasUtil.selectAt(cas, Claim.class, 0, 30);

        System.out.println("Run Composer");

        composer.run(cas);

        composer.shutdown();
//        System.out.println(XmlFormatter.getPrettyString(cas));
        Collection<FactChecking> factChecks = JCasUtil.select(cas, FactChecking.class);
        for(FactChecking factCheck : factChecks){
            Claim claim = factCheck.getClaim();
            Fact fact = factCheck.getFact();
            if (fact.getBegin() == fact1.getBegin() & fact.getEnd() == fact1.getEnd() & claim.getBegin() == c1.getBegin() & claim.getEnd() == c1.getEnd()){
                double confidence = factCheck.getConsistency();
                Assertions.assertTrue(confidence > 0.5);
            }
            if (fact.getBegin() == fact2.getBegin() & fact.getEnd() == fact2.getEnd() & claim.getBegin() == claim2.getBegin() & claim.getEnd() == claim2.getEnd()){
                double confidence = factCheck.getConsistency();
                Assertions.assertTrue(confidence > 0.5);
            }
            if (fact.getBegin() == fact1.getBegin() & fact.getEnd() == fact1.getEnd() & claim.getBegin() == claim2.getBegin() & claim.getEnd() == claim2.getEnd()){
                double confidence = factCheck.getConsistency();
                Assertions.assertTrue(confidence < 0.5);
            }
            if (fact.getBegin() == fact2.getBegin() & fact.getEnd() == fact2.getEnd() & claim.getBegin() == c1.getBegin() & claim.getEnd() == c1.getEnd()){
                double confidence = factCheck.getConsistency();
                Assertions.assertTrue(confidence < 0.5);
            }
        }
    }

    @Test
    public void multiTestDe() throws Exception {
        DUUIComposer composer = new DUUIComposer()
                .withSkipVerification(true)
                .withLuaContext(
                        new DUUILuaContext()
                                .withJsonLibrary()
                );

        DUUIRemoteDriver remote_driver = new DUUIRemoteDriver(10000);
        composer.addDriver(remote_driver);
        DUUIDockerDriver dockerDriver = new DUUIDockerDriver()
                .withTimeout(10000);
        composer.addDriver(dockerDriver);

        composer.add(
                new DUUIRemoteDriver.Component("http://127.0.0.1:9714")
        );

//        composer.add(new DUUIDockerDriver.Component("textimager_duui_spellcheck:0.1.3"));

        JCas cas = JCasFactory.createText("Justine Bateman ist eine Produzentin. Justine Tanya Bateman (geboren am 19. Februar 1966) ist eine amerikanische Schriftstellerin, Produzentin und Schauspielerin. Sie ist am besten bekannt für ihre regelmäßige Rolle als Mallory Keaton in der Sitcom Family Ties (1982 -- 1989). Bis vor kurzem leitete Bateman ein Produktions- und Beratungsunternehmen, SECTION 5. Im Herbst 2012 begann sie ein Informatikstudium an der UCLA. Jane ist eine Ingenieurin. Jane schreibt Code für Huggingface.");
        cas.setDocumentLanguage("de");


        Claim c1 = new Claim(cas, 0, 37);
        Fact fact1 = new Fact(cas, 38, 422);
        Claim claim2 = new Claim(cas, 423, 448);
        Fact fact2 = new Fact(cas, 449, 483);
        AnomlySpelling anomlySpelling = new AnomlySpelling(cas);
        c1.setFacts(new FSArray(cas, 2));
        c1.setFacts(0, fact1);
        c1.setFacts(1, fact2);
        c1.addToIndexes();
        claim2.setFacts(new FSArray(cas, 2));
        claim2.setFacts(0, fact1);
        claim2.setFacts(1, fact2);
        claim2.addToIndexes();
        fact1.setClaims(new FSArray(cas, 2));
        fact1.setClaims(0, c1);
        fact1.setClaims(1, claim2);
        fact1.addToIndexes();
        fact2.setClaims(new FSArray(cas, 2));
        fact2.setClaims(0, c1);
        fact2.setClaims(1, claim2);
        fact2.addToIndexes();
        JCasUtil.selectCovered(Fact.class, c1);
        JCasUtil.selectAt(cas, Claim.class, 0, 30);

        System.out.println("Run Composer");

        composer.run(cas);

        composer.shutdown();
//        System.out.println(XmlFormatter.getPrettyString(cas));
        Collection<FactChecking> factChecks = JCasUtil.select(cas, FactChecking.class);
        for(FactChecking factCheck : factChecks){
            Claim claim = factCheck.getClaim();
            Fact fact = factCheck.getFact();
            if (fact.getBegin() == fact1.getBegin() & fact.getEnd() == fact1.getEnd() & claim.getBegin() == c1.getBegin() & claim.getEnd() == c1.getEnd()){
                double confidence = factCheck.getConsistency();
                Assertions.assertTrue(confidence > 0.5);
            }
            if (fact.getBegin() == fact2.getBegin() & fact.getEnd() == fact2.getEnd() & claim.getBegin() == claim2.getBegin() & claim.getEnd() == claim2.getEnd()){
                double confidence = factCheck.getConsistency();
                Assertions.assertTrue(confidence > 0.4);
            }
            if (fact.getBegin() == fact1.getBegin() & fact.getEnd() == fact1.getEnd() & claim.getBegin() == claim2.getBegin() & claim.getEnd() == claim2.getEnd()){
                double confidence = factCheck.getConsistency();
                Assertions.assertTrue(confidence < 0.5);
            }
            if (fact.getBegin() == fact2.getBegin() & fact.getEnd() == fact2.getEnd() & claim.getBegin() == c1.getBegin() & claim.getEnd() == c1.getEnd()){
                double confidence = factCheck.getConsistency();
                Assertions.assertTrue(confidence < 0.5);
            }
        }
    }
}
