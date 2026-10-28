package org.hucompute.textimager.uima.spacy;

import de.tudarmstadt.ukp.dkpro.core.api.lexmorph.type.pos.POS;
import de.tudarmstadt.ukp.dkpro.core.api.ner.type.NamedEntity;
import de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Paragraph;
import de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence;
import de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token;
import de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.Dependency;
import org.apache.uima.cas.CAS;
import org.apache.uima.fit.factory.JCasFactory;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;
import org.apache.uima.jcas.tcas.Annotation;
import org.junit.jupiter.api.Test;
import org.texttechnologylab.DockerUnifiedUIMAInterface.DUUIComposer;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIRemoteDriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaContext;
import org.texttechnologylab.DockerUnifiedUIMAInterface.segmentation.DUUISegmentationStrategyByAnnotation;
import org.texttechnologylab.annotation.SpacyAnnotatorMetaData;
import org.texttechnologylab.uima.type.spacy.SpacyNounChunk;
import org.texttechnologylab.uima.type.spacy.SpacyToken;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Pretokenized input ({@code use_existing_tokens}, {@code use_existing_sentences}): the given plain {@link Token}s and
 * {@link Sentence}s (here made by the test, in a pipeline by e.g. SoMaJo and wtpsplit) are kept, spaCy writes a new
 * {@link SpacyToken} per token with all features, and every other annotation starts and ends on token boundaries.
 *
 * <p>Needs a running service, URL from the system property {@code spacy.url} (default {@code http://localhost:9801}).
 */
public class SpaCyPretokenizedTest {

    private static final String SPACY_URL = System.getProperty("spacy.url", "http://localhost:9801");

    private static final String TOKEN_TYPE = Token.class.getName();

    // test fixture tokenization: words, numbers or single other characters
    private static final Pattern TOKEN_PATTERN = Pattern.compile("\\p{L}+|\\p{N}+|[^\\s\\p{L}\\p{N}]");

    // several spaces, line breaks, blank lines (paragraphs) and a character outside the BMP
    private static final String TEXT_DE = "Angela Merkel besuchte am Montag Berlin.  Dort traf sie   den Bürgermeister.\n"
            + "Die Stadt war voll.\n\n"
            + "Im zweiten Absatz steht das Zeichen 𐍁 im Text. Der Hund bellt laut!";
    private static final String TEXT_EN = "Barack Obama visited Paris on Monday.\nThe weather was nice.\n\n"
            + "The second paragraph has two sentences. It ends here.";

    private static DUUIComposer composer(boolean segmentByParagraph) throws Exception {
        DUUIComposer composer = new DUUIComposer()
                .withSkipVerification(true)
                .withLuaContext(new DUUILuaContext().withJsonLibrary());
        composer.addDriver(new DUUIRemoteDriver());
        DUUIRemoteDriver.Component component = new DUUIRemoteDriver.Component(SPACY_URL)
                .withParameter("model_variant", "large")
                .withParameter("use_existing_tokens", "true")
                .withParameter("use_existing_sentences", "true");
        if (segmentByParagraph) {
            // as in the Coh-Metrix paper pipeline: one paragraph per segment
            component.withSegmentationStrategy(new DUUISegmentationStrategyByAnnotation()
                    .withSegmentationClass(Paragraph.class)
                    .withView(CAS.NAME_DEFAULT_SOFA)
                    .withStrictSegmentationClass(true)
                    .withMaxCharsPerSegment(1000000000)
                    .withMaxAnnotationsPerSegment(1));
        }
        composer.add(component.build().withTimeout(1000L));
        return composer;
    }

    /** CAS with paragraphs (blank lines), sentences (after "." or "!") and tokens of the fixture tokenization. */
    private static JCas pretokenizedCas(String text, String language) throws Exception {
        JCas cas = JCasFactory.createJCas();
        cas.setDocumentText(text);
        cas.setDocumentLanguage(language);

        int paragraphBegin = 0;
        for (String paragraph : text.split("\n\n", -1)) {
            new Paragraph(cas, paragraphBegin, paragraphBegin + paragraph.length()).addToIndexes();
            paragraphBegin += paragraph.length() + 2;
        }

        Matcher m = TOKEN_PATTERN.matcher(text);
        int sentenceBegin = -1;
        while (m.find()) {
            new Token(cas, m.start(), m.end()).addToIndexes();
            if (sentenceBegin < 0) {
                sentenceBegin = m.start();
            }
            if (m.group().equals(".") || m.group().equals("!")) {
                new Sentence(cas, sentenceBegin, m.end()).addToIndexes();
                sentenceBegin = -1;
            }
        }
        return cas;
    }

    private static List<Token> plainTokens(JCas cas) {
        List<Token> tokens = new ArrayList<>();
        for (Token token : JCasUtil.select(cas, Token.class)) {
            if (token.getType().getName().equals(TOKEN_TYPE)) {
                tokens.add(token);
            }
        }
        return tokens;
    }

    private static void checkPretokenized(String text, String language, boolean segmentByParagraph) throws Exception {
        JCas cas = pretokenizedCas(text, language);
        List<int[]> givenTokens = new ArrayList<>();
        for (Token token : plainTokens(cas)) {
            givenTokens.add(new int[]{token.getBegin(), token.getEnd()});
        }
        List<int[]> givenSentences = new ArrayList<>();
        for (Sentence sentence : JCasUtil.select(cas, Sentence.class)) {
            givenSentences.add(new int[]{sentence.getBegin(), sentence.getEnd()});
        }

        DUUIComposer composer = composer(segmentByParagraph);
        try {
            composer.run(cas);
        } finally {
            composer.shutdown();
        }

        // given tokens and sentences unchanged, no sentences added
        List<Token> plain = plainTokens(cas);
        assertEquals(givenTokens.size(), plain.size());
        List<Sentence> sentences = new ArrayList<>(JCasUtil.select(cas, Sentence.class));
        assertEquals(givenSentences.size(), sentences.size(), "spaCy must not write sentences");
        for (int i = 0; i < sentences.size(); i++) {
            assertEquals(givenSentences.get(i)[0], sentences.get(i).getBegin());
            assertEquals(givenSentences.get(i)[1], sentences.get(i).getEnd());
        }

        // one SpacyToken per given token, same offsets, all features
        List<SpacyToken> spacyTokens = new ArrayList<>(JCasUtil.select(cas, SpacyToken.class));
        assertEquals(givenTokens.size(), spacyTokens.size());
        Set<Integer> tokenBegins = new HashSet<>();
        Set<Integer> tokenEnds = new HashSet<>();
        int withVector = 0;
        for (int i = 0; i < spacyTokens.size(); i++) {
            SpacyToken token = spacyTokens.get(i);
            assertEquals(givenTokens.get(i)[0], token.getBegin());
            assertEquals(givenTokens.get(i)[1], token.getEnd());
            tokenBegins.add(token.getBegin());
            tokenEnds.add(token.getEnd());
            String covered = token.getCoveredText();
            assertNotNull(token.getLemma(), covered);
            assertNotNull(token.getPos(), covered);
            assertNotNull(token.getPos().getPosValue(), covered);
            assertNotNull(token.getMorph(), covered);
            boolean letters = covered.codePoints().allMatch(Character::isLetter);
            assertEquals(letters, token.getIsAlpha(), "is_alpha of " + covered);
            boolean punct = covered.equals(".") || covered.equals("!");
            if (punct) {
                assertTrue(token.getIsPunct(), "is_punct of " + covered);
            }
            if (letters) {
                assertFalse(token.getIsPunct(), "is_punct of " + covered);
            }
            if (token.getHasVector()) {
                assertTrue(token.getVector().size() > 0);
                withVector++;
            }
            // lemma, POS and morphology annotations have the token's offsets
            assertEquals(token.getBegin(), token.getLemma().getBegin());
            assertEquals(token.getEnd(), token.getPos().getEnd());
        }
        assertTrue(withVector > spacyTokens.size() / 2, "vectors: " + withVector + " of " + spacyTokens.size());
        for (POS pos : JCasUtil.select(cas, POS.class)) {
            assertTrue(tokenBegins.contains(pos.getBegin()) && tokenEnds.contains(pos.getEnd()));
        }

        // dependencies: on token boundaries, between indexed SpacyTokens of the same sentence
        Map<SpacyToken, Boolean> indexed = new IdentityHashMap<>();
        for (SpacyToken token : spacyTokens) {
            indexed.put(token, true);
        }
        Map<SpacyToken, Sentence> sentenceOf = new IdentityHashMap<>();
        for (Sentence sentence : sentences) {
            for (SpacyToken token : JCasUtil.selectCovered(SpacyToken.class, sentence)) {
                sentenceOf.put(token, sentence);
            }
        }
        assertEquals(spacyTokens.size(), sentenceOf.size(), "every token inside a given sentence");
        List<Dependency> dependencies = new ArrayList<>(JCasUtil.select(cas, Dependency.class));
        assertEquals(spacyTokens.size(), dependencies.size(), "one dependency per token");
        for (Dependency dep : dependencies) {
            assertTrue(indexed.containsKey((SpacyToken) dep.getDependent()), "dependent is an indexed SpacyToken");
            assertTrue(indexed.containsKey((SpacyToken) dep.getGovernor()), "governor is an indexed SpacyToken");
            assertEquals(dep.getDependent().getBegin(), dep.getBegin());
            assertEquals(dep.getDependent().getEnd(), dep.getEnd());
            assertSame(sentenceOf.get((SpacyToken) dep.getDependent()), sentenceOf.get((SpacyToken) dep.getGovernor()),
                    "dependency crosses the given sentences: " + dep.getCoveredText());
        }

        // entities and noun chunks on token boundaries
        List<Annotation> spans = new ArrayList<>(JCasUtil.select(cas, NamedEntity.class));
        assertFalse(spans.isEmpty(), "named entities");
        int entities = spans.size();
        spans.addAll(JCasUtil.select(cas, SpacyNounChunk.class));
        assertTrue(spans.size() > entities, "noun chunks");
        for (Annotation span : spans) {
            assertTrue(tokenBegins.contains(span.getBegin()) && tokenEnds.contains(span.getEnd()),
                    span.getType().getShortName() + " not on token boundaries: \"" + span.getCoveredText() + "\"");
        }
        // every noun chunk has its meta data annotation
        long nounChunkMeta = JCasUtil.select(cas, SpacyAnnotatorMetaData.class).stream()
                .filter(meta -> meta.getReference() instanceof SpacyNounChunk).count();
        assertEquals(spans.size() - entities, nounChunkMeta, "meta data of the noun chunks");

        System.out.println(language + (segmentByParagraph ? " segmented" : "") + ": " + spacyTokens.size()
                + " tokens, " + sentences.size() + " sentences, " + entities + " entities, "
                + (spans.size() - entities) + " noun chunks, " + withVector + " with vector");
    }

    @Test
    public void pretokenizedDe() throws Exception {
        checkPretokenized(TEXT_DE, "de", false);
    }

    @Test
    public void pretokenizedEn() throws Exception {
        checkPretokenized(TEXT_EN, "en", false);
    }

    @Test
    public void pretokenizedDeSegmentedByParagraph() throws Exception {
        checkPretokenized(TEXT_DE, "de", true);
    }

    @Test
    public void pretokenizedEnSegmentedByParagraph() throws Exception {
        checkPretokenized(TEXT_EN, "en", true);
    }

    /**
     * On a text that spaCy tokenizes like the given tokens (single spaces), the pretokenized mode gives the same
     * values as the normal mode.
     */
    @Test
    public void pretokenizedEqualsNormalMode() throws Exception {
        String text = "Angela Merkel besuchte am Montag Berlin. Dort traf sie den Bürgermeister. Die Stadt war voll.";

        JCas normal = JCasFactory.createJCas();
        normal.setDocumentText(text);
        normal.setDocumentLanguage("de");
        DUUIComposer composer = new DUUIComposer()
                .withSkipVerification(true)
                .withLuaContext(new DUUILuaContext().withJsonLibrary());
        composer.addDriver(new DUUIRemoteDriver());
        composer.add(new DUUIRemoteDriver.Component(SPACY_URL).withParameter("model_variant", "large")
                .build().withTimeout(1000L));
        try {
            composer.run(normal);
        } finally {
            composer.shutdown();
        }

        // given tokens and sentences = those of the normal mode
        JCas pretokenized = JCasFactory.createJCas();
        pretokenized.setDocumentText(text);
        pretokenized.setDocumentLanguage("de");
        for (SpacyToken token : JCasUtil.select(normal, SpacyToken.class)) {
            new Token(pretokenized, token.getBegin(), token.getEnd()).addToIndexes();
        }
        for (Sentence sentence : JCasUtil.select(normal, Sentence.class)) {
            new Sentence(pretokenized, sentence.getBegin(), sentence.getEnd()).addToIndexes();
        }
        composer = composer(false);
        try {
            composer.run(pretokenized);
        } finally {
            composer.shutdown();
        }

        List<SpacyToken> expected = new ArrayList<>(JCasUtil.select(normal, SpacyToken.class));
        List<SpacyToken> actual = new ArrayList<>(JCasUtil.select(pretokenized, SpacyToken.class));
        assertEquals(expected.size(), actual.size());
        for (int i = 0; i < expected.size(); i++) {
            SpacyToken e = expected.get(i);
            SpacyToken a = actual.get(i);
            String what = "token " + i + " \"" + e.getCoveredText() + "\"";
            assertEquals(e.getBegin(), a.getBegin(), what);
            assertEquals(e.getEnd(), a.getEnd(), what);
            assertEquals(e.getLemma().getValue(), a.getLemma().getValue(), what);
            assertEquals(e.getPos().getPosValue(), a.getPos().getPosValue(), what);
            assertEquals(e.getPos().getCoarseValue(), a.getPos().getCoarseValue(), what);
            assertEquals(e.getMorph().getValue(), a.getMorph().getValue(), what);
            assertEquals(e.getIsAlpha(), a.getIsAlpha(), what);
            assertEquals(e.getIsPunct(), a.getIsPunct(), what);
            assertEquals(e.getIsStop(), a.getIsStop(), what);
            assertEquals(e.getLikeNum(), a.getLikeNum(), what);
            assertEquals(e.getHasVector(), a.getHasVector(), what);
            if (e.getHasVector()) {
                assertEquals(e.getVector().size(), a.getVector().size(), what);
                for (int v = 0; v < e.getVector().size(); v++) {
                    assertEquals(e.getVector().get(v), a.getVector().get(v), what);
                }
            }
            assertEquals(e.getParent().getBegin(), a.getParent().getBegin(), what + " head");
        }
        assertEquals(spans(normal, NamedEntity.class), spans(pretokenized, NamedEntity.class));
        assertEquals(spans(normal, SpacyNounChunk.class), spans(pretokenized, SpacyNounChunk.class));
    }

    private static <T extends Annotation> List<String> spans(JCas cas, Class<T> type) {
        List<String> spans = new ArrayList<>();
        for (T span : JCasUtil.select(cas, type)) {
            spans.add(span.getBegin() + "-" + span.getEnd() + (span instanceof NamedEntity ne ? " " + ne.getValue() : ""));
        }
        return spans;
    }
}
