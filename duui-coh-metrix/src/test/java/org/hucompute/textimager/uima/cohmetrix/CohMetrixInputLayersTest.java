package org.hucompute.textimager.uima.cohmetrix;

import de.tudarmstadt.ukp.dkpro.core.api.lexmorph.type.pos.POS;
import de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Paragraph;
import de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence;
import de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token;
import de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.TokenForm;
import de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.Dependency;
import org.apache.uima.fit.factory.JCasFactory;
import org.apache.uima.jcas.JCas;
import org.apache.uima.jcas.cas.FloatArray;
import org.json.JSONArray;
import org.json.JSONObject;
import org.junit.jupiter.api.Test;
import org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaCommunicationLayer;
import org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaContext;
import org.texttechnologylab.uima.type.WordEmbedding;
import org.texttechnologylab.uima.type.spacy.SpacyToken;
import org.texttechnologylab.uima.type.stanza.StanzaSentence;
import org.texttechnologylab.uima.type.stanza.StanzaToken;

import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;

/**
 * The input layers communication.lua reads (README, "Prerequisites and preprocessing"), without a running service: the
 * request JSON for CASes with several token and sentence layers over the same text.
 * <ul>
 *   <li>Token layer: StanzaToken, else SpacyToken, else plain Token; exactly one layer (no word counted twice).</li>
 *   <li>Sentence layer: StanzaSentence if present (duui-stanza splits sentences too long for the parser).</li>
 *   <li>Vectors: the WordEmbeddings reference SpacyTokens (duui-spacy, {@code token_type=StanzaToken}); a StanzaToken takes
 *       the next unused one of its span, so the two words of a split contraction get their own vectors.</li>
 * </ul>
 */
public class CohMetrixInputLayersTest {

    // two sentences for Stanza, one plain Sentence (as if the sentence splitter missed the boundary)
    private static final String TEXT = "Er geht zum Haus . Es ist groß .";
    // the Stanza words: "zum" = "zu" + "dem", both over the span of "zum"
    private static final String[] WORDS = {"Er", "geht", "zu", "dem", "Haus", ".", "Es", "ist", "groß", "."};
    private static final int[][] SPANS = {{0, 2}, {3, 7}, {8, 11}, {8, 11}, {12, 16}, {17, 18}, {19, 21}, {22, 25},
            {26, 30}, {31, 32}};
    private static final String[] UD = {"nsubj", "root", "case", "det", "obl", "punct", "nsubj", "cop", "root", "punct"};
    private static final int[] HEADS = {1, 1, 4, 4, 1, 1, 8, 8, 8, 8};   // index of the head word (root: itself)

    private static JCas cas(boolean stanza, boolean spacy) throws Exception {
        JCas cas = JCasFactory.createJCas();
        cas.setDocumentText(TEXT);
        cas.setDocumentLanguage("de");
        new Paragraph(cas, 0, TEXT.length()).addToIndexes();
        new Sentence(cas, 0, TEXT.length()).addToIndexes();
        // plain Tokens as a tokenizer writes them: "zum" is one token
        for (int[] span : new int[][]{{0, 2}, {3, 7}, {8, 11}, {12, 16}, {17, 18}, {19, 21}, {22, 25}, {26, 30}, {31, 32}}) {
            new Token(cas, span[0], span[1]).addToIndexes();
        }
        List<Token> words = new ArrayList<>();
        if (stanza) {
            new StanzaSentence(cas, 0, 18).addToIndexes();
            new StanzaSentence(cas, 19, 32).addToIndexes();
            for (int i = 0; i < WORDS.length; i++) {
                StanzaToken token = new StanzaToken(cas, SPANS[i][0], SPANS[i][1]);
                if (!WORDS[i].equals(token.getCoveredText())) {
                    TokenForm form = new TokenForm(cas, SPANS[i][0], SPANS[i][1]);
                    form.setValue(WORDS[i]);
                    form.addToIndexes();
                    token.setForm(form);
                }
                POS pos = new POS(cas, SPANS[i][0], SPANS[i][1]);
                pos.setPosValue("X" + i);
                pos.addToIndexes();
                token.setPos(pos);
                token.addToIndexes();
                words.add(token);
            }
            for (int i = 0; i < WORDS.length; i++) {
                Dependency dependency = new Dependency(cas, SPANS[i][0], SPANS[i][1]);
                dependency.setDependent(words.get(i));
                dependency.setGovernor(words.get(HEADS[i]));
                dependency.setDependencyType(UD[i]);
                dependency.addToIndexes();
            }
        }
        if (spacy) {
            // a SpacyToken and a WordEmbedding per Stanza word, in the order of the words; vector of word i: (i, -i)
            for (int i = 0; i < WORDS.length; i++) {
                SpacyToken token = new SpacyToken(cas, SPANS[i][0], SPANS[i][1]);
                token.addToIndexes();
                WordEmbedding embedding = new WordEmbedding(cas, SPANS[i][0], SPANS[i][1]);
                embedding.setReference(token);
                embedding.setShortName("SPACYlg");
                FloatArray vector = new FloatArray(cas, 2);
                vector.set(0, i);
                vector.set(1, -i);
                embedding.setEmbedding(vector);
                embedding.addToIndexes();
            }
        }
        return cas;
    }

    private static JSONObject request(JCas cas) throws Exception {
        String script = Files.readString(Path.of("src/main/lua/communication.lua"));
        DUUILuaCommunicationLayer lua = new DUUILuaCommunicationLayer(script, "test",
                new DUUILuaContext().withJsonLibrary());
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        lua.serialize(cas, out, Map.of(), "_InitialView");
        return new JSONObject(out.toString(StandardCharsets.UTF_8));
    }

    private static List<JSONObject> sentences(JSONObject request) {
        List<JSONObject> out = new ArrayList<>();
        JSONArray paragraphs = request.getJSONArray("paragraphs");
        for (int p = 0; p < paragraphs.length(); p++) {
            JSONArray sentences = paragraphs.getJSONObject(p).getJSONArray("sentences");
            for (int s = 0; s < sentences.length(); s++) {
                out.add(sentences.getJSONObject(s));
            }
        }
        return out;
    }

    private static List<JSONObject> tokens(JSONObject request) {
        List<JSONObject> out = new ArrayList<>();
        for (JSONObject sentence : sentences(request)) {
            JSONArray tokens = sentence.getJSONArray("tokens");
            for (int t = 0; t < tokens.length(); t++) {
                out.add(tokens.getJSONObject(t));
            }
        }
        return out;
    }

    @Test
    public void stanzaLayerWithVectorsPerWord() throws Exception {
        JSONObject request = request(cas(true, true));

        // the two StanzaSentences, not the one plain Sentence
        List<JSONObject> sentences = sentences(request);
        assertEquals(2, sentences.size());

        // exactly the Stanza words: not the plain Tokens, not the SpacyTokens
        List<JSONObject> tokens = tokens(request);
        assertEquals(WORDS.length, tokens.size());
        for (int i = 0; i < WORDS.length; i++) {
            JSONObject token = tokens.get(i);
            assertEquals(WORDS[i], token.getString("text"));
            assertEquals("X" + i, token.getString("pos_value"), "the Stanza POS");
            assertEquals(UD[i], token.getString("dep_type"));
            // the vector of the i-th SpacyToken: "zu" and "dem" (same span) get their own
            JSONArray vector = token.getJSONArray("vector");
            assertEquals(i, vector.getDouble(0), 0.0, WORDS[i]);
            assertEquals(-i, vector.getDouble(1), 0.0, WORDS[i]);
        }
        // the head of "dem" is "Haus", the head of "zu" too (resolved by identity, not by the shared span)
        assertEquals(4, tokens.get(2).getInt("head_index"));
        assertEquals(4, tokens.get(3).getInt("head_index"));
        // the second sentence: sentence-local heads
        assertEquals(2, sentences.get(1).getJSONArray("tokens").getJSONObject(0).getInt("head_index"));
        assertEquals("WordEmbedding", request.getString("vectors_source"));
        assertEquals("SPACYlg", request.getString("vectors"));
    }

    @Test
    public void spacyLayerWithoutStanza() throws Exception {
        JSONObject request = request(cas(false, true));
        // the SpacyTokens (one per word of the CAS above, "zum" twice), not the plain Tokens; the plain Sentence
        assertEquals(1, sentences(request).size());
        List<JSONObject> tokens = tokens(request);
        assertEquals(WORDS.length, tokens.size());
        for (int i = 0; i < WORDS.length; i++) {
            assertEquals(i, tokens.get(i).getJSONArray("vector").getDouble(0), 0.0, "reference of SpacyToken " + i);
        }
    }

    @Test
    public void plainTokensOnly() throws Exception {
        JSONObject request = request(cas(false, false));
        List<JSONObject> tokens = tokens(request);
        assertEquals(9, tokens.size());
        assertEquals("zum", tokens.get(2).getString("text"));
        assertFalse(tokens.get(2).getBoolean("has_vector"));
    }
}
