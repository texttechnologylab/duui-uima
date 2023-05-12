package org.hucompute.textimager.uima.ddc.fasttext.service.service;

import de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.*;
import org.apache.uima.analysis_engine.AnalysisEngineProcessException;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;
import org.apache.uima.jcas.tcas.Annotation;
import org.apache.uima.resource.ResourceInitializationException;
import org.hucompute.textimager.uima.type.category.CategoryCoveredTagged;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.*;

public class FastTextService {
    // Für jede Sprache eine Liste an Prozessen
    private HashMap<String, ArrayList<FastTextProcess>> fasttext_procs;
    boolean lazy_load;
    int max_loaded;
    Queue<FastTextProcess> currently_loaded;

    boolean useLemma = true;
    boolean addPOS = true;
    String posmapLocation = "";
    boolean removePunct = true;
    boolean removeFunctionwords = true;
    boolean ignoreMissingLemmaPOS = true;
    boolean cutoff = false;
    int fasttextK = 1000;
    String tags = "";

    // POS Mapping
    protected static HashMap<String, String> posMapping = new HashMap<>();

    // FastText im "Stdin" Modus starten
    public FastTextService(
            String fasttextLocation,
            String fastTextLanguageModelsLabels,
            boolean lazyLoad,
            int maxLoaded,
            boolean useLemma,
            boolean addPOS,
            String posmapLocation,
            boolean removePunct,
            boolean removeFunctionwords,
            boolean ignoreMissingLemmaPOS,
            boolean cutoff,
            int fasttextK,
            String tags
    ) throws Exception {
        System.out.println("Initializing fastText processes...");

        // TODO parameters add later,
        // can not be added on process, but init
        this.useLemma = useLemma;
        this.addPOS = addPOS;
        this.posmapLocation = posmapLocation;
        this.removePunct = removePunct;
        this.removeFunctionwords = removeFunctionwords;
        this.ignoreMissingLemmaPOS = ignoreMissingLemmaPOS;
        this.cutoff = cutoff;
        this.fasttextK = fasttextK;
        this.tags = tags;

        System.out.println("- useLemma: " + useLemma);
        System.out.println("- addPOS: " + addPOS);
        System.out.println("- removePunct: " + removePunct);
        System.out.println("- removeFunctionwords: " + removeFunctionwords);
        System.out.println("- ignoreMissingLemmaPOS: " + ignoreMissingLemmaPOS);

        // TODO weitere Vorbedingungen prüfen!

        if (removeFunctionwords && posmapLocation.isEmpty()) {
            throw new ResourceInitializationException("removeFunctionwords = true but no posmap location specified", null);
        }

        if (!posmapLocation.isEmpty()) {
            try {
                readPOSMappingFile(posmapLocation);
            } catch (IOException e) {
                throw new ResourceInitializationException("error loading posmapping file", null, e);
            }
        }


        System.out.println("initializing done.");

        // TODO Später auslagern in DUCC, der kann Annotatoren einfach starten und eine Zeit laufen lassen
        lazy_load = lazyLoad;
        max_loaded = maxLoaded;
        currently_loaded = new LinkedList<>();

        fasttext_procs = new HashMap<>();

        try {
            String[] ftlml = fastTextLanguageModelsLabels.split(",", -1);
            for (int ind = 0; ind < ftlml.length; ind += 3) {
                String lang = ftlml[ind];
                String model = ftlml[ind + 1];
                int numLables = Integer.parseInt(ftlml[ind + 2]);
                if (!fasttext_procs.containsKey(lang)) {
                    fasttext_procs.put(lang, new ArrayList<>());
                }
                fasttext_procs.get(lang).add(new FastTextProcess(lang, model, fasttextLocation, numLables));
            }
        } catch (Exception ex) {
            throw new Exception("Error initializing fastText Process Handler", ex);
        }


        if (!lazyLoad) {
            System.out.println("Starting fastText processes...");
            for (HashMap.Entry<String, ArrayList<FastTextProcess>> ftps : fasttext_procs.entrySet()) {
                for (FastTextProcess ftp : ftps.getValue()) {
                    ftp.start();
                }
            }
        } else {
            System.out.println("Not starting fastText processes, lazy loading enabled wit max=" + maxLoaded);
        }
    }

    public void exit() {
        System.out.println("Exiting fastText processes...");

        for (HashMap.Entry<String, ArrayList<FastTextProcess>> ftps : fasttext_procs.entrySet()) {
            for (FastTextProcess ftp : ftps.getValue()) {
                ftp.exit();
            }
        }

        fasttext_procs.clear();
    }

    private String cleanStringForFastText(String input) {
        // Alles auf eine Zeile reduzieren...
        // Sowie fastText "Satzende" usw Zeichen entfernen
        return input
                .replaceAll("</s>|<|>", " ")
                .replaceAll("\\r\\n|\\r|\\n", " ");
    }

    public ArrayList<FastTextResult> input(String language, String inputText) throws Exception {
        //System.out.println("fastText input");
        //System.out.println("!!!" + inputText + "!!!");

        // Input Text modifizieren:
        // Wenn der Text weniger als 1000 Zeichen hat alles solange wiederholen bis drüber
        // TODO Als Configparam
        // TODO auch bei Disambiguierung?
        StringBuilder temp = new StringBuilder();
        temp.append(inputText).append(" ");
        int inputTextLength = temp.length();
        while (inputTextLength < 1000) {
            temp.append(inputText).append(" ");
            inputTextLength = temp.length();
        }
        inputText = temp.toString();

        // Sprache  wählen
        ArrayList<FastTextProcess> ftps;
        if (fasttext_procs.containsKey(language)) {
            ftps = fasttext_procs.get(language);
            //System.out.println("  language ok: " + language);
        } else {
            // Keine Sprache, einfach erstes wählen...
            try {
                ftps = fasttext_procs.entrySet().iterator().next().getValue();
                //System.out.println("  language not ok, took first available: " + language);
            } catch (Exception ex) {
                throw new Exception("fastText could not find language [" + language + "]", ex);
            }
        }

        String text = cleanStringForFastText(inputText);

        ArrayList<FastTextResult> results = new ArrayList<>();

        for (FastTextProcess ftp : ftps) {
            if (lazy_load && !ftp.isLoaded()) {
                try {
                    if (currently_loaded.size() >= max_loaded) {
                        currently_loaded.poll().exit();
                    }
                    ftp.start();
                    currently_loaded.add(ftp);
                } catch (Exception e) {
                    throw new Exception("fastText could not lazy load language [" + language + "], model [" + ftp.model + "]", e);
                }
            }

            FastTextResult niceResult = new FastTextResult();

            //System.out.println("processing with model [" + ftp.model + "]...");
            try {
                if (ftp.stdin(text)) {
                    ArrayList<String> result = new ArrayList<>();

                    // Immer genau 2 Ausgaben pro Label
                    for (int i = 0; i < 2 * ftp.getNumLabels(); ++i) {
                        result.add(ftp.next());
                    }

                    niceResult.parseResults(result);
                }
            } catch (Exception ex) {
                throw new Exception("fastText could not get output", ex);
            }
            //System.out.println("processing with model [" + ftp.model + "] done.");

            results.add(niceResult);
        }

        return results;
    }

    protected void readPOSMappingFile(String posmapLocation) throws IOException, ResourceInitializationException {
        System.out.println("loading pos map file: " + posmapLocation);

        posMapping = new HashMap<>();

        BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(posmapLocation), StandardCharsets.UTF_8));
        String line;
        int lineCounter = 0;
        while ((line = reader.readLine()) != null) {
            lineCounter++;
            String[] lineData = line.split("\t", 2);

            if (lineData.length != 2) {
                throw new ResourceInitializationException("error loading posmapping file: line " + lineCounter + " is invalid.", null);
            } else {
                posMapping.put(lineData[0], lineData[1]);
            }
        }
        reader.close();
    }

    public void process(JCas jCas, String selection) throws AnalysisEngineProcessException {
        // TODO let user give Class Names instead of "special" names and use class automatically
        //try {
        //    Class.forName(Paragraph.class.toString());
        //} catch (ClassNotFoundException e) {
        //    e.printStackTrace();
        //}

        String[] selections = selection.split(",", -1);
        for (String sel : selections) {
            if (sel.equals("paragraph")) {
                for (Paragraph paragraph : JCasUtil.select(jCas, Paragraph.class)) {
                    processCoveredWithFastText(jCas, paragraph);
                }
            } else if (sel.equals("sentence")) {
                for (Sentence sentence : JCasUtil.select(jCas, Sentence.class)) {
                    processCoveredWithFastText(jCas, sentence);
                }
            } else if (sel.equals("token")) {
                for (Token token : JCasUtil.select(jCas, Token.class)) {
                    processCoveredWithFastText(jCas, token);
                }
            } else if (sel.startsWith("line")) {
                String divType = null;
                if (sel.contains(";")) {
                    divType = sel.substring(5);
                }
                for (Div div : JCasUtil.select(jCas, Div.class)) {
                    if (divType == null || (div.getDivType() != null && div.getDivType().equals(divType))) {
                        processCoveredWithFastText(jCas, div);
                    }
                }
            } else /*if (sel.equals("text")) */ {
                // text (= alles als Standard)
                processCoveredWithFastText(jCas, null);
            }
        }
    }

    // Funktion die abgeleitete Klassen implementieren müssen
    protected void processCoveredWithFastText(JCas jCas, Annotation ref) throws AnalysisEngineProcessException {
        // should be abstract...
    }

    // Holt den Text aus CAS oder ref mit Option Lemma, POS, Filter...
    protected static String getText(JCas jCas, Annotation ref, boolean useLemma, boolean addPOS, boolean removePunct, boolean removeFunctionwords, boolean ignoreMissingLemmaPOS) {
        StringBuilder sb = new StringBuilder();

        Collection<Token> tokens;
        if (ref != null) {
            tokens = JCasUtil.selectCovered(Token.class, ref);
        } else {
            tokens = JCasUtil.select(jCas, Token.class);
        }

        for (Token token : tokens) {

            String pos = mapPOS(token);

            if (removeFunctionwords) {
                if (isPOSFunctionWord(pos)) {
                    continue;
                }
            }

            String text = getTextFromTokenOrLemma(token, useLemma, ignoreMissingLemmaPOS);

            if (removePunct) {
                if (isPunctuation(text)) {
                    continue;
                }
            }

            sb.append(text);

            if (addPOS) {
                sb.append("_").append(pos);
            }

            sb.append(" ");
        }

        return sb.toString();
    }

    protected static String getTextWithDisambig(JCas jCas, Annotation ref, boolean useLemma, boolean addPOS, boolean removePunct, boolean removeFunctionwords, String disambigTag, String disambigLabelReplace, String disambigLabelReplaceWith, boolean ignoreMissingLemmaPOS) {
        StringBuilder sb = new StringBuilder();

        Collection<Token> tokens;
        if (ref != null) {
            tokens = JCasUtil.selectCovered(Token.class, ref);
        } else {
            tokens = JCasUtil.select(jCas, Token.class);
        }

        for (Token token : tokens) {

            String pos = mapPOS(token);

            if (removeFunctionwords) {
                if (isPOSFunctionWord(pos)) {
                    continue;
                }
            }

            String text = getTextFromTokenOrLemma(token, useLemma, ignoreMissingLemmaPOS);

            if (removePunct) {
                if (isPunctuation(text)) {
                    continue;
                }
            }

            sb.append(text);

            if (addPOS) {
                sb.append("_").append(pos);
            }

            sb.append(" ");

            if (!disambigTag.isEmpty()) {
                ArrayList<CategoryCoveredTagged> disambigs = new ArrayList<>();
                disambigs.addAll(JCasUtil.selectCovered(CategoryCoveredTagged.class, token));
                if (!disambigs.isEmpty()) {
                    // Alle entfernen mit falschem Tag
                    disambigs.removeIf(dis -> {
                        String[] tags = dis.getTags().split(",", -1);
                        for (int ind = 0; ind < tags.length; ++ind) {
                            if (tags[ind].equals(disambigTag)) {
                                return false;
                            }
                        }
                        return true;
                    });

                    // Sortieren
                    Collections.sort(disambigs, (r1, r2) -> ((r1.getScore() > r2.getScore()) ? -1 : ((r1.getScore() < r2.getScore()) ? 1 : 0)));

                    // Label anpassen für weitere Schritte
                    String bestDisambig = disambigs.get(0).getValue();
                    if (!disambigLabelReplace.isEmpty() || !disambigLabelReplaceWith.isEmpty()) {
                        bestDisambig = bestDisambig.replaceFirst(disambigLabelReplace, disambigLabelReplaceWith);
                    }

                    // Bestes hinzufügen
                    sb.append(bestDisambig).append(" ");
                }
            }
        }

        return sb.toString();
    }

    private static String getTextFromTokenOrLemma(Token token, boolean useLemma, boolean ignoreMissingLemmaPOS) {
        String text;
        if (useLemma) {
            Lemma lemma = token.getLemma();
            // Wenn kein Lemma existiert und wir diesen Fehler ignorieren wollen...
            if (lemma == null && ignoreMissingLemmaPOS) {
                text = "";
            } else {
                text = lemma.getValue();
            }
        } else {
            text = token.getCoveredText();
        }
        return text;
    }

    protected static String mapPOS(Token token) {
        if (token.getPos() == null) {
            return "";
        }

        String posStrOrig = token.getPos().getPosValue();

        if (posMapping.containsKey(posStrOrig)) {
            return posMapping.get(posStrOrig);
        }

        return posStrOrig;
    }

    protected static boolean isPOSFunctionWord(String posStr) {
        return posStr.equals("ART")
                || posStr.startsWith("K")
                || posStr.startsWith("PTK")
                || posStr.startsWith("AP")
                || posStr.startsWith("VM")
                || posStr.startsWith("VA");
    }

    protected static boolean isPunctuation(String input) {
        String temp = input.replaceAll("\\p{P}", " ").replaceAll("\\|", " ").trim();
        return temp.isEmpty();
    }
}
