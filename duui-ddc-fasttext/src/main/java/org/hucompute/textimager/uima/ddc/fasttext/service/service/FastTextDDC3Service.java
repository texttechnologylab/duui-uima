package org.hucompute.textimager.uima.ddc.fasttext.service.service;

import org.apache.uima.analysis_engine.AnalysisEngineProcessException;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;
import org.apache.uima.jcas.tcas.Annotation;
import org.hucompute.textimager.uima.type.category.CategoryCoveredTagged;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;

public class FastTextDDC3Service extends FastTextService {
    String disambigTag = "disambigTag";
    String disambigLabelReplace = "__label__";
    String disambigLabelReplaceWith = "__disambig_word__";

    boolean removeOldScores = true;

    String tagsDDC2 = "ddc2";
    String tagsDDC3 = "ddc3";

    // DDC2 / DDC3 Ergebnisse speichern
    private class CategoryCoveredTaggedSimple {
        public int begin;
        public int end;
        public String value;
        public double score;
        public Annotation ref;
    }

    // Speichern der Labels der 2. / 3. Ebene
    protected HashMap<String, ArrayList<CategoryCoveredTaggedSimple>> categoriesLevel2 = new HashMap<>();
    protected HashMap<String, ArrayList<CategoryCoveredTaggedSimple>> categoriesLevel3 = new HashMap<>();

    FastTextDDC3Service(String fasttextLocation,
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
                        String tagsDDC2,
                        String tagsDDC3,
                        String disambigTag,
                        String disambigLabelReplace,
                        String disambigLabelReplaceWith
    ) throws Exception {
        super(
            fasttextLocation,
            fastTextLanguageModelsLabels,
            lazyLoad,
            maxLoaded,
            useLemma,
            addPOS,
            posmapLocation,
            removePunct,
            removeFunctionwords,
            ignoreMissingLemmaPOS,
            cutoff,
            fasttextK,
            tagsDDC3
        );

        this.disambigTag = disambigTag;
        this.disambigLabelReplace = disambigLabelReplace;
        this.disambigLabelReplaceWith = disambigLabelReplaceWith;

        this.tagsDDC2 = tagsDDC2;
        this.tagsDDC3 = tagsDDC3;

        categoriesLevel2 = new HashMap<>();
        categoriesLevel3 = new HashMap<>();

        // NOTE: we expect ddc2 to already be processed
    }

    // Prüft ob 1/2 Stelle von DDC2 und DDC3 Label übereinstimmen
    private boolean levelsMatch(CategoryCoveredTaggedSimple catLevel2, CategoryCoveredTaggedSimple catLevel3) {

        int l3 = catLevel3.value.length();
        int l2 = catLevel2.value.length();

        char l3_1 = catLevel3.value.charAt(l3 - 2);
        char l3_2 = catLevel3.value.charAt(l3 - 3);
        char l2_1 = catLevel2.value.charAt(l2 - 2);
        char l2_2 = catLevel2.value.charAt(l2 - 3);

        return (l3_1 == l2_1 && l3_2 == l2_2);

    }

    @Override
    protected void processCoveredWithFastText(JCas jCas, Annotation ref) throws AnalysisEngineProcessException {

        String documentText = getTextWithDisambig(jCas, ref, useLemma, addPOS, removePunct, removeFunctionwords, disambigTag, disambigLabelReplace, disambigLabelReplaceWith, ignoreMissingLemmaPOS);
        //System.out.println(documentText);

        // Begin und End setzen, entweder passend zu Ref oder kompletter Text
        int begin = (ref != null ? ref.getBegin() : 0);
        int end = (ref != null ? ref.getEnd() : jCas.getDocumentText().length());

        try {
            // FastText für DDC2 und DDC3 aufrufen
            addDDC(jCas, ref, begin, end, documentText, categoriesLevel3, tagsDDC3);
            // NOTE we expect ddc2 to already be processed, only adding to internal datastructures
            //addDDC(jCas, ref, begin, end, documentText, fasttext, categoriesLevel2, tagsDDC2);
            addDDC2(jCas, ref, categoriesLevel2);
        } catch (Exception ex) {
            throw new AnalysisEngineProcessException("error processing: " + ex.getMessage(), null, ex);
        }

        ArrayList<CategoryCoveredTaggedSimple> finalList = new ArrayList<>();

        // Alle mit passedem begin/end durchgehen und multiplizieren
        for (HashMap.Entry<String, ArrayList<CategoryCoveredTaggedSimple>> level2 : categoriesLevel2.entrySet()) {
            for (CategoryCoveredTaggedSimple catLevel2 : level2.getValue()) {
                // Alle zu diesem Begin/End passenden Level3 holen
                ArrayList<CategoryCoveredTaggedSimple> level3List = categoriesLevel3.get(beginEndKey(catLevel2.begin, catLevel2.end));
                for (CategoryCoveredTaggedSimple catLevel3 : level3List) {
                    // Passen die Label?
                    // Wenn in Ebene 2 DDC520 rauskommt, dann alle aus Ebene 2 mit DDC52_ multiplizieren
                    if (levelsMatch(catLevel2, catLevel3)) {

                        // Label von DDC3 nehmen (ohne "_old")
                        String label = catLevel3.value;

                        // Neuer Score ist Produkt der beiden Ebenen
                        double score = catLevel2.score * catLevel3.score;

                        // Ref von Category3
                        Annotation newRef = catLevel3.ref;

                        CategoryCoveredTaggedSimple c = new CategoryCoveredTaggedSimple();
                        c.begin = catLevel3.begin;
                        c.end = catLevel3.end;
                        c.value = label;
                        c.score = score;
                        c.ref = newRef;
                        finalList.add(c);
                    }
                }
            }
        }

        //System.out.println("ddcMul matching done");
        // Sortieren
        Collections.sort(finalList, (r1, r2) -> ((r1.score > r2.score) ? -1 : ((r1.score < r2.score) ? 1 : 0)));
        if (cutoff) {
            CategoryCoveredTaggedSimple lowest = finalList.get(finalList.size() - 1);
            finalList.removeIf(v -> v.score <= lowest.score);
            // Letztes wieder hinzufügen um niedrigsten Wert zu behalten
            finalList.add(lowest);
        }

        //System.out.println("ddcMul sorting done");
        int num = 0;
        for (CategoryCoveredTaggedSimple c : finalList) {
            if (num >= fasttextK) {
                break;
            }
            num++;

            CategoryCoveredTagged cat = new CategoryCoveredTagged(jCas, c.begin, c.end);
            cat.setValue(c.value);
            cat.setScore(c.score);
            cat.setTags(tags);
            cat.setRef(c.ref);
            cat.addToIndexes();
        }

        categoriesLevel2 = new HashMap<>();
        categoriesLevel3 = new HashMap<>();

    }

    private String beginEndKey(int begin, int end) {
        return String.valueOf(begin + "_" + end);
    }

    private void addDDC(JCas jCas, Annotation ref, int begin, int end, String documentText, HashMap<String, ArrayList<CategoryCoveredTaggedSimple>> list, String tagsDDC) throws Exception {

        if (documentText.isEmpty()) {
            return;
        }

        // Only supports 1 model!
        FastTextResult ftResult = input(jCas.getDocumentLanguage(), documentText).get(0);
        ArrayList<ProbabilityLabel> labels = ftResult.getSortedResults(false);

        int num = 0;
        for (ProbabilityLabel result : labels) {
            num++;

            CategoryCoveredTaggedSimple c = new CategoryCoveredTaggedSimple();
            c.begin = begin;
            c.end = end;
            c.value = result.getLabel();
            c.score = result.getLogProb();
            c.ref = ref;

            String key = beginEndKey(begin, end);
            if (!list.containsKey(key)) {
                list.put(key, new ArrayList<>());
            }
            list.get(key).add(c);

            // Ergebnisse dem CAS hinzufügen
            if (!removeOldScores && num <= fasttextK) {
                CategoryCoveredTagged cat = new CategoryCoveredTagged(jCas, begin, end);
                cat.setValue(result.getLabel());
                cat.setScore(result.getLogProb());
                cat.setTags(tagsDDC);
                cat.setRef(ref);
                cat.addToIndexes(jCas);
            }
        }

    }

    private void addDDC2(JCas jCas, Annotation ref, HashMap<String, ArrayList<CategoryCoveredTaggedSimple>> list) throws Exception {

        // Create internal data from existing DDC2 annotations
        Collection<CategoryCoveredTagged> categories;
        if (ref != null) {
            categories = JCasUtil.selectCovered(CategoryCoveredTagged.class, ref);
        }
        else {
            categories = JCasUtil.select(jCas, CategoryCoveredTagged.class);
        }
        for (CategoryCoveredTagged catgory : categories) {
            // Only consider categories with the correct tag
            if (catgory.getTags() == null || !catgory.getTags().equals(tagsDDC2)) {
                continue;
            }

            CategoryCoveredTaggedSimple c = new CategoryCoveredTaggedSimple();
            c.begin = catgory.getBegin();
            c.end = catgory.getEnd();
            c.value = catgory.getValue();
            c.score = catgory.getScore();
            c.ref = ref;

            String key = beginEndKey(catgory.getBegin(), catgory.getEnd());
            if (!list.containsKey(key)) {
                list.put(key, new ArrayList<>());
            }
            list.get(key).add(c);
        }

    }
}
