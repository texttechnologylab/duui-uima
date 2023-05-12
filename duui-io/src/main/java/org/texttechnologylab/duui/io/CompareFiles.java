package org.texttechnologylab.duui.io;

import de.tudarmstadt.ukp.dkpro.core.api.lexmorph.type.pos.POS;
import de.tudarmstadt.ukp.dkpro.core.api.ner.type.NamedEntity;
import de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token;
import de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.Dependency;
import javafx.util.Pair;
import org.apache.uima.UIMAException;
import org.apache.uima.fit.factory.JCasFactory;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;
import org.apache.uima.util.CasIOUtils;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

/**
 * Class to compare files
 *
 * @author Anna-Lena Buccoli
 */
public class CompareFiles {

    /**
     * This function compares two files
     *
     * @param path1 path of the first file
     * @param path2 path of the second file
     * @return boolean whether the files are equal
     * @throws UIMAException
     */
    public static boolean compare(String path1, String path2) throws UIMAException {

        File file1 = new File(path1);
        File file2 = new File(path2);

        try {
            JCas jcas1 = JCasFactory.createJCas();
            JCas jcas2 = JCasFactory.createJCas();
            CasIOUtils.load(file1.toURL(), jcas1.getCas());
            CasIOUtils.load(file2.toURL(), jcas2.getCas());
            List<String> obj1 = new ArrayList<>();
            List<String> obj2 = new ArrayList<>();
            ArrayList<Pair<String, String>> obj1Pair = new ArrayList<Pair<String, String>>();
            ArrayList<Pair<String, String>> obj2Pair = new ArrayList<Pair<String, String>>();

            // Compare all Token
            JCasUtil.select(jcas1, Token.class).forEach(i -> {
                obj1.add(i.getCoveredText());
            });
            JCasUtil.select(jcas2, Token.class).forEach(i -> {
                obj2.add(i.getCoveredText());
            });

            if (!obj1.equals(obj2)) {
                return false;
            }

            obj1.clear();
            obj2.clear();

            // Compare all POS
            JCasUtil.select(jcas1, POS.class).forEach(i -> {
                obj1.add(i.getPosValue());
            });
            JCasUtil.select(jcas2, POS.class).forEach(i -> {
                obj2.add(i.getPosValue());
            });

            if (!obj1.equals(obj2)) {
                return false;
            }

            // Compare all named entities
            JCasUtil.select(jcas1, NamedEntity.class).forEach(i -> {
                obj1Pair.add(new Pair<>(i.getCoveredText(), i.getValue()));
            });
            JCasUtil.select(jcas2, NamedEntity.class).forEach(i -> {
                obj2Pair.add(new Pair<>(i.getCoveredText(), i.getValue()));
            });

            if (!obj1Pair.equals(obj2Pair)) {
                return false;
            }

            obj1Pair.clear();
            obj2Pair.clear();

            // Compare all dependencies
            JCasUtil.select(jcas1, Dependency.class).forEach(i -> {
                obj1Pair.add(new Pair<>(i.getDependent().getCoveredText(), i.getDependencyType()));
            });
            JCasUtil.select(jcas1, Dependency.class).forEach(i -> {
                obj2Pair.add(new Pair<>(i.getDependent().getCoveredText(), i.getDependencyType()));
            });

            if (!obj1Pair.equals(obj2Pair)) {
                return false;
            }

        } catch (Exception e) {
            throw new UIMAException();
        }

        return true;
    }
}