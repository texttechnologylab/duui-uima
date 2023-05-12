package org.texttechnologylab.duui.io.finance;

import de.tudarmstadt.ukp.dkpro.core.api.metadata.type.DocumentMetaData;
import org.apache.uima.cas.CAS;
import org.apache.uima.cas.CASException;
import org.apache.uima.collection.CollectionException;
import org.bson.Document;
import org.texttechnologylab.annotation.AnnotationComment;
import org.texttechnologylab.duui.io.patent.PatentReaderMongoDB;

import java.io.IOException;

public class FinanceReaderMongoDB extends PatentReaderMongoDB {

    @Override
    public void getNext(CAS aCAS) throws IOException, CollectionException {

        boolean bProceed = false;

        if (this.current.hasNext()) {
            bProceed = true;
        } else if (findNext()) {
            bProceed = true;
        }

        if (bProceed) {

            Document pDocument = this.current.next();

            aCAS.setDocumentText(pDocument.getString("headline") + "\n" + pDocument.getString("text"));

            try {
                DocumentMetaData dmd = DocumentMetaData.create(aCAS);
                dmd.setDocumentTitle(pDocument.getString("title"));
                try {
                    System.out.println("Reading: " + pDocument.getObjectId("_id"));
                    dmd.setDocumentId(pDocument.getObjectId("_id").toString());
                } catch (Exception e) {
                    System.out.println("Reading: " + pDocument.getString("_id"));
                    dmd.setDocumentId(pDocument.getString("_id"));
                }

                AnnotationComment ac = new AnnotationComment(aCAS.getJCas());
                ac.setKey("date");
                ac.setValue(pDocument.getString("date"));
                ac.setReference(dmd);
                ac.addToIndexes();

            } catch (CASException e) {
                throw new RuntimeException(e);
            }

        }

        processed++;

    }

}
