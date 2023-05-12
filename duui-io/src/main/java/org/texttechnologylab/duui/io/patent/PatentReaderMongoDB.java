package org.texttechnologylab.duui.io.patent;

import com.mongodb.client.MongoCollection;
import com.mongodb.client.MongoCursor;
import de.tudarmstadt.ukp.dkpro.core.api.metadata.type.DocumentMetaData;
import org.apache.uima.UimaContext;
import org.apache.uima.cas.CAS;
import org.apache.uima.cas.CASException;
import org.apache.uima.collection.CollectionException;
import org.apache.uima.fit.descriptor.ConfigurationParameter;
import org.apache.uima.resource.ResourceInitializationException;
import org.apache.uima.util.Progress;
import org.apache.uima.util.ProgressImpl;
import org.bson.Document;
import org.texttechnologylab.annotation.AnnotationComment;
import org.texttechnologylab.duui.io.mongoDB.MongoDBReader;

import java.io.IOException;

public class PatentReaderMongoDB extends MongoDBReader {

    public static final String PARAM_QUERY_LIMIT = "iLimit";
    @ConfigurationParameter(name = PARAM_QUERY_LIMIT, mandatory = false, defaultValue = "100")
    protected int iLimit;


    protected MongoCursor<Document> current = null;

    protected MongoCollection pCollection = null;

    @Override
    public void initialize(UimaContext context) throws ResourceInitializationException {

        this.pCollection = this.connect().getCollection(sCollection);

    }

    int iSkip = 0;

    public long getCount() {
        return this.pCollection.count();
    }

    public boolean findNext() {
        current = this.pCollection.find().limit(iLimit).skip(iSkip * iLimit).noCursorTimeout(true).iterator();
        iSkip++;
        return current.hasNext();
    }

    @Override
    public boolean hasNext() throws IOException, CollectionException {
        if (current == null) {
            return findNext();
        }
        if (!current.hasNext()) {
            return findNext();
        }
        return current.hasNext();
    }

    @Override
    public void getNext(CAS aCAS) throws IOException, CollectionException {

        boolean bProceed = false;

        if (current.hasNext()) {
            bProceed = true;
        } else if (findNext()) {
            bProceed = true;
        }

        if (bProceed) {

            Document pDocument = current.next();

            aCAS.setDocumentText(pDocument.getString("title") + "\n" + pDocument.getString("abstract"));

            try {
                DocumentMetaData dmd = DocumentMetaData.create(aCAS);
                dmd.setDocumentTitle(pDocument.getString("title"));
                dmd.setDocumentId(pDocument.getObjectId("_id").toString());

                AnnotationComment ac = new AnnotationComment(aCAS.getJCas());
                ac.setKey("id");
                ac.setValue(pDocument.getString("id"));
                ac.setReference(dmd);
                AnnotationComment authors = new AnnotationComment(aCAS.getJCas());
                authors.setKey("authors");
                authors.setValue(pDocument.getString("authors"));
                authors.setReference(dmd);
                AnnotationComment uri = new AnnotationComment(aCAS.getJCas());
                uri.setKey("uri");
                uri.setValue(pDocument.getString("link"));
                uri.setReference(dmd);

                ac.addToIndexes();
                authors.addToIndexes();
                uri.addToIndexes();

            } catch (CASException e) {
                throw new RuntimeException(e);
            }

        }

        processed++;

    }

    @Override
    public Progress[] getProgress() {
        return new Progress[]{new ProgressImpl(processed, (int) getCount(), Progress.ENTITIES)};
    }
}
