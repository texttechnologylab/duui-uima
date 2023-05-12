package org.texttechnologylab.duui.io.mongoDB;

import com.mongodb.*;
import com.mongodb.client.MongoDatabase;
import com.mongodb.gridfs.GridFS;
import com.mongodb.util.JSON;
import de.tudarmstadt.ukp.dkpro.core.api.metadata.type.DocumentMetaData;
import org.apache.uima.UimaContext;
import org.apache.uima.cas.CAS;
import org.apache.uima.collection.CollectionException;
import org.apache.uima.fit.descriptor.ConfigurationParameter;
import org.apache.uima.resource.ResourceInitializationException;
import org.apache.uima.util.Progress;
import org.apache.uima.util.ProgressImpl;
import org.hucompute.services.uima.database.AbstractCollectionReader;
import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.Arrays;

/**
 * MongoDBReader
 * <p>
 * Reads the data from a MongoDB
 *
 * @author Anna-Lena Buccoli
 */
public class MongoDBReader extends AbstractCollectionReader {

    /**
     * DB Username
     */
    public static final String PARAM_COLLECTION = "sCollection";
    @ConfigurationParameter(name = PARAM_COLLECTION, mandatory = false, defaultValue = "data")
    protected String sCollection;

    /**
     * DB Username
     */
    public static final String PARAM_DB_USER = "db_connection_user";
    @ConfigurationParameter(name = PARAM_DB_USER, mandatory = false)
    protected String db_connection_user;

    /**
     * DB Password
     */
    public static final String PARAM_DB_PW = "db_connection_pw";
    @ConfigurationParameter(name = PARAM_DB_PW, mandatory = false)
    protected String db_connection_pw;

    /**
     * DB Host
     */
    public static final String PARAM_DB_HOST = "db_connection_host";
    @ConfigurationParameter(name = PARAM_DB_HOST, mandatory = false, defaultValue = "localhost")
    protected String db_connection_host;

    /**
     * DB Name
     */
    public static final String PARAM_DB_DBNAME = "db_connection_dbname";
    @ConfigurationParameter(name = PARAM_DB_DBNAME, mandatory = true)
    protected String db_connection_dbname;
    // UserRO
    //=MZnJ%AdjQ:Y
    /**
     * DB authentication source
     */
    public static final String PARAM_DB_AUTH_SOURCE = "db_connection_auth_source";
    @ConfigurationParameter(name = PARAM_DB_AUTH_SOURCE, mandatory = false, defaultValue = "admin")
    protected String db_connection_auth_source;

    /**
     * DB Port
     */
    public static final String PARAM_DB_PORT = "db_connection_port";
    @ConfigurationParameter(name = PARAM_DB_PORT, mandatory = false, defaultValue = "27017")
    protected String db_connection_port;

    /**
     * Docker container name
     */
    public static final String PARAM_CONTAINER_NAME = "mongo_container_name";
    @ConfigurationParameter(name = PARAM_CONTAINER_NAME, mandatory = false)
    protected String mongo_container_name;

    /**
     * GridFS Bucket name
     */
    public static final String PARAM_GRIDFS_BUCKET = "gridfs_bucket";
    @ConfigurationParameter(name = PARAM_GRIDFS_BUCKET, mandatory = false, defaultValue = "fs")
    protected String gridfs_bucket;

    protected int processed = 0;
    private DBCursor cur;
    private GridFS grid;

    protected MongoClient pClient = null;

    public MongoDatabase connect() throws ResourceInitializationException {
        MongoClient mongoClient;

        if (db_connection_host != null && db_connection_host.equals("localhost")) {
            db_connection_host = null;
        }

        // Create database connection
        // If the parameter host is defined, the connection is established to this host (otherwise localhost is used)
        if (db_connection_host != null) {
            MongoCredential credential = MongoCredential.createScramSha1Credential(db_connection_user, db_connection_auth_source, db_connection_pw.toCharArray());
            mongoClient = new MongoClient(new ServerAddress(db_connection_host, Integer.parseInt(db_connection_port)), Arrays.asList(credential));
        } else {
            // Start a mongo docker container
            // If a name of a docker container is defined, this container will be started
            if (mongo_container_name != null) {
                try {
                    Process startContainer = Runtime.getRuntime().exec("docker start " + mongo_container_name);
                } catch (Exception e) {
                    throw new ResourceInitializationException(e);
                }
            }
            mongoClient = new MongoClient(new MongoClientURI("mongodb://localhost:" + db_connection_port));
            pClient = mongoClient;
        }
        MongoDatabase pDatabase = mongoClient.getDatabase(db_connection_dbname);

        return pDatabase;

    }

    /**
     * Initialize database connection
     *
     * @param context UimaContext
     * @throws ResourceInitializationException
     */
    @Override
    public void initialize(UimaContext context) throws ResourceInitializationException {

        super.initialize(context);
        MongoDatabase pDatabase = connect();
        // All documents of the GridFS bucket will be read
        grid = new GridFS(pClient.getDB(db_connection_dbname), gridfs_bucket);
        cur = grid.getFileList();

    }

    /**
     * Read a file from MongoDB and convert it into a CAS
     *
     * @param aCAS contains the data
     * @throws IOException
     * @throws CollectionException
     */
    @Override
    public void getNext(CAS aCAS) throws IOException, CollectionException {

        DBObject doc = cur.next();
        DBObject metaData = grid.findOne(doc).getMetaData();
        JSONObject jsonMetaData = new JSONObject(JSON.serialize(metaData));

        // Initialize CAS
        try {
            DocumentMetaData docMetaData = DocumentMetaData.create(aCAS);
            docMetaData.setDocumentTitle(jsonMetaData.getString("documentTitle"));
            docMetaData.setDocumentUri(jsonMetaData.getString("documentUri"));
            docMetaData.setDocumentId(jsonMetaData.getString("documentId"));
            if (jsonMetaData.has("documentBaseUri")) {
                docMetaData.setDocumentBaseUri(jsonMetaData.getString("documentBaseUri"));
            }
            if (jsonMetaData.has("documentCollectionId")) {
                docMetaData.setCollectionId(jsonMetaData.getString("collectionId"));
            }
        } catch (Exception e) {
            System.out.println(e);
        }

        // Set language and text into the CAS Object
        String docLanguage = new JSONObject(JSON.serialize(metaData)).getString("language");
        aCAS.setDocumentLanguage(docLanguage);

        String sofa = new JSONObject(JSON.serialize(metaData)).get("sofa").toString();
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        grid.findOne(doc).writeTo(out);
        JSONObject jsonObject = new JSONObject(out.toString());
        String docText = jsonObject.getJSONObject("_referenced_fss").getJSONObject(sofa).getString("sofaString");
        aCAS.setDocumentText(docText);

        if (!hasNext()) {
            Process stopContainer = Runtime.getRuntime().exec("docker stop " + mongo_container_name);
        }

        processed++;
    }

    @Override
    public boolean hasNext() throws IOException, CollectionException {
        return cur.hasNext();
    }

    @Override
    public Progress[] getProgress() {
        return new Progress[]{new ProgressImpl(processed, cur.count(), Progress.ENTITIES)};
    }
}
