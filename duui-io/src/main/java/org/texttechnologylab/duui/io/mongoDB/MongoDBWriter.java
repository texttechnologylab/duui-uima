package org.texttechnologylab.duui.io.mongoDB;

import com.mongodb.*;
import com.mongodb.gridfs.GridFS;
import com.mongodb.gridfs.GridFSInputFile;
import org.apache.uima.UimaContext;
import org.apache.uima.analysis_engine.AnalysisEngineProcessException;
import org.apache.uima.fit.descriptor.ConfigurationParameter;
import org.apache.uima.jcas.JCas;
import org.apache.uima.json.JsonCasSerializer;
import org.apache.uima.resource.ResourceInitializationException;
import org.dkpro.core.api.io.JCasFileWriter_ImplBase;
import org.json.JSONObject;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.StringWriter;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;

/**
 * MongoDBWriter
 * <p>
 * Writes the data into a MongoDB
 *
 * @author Anna-Lena Buccoli
 */
public class MongoDBWriter extends JCasFileWriter_ImplBase {

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
    @ConfigurationParameter(name = PARAM_DB_DBNAME)
    protected String db_connection_dbname;

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
     * Docker image name
     */
    public static final String PARAM_DOCKER_IMAGE = "docker_image";
    @ConfigurationParameter(name = PARAM_DOCKER_IMAGE, mandatory = false, defaultValue = "mongo")
    protected String docker_image;

    /**
     * GridFS Bucket name
     */
    public static final String PARAM_GRIDFS_BUCKET = "gridfs_bucket";
    @ConfigurationParameter(name = PARAM_GRIDFS_BUCKET, mandatory = false, defaultValue = "fs")
    protected String gridfs_bucket;

    DB database;


    /**
     * Creates a MongoDBConnection
     *
     * @param context UimaContext
     * @throws ResourceInitializationException
     */
    @Override
    public void initialize(UimaContext context) throws ResourceInitializationException {

        super.initialize(context);
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
            // Create a mongo docker container
            // If a name of a docker container is defined, this container will be started
            if (mongo_container_name != null) {
                try {
                    Process createContainer = Runtime.getRuntime().exec("docker run --name " + mongo_container_name + " -d -p " + db_connection_port + ":27017 " + docker_image);
                    Process startContainer = Runtime.getRuntime().exec("docker start " + mongo_container_name);

                } catch (IOException ioException) {
                    ioException.printStackTrace();
                }
            }
            mongoClient = new MongoClient(new MongoClientURI("mongodb://localhost:" + db_connection_port));
        }
        database = mongoClient.getDB(db_connection_dbname);
    }


    /**
     * Saves the given JCas into a MongoDB
     *
     * @param aJCas contains the data
     * @throws AnalysisEngineProcessException
     */
    @Override
    public void process(JCas aJCas) throws AnalysisEngineProcessException {


        try {

            // Read out the data into a String
            JsonCasSerializer xcs = new JsonCasSerializer();
            xcs.setOmit0Values(true);
            StringWriter sw = new StringWriter();
            xcs.serialize(aJCas.getCas(), sw);

            // Config Document MetaData
            JSONObject docMetaData = new JSONObject(sw.toString()).getJSONObject("_views").getJSONObject("_InitialView").getJSONArray("DocumentMetaData").getJSONObject(0);
            DBObject metaDataDBObject = new BasicDBObject();
            for (String i : JSONObject.getNames(docMetaData)) {
                metaDataDBObject.put(i, docMetaData.get(i));
            }

            // Save data by using GridFS
            InputStream inputstream = new ByteArrayInputStream(sw.toString().getBytes(StandardCharsets.UTF_8));
            GridFS grid = new GridFS(database, gridfs_bucket);
            GridFSInputFile gridFile = grid.createFile(inputstream);
            gridFile.setMetaData(metaDataDBObject);
            gridFile.setChunkSize(15000000);
            gridFile.save();


        } catch (Exception e) {
            throw new AnalysisEngineProcessException();
        }
    }
}
