package org.texttechnologylab.duui.io.elasticsearch;

import co.elastic.clients.elasticsearch.ElasticsearchClient;
import co.elastic.clients.elasticsearch.core.SearchResponse;
import co.elastic.clients.elasticsearch.core.search.Hit;
import co.elastic.clients.json.JsonData;
import co.elastic.clients.json.jackson.JacksonJsonpMapper;
import co.elastic.clients.transport.ElasticsearchTransport;
import co.elastic.clients.transport.rest_client.RestClientTransport;
import de.tudarmstadt.ukp.dkpro.core.api.metadata.type.DocumentMetaData;
import org.apache.http.HttpHost;
import org.apache.http.auth.AuthScope;
import org.apache.http.auth.UsernamePasswordCredentials;
import org.apache.http.client.config.RequestConfig;
import org.apache.http.impl.client.BasicCredentialsProvider;
import org.apache.http.impl.nio.client.HttpAsyncClientBuilder;
import org.apache.http.ssl.SSLContextBuilder;
import org.apache.http.ssl.SSLContexts;
import org.apache.uima.UimaContext;
import org.apache.uima.cas.CAS;
import org.apache.uima.collection.CollectionException;
import org.apache.uima.fit.descriptor.ConfigurationParameter;
import org.apache.uima.resource.ResourceInitializationException;
import org.apache.uima.util.Progress;
import org.apache.uima.util.ProgressImpl;
import org.elasticsearch.client.RestClient;
import org.elasticsearch.client.RestClientBuilder;
import org.hucompute.services.uima.database.AbstractCollectionReader;
import org.json.JSONObject;

import javax.net.ssl.SSLContext;
import java.io.IOException;
import java.util.List;
import java.util.concurrent.TimeUnit;

/**
 * ElasticsearchReader
 * <p>
 * Reads the data from elasticsearch
 *
 * @author Anna-Lena Buccoli
 */
public class ElasticsearchReader extends AbstractCollectionReader {

    /**
     * Elasticsearch index name
     */
    public static final String PARAM_INDEX = "index";
    @ConfigurationParameter(name = PARAM_INDEX, mandatory = true)
    protected String index;

    /**
     * Elasticsearch username
     */
    public static final String PARAM_USERNAME = "username";
    @ConfigurationParameter(name = PARAM_USERNAME, mandatory = false, defaultValue = "elastic")
    protected String username;

    /**
     * Elasticsearch password
     */
    public static final String PARAM_PASSWORD = "password";
    @ConfigurationParameter(name = PARAM_PASSWORD, mandatory = true)
    protected String password;

    /**
     * Elasticsearch host
     */
    public static final String PARAM_HOST = "host";
    @ConfigurationParameter(name = PARAM_HOST, mandatory = false, defaultValue = "localhost")
    protected String host;

    /**
     * Elasticsearch port
     */
    public static final String PARAM_PORT = "port";
    @ConfigurationParameter(name = PARAM_PORT, mandatory = false, defaultValue = "9200")
    protected String port;

    /**
     * Connection scheme
     */
    public static final String PARAM_SCHEME = "scheme";
    @ConfigurationParameter(name = PARAM_SCHEME, mandatory = false, defaultValue = "https")
    protected String scheme;

    /**
     * Connection timeout
     */
    public static final String PARAM_CONNECT_TIMEOUT = "connectTimeout";
    @ConfigurationParameter(name = PARAM_CONNECT_TIMEOUT, mandatory = false, defaultValue = "5000")
    protected String connectTimeout;

    /**
     * Socket timeout
     */
    public static final String PARAM_SOCKET_TIMEOUT = "socketTimeout";
    @ConfigurationParameter(name = PARAM_SOCKET_TIMEOUT, mandatory = false, defaultValue = "60000")
    protected String socketTimeout;

    /**
     * Docker container name
     */
    public static final String PARAM_CONTAINER_NAME = "containerName";
    @ConfigurationParameter(name = PARAM_CONTAINER_NAME, mandatory = false)
    protected String containerName;

    /**
     * Docker timeout
     */
    public static final String PARAM_DOCKER_TIMEOUT = "dockerTimeout";
    @ConfigurationParameter(name = PARAM_DOCKER_TIMEOUT, mandatory = false, defaultValue = "30")
    protected String dockerTimeout;

    private int processed = 0;
    private List<Hit<JsonData>> hits;
    private RestClient restClient;

    /**
     * Initialize org.texttechnologylab.duui.io.elasticsearch connection
     *
     * @param context UimaContext
     * @throws ResourceInitializationException
     */
    @Override
    public void initialize(UimaContext context) throws ResourceInitializationException {

        super.initialize(context);

        // Start a org.texttechnologylab.duui.io.elasticsearch docker container
        // If a name of a docker container is defined, this container will be started
        if (containerName != null) {
            try {
                Process startContainer = Runtime.getRuntime().exec("docker start " + containerName);
                TimeUnit.SECONDS.sleep(Integer.parseInt(dockerTimeout));
            } catch (Exception e) {
                e.printStackTrace();
            }
        }

        // Connection to org.texttechnologylab.duui.io.elasticsearch
        BasicCredentialsProvider credentials = new BasicCredentialsProvider();
        credentials.setCredentials(AuthScope.ANY,
                new UsernamePasswordCredentials(username, password));

        restClient = RestClient.builder(
                        new HttpHost(host, Integer.parseInt(port), scheme))
                .setRequestConfigCallback(new RestClientBuilder.RequestConfigCallback() {
                    @Override
                    public RequestConfig.Builder customizeRequestConfig(RequestConfig.Builder builder) {
                        return builder.setConnectTimeout(Integer.parseInt(connectTimeout)).setSocketTimeout(Integer.parseInt(socketTimeout));
                    }
                })
                .setHttpClientConfigCallback(new RestClientBuilder.HttpClientConfigCallback() {
                    @Override
                    public HttpAsyncClientBuilder customizeHttpClient(HttpAsyncClientBuilder httpAsyncClientBuilder) {
                        SSLContextBuilder sslBuilder = null;
                        SSLContext sslContext = null;
                        try {
                            sslBuilder = SSLContexts.custom().loadTrustMaterial(null, ((x509Certificates, s) -> true));
                            sslContext = sslBuilder.build();
                        } catch (Exception e) {
                            e.printStackTrace();
                        }

                        httpAsyncClientBuilder.setSSLContext(sslContext);
                        return httpAsyncClientBuilder.setDefaultCredentialsProvider(credentials)
                                .setSSLHostnameVerifier((s, sslsession) -> true);
                    }
                }).build();

        ElasticsearchTransport transport = new RestClientTransport(restClient, new JacksonJsonpMapper());

        ElasticsearchClient esClient = new ElasticsearchClient(transport);

        // Read the data from the index
        SearchResponse<JsonData> response = null;
        try {
            response = esClient.search(s -> s
                    .index(index), JsonData.class);
        } catch (IOException e) {
            e.printStackTrace();
        }

        hits = response.hits().hits();

    }

    /**
     * Read a file from org.texttechnologylab.duui.io.elasticsearch and convert it into a CAS
     *
     * @param aCAS contains the data
     * @throws IOException
     * @throws CollectionException
     */
    @Override
    public void getNext(CAS aCAS) throws IOException, CollectionException {

        // Read the data and metadata into a JSONObject
        JsonData data = hits.get(processed).source();
        String jsonString = data.toJson().asJsonObject().getString("value");
        JSONObject jsonData = new JSONObject(jsonString);
        JSONObject jsonMetaData = jsonData.getJSONObject("_views").getJSONObject("_InitialView").getJSONArray("DocumentMetaData").getJSONObject(0);

        // initCas
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
        String docLanguage = jsonMetaData.getString("language");
        aCAS.setDocumentLanguage(docLanguage);
        String sofa = jsonMetaData.get("sofa").toString();
        String docText = jsonData.getJSONObject("_referenced_fss").getJSONObject(sofa).getString("sofaString");
        aCAS.setDocumentText(docText);

        if (!hasNext()) {
            Process stopContainer = Runtime.getRuntime().exec("docker stop " + containerName);
        }

        processed++;

    }

    @Override
    public boolean hasNext() throws IOException, CollectionException {
        if (processed < hits.size()) {
            return true;
        } else {
            try {
                restClient.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
            return false;
        }
    }

    @Override
    public Progress[] getProgress() {
        return new Progress[]{new ProgressImpl(processed, hits.size(), Progress.ENTITIES)};
    }
}
