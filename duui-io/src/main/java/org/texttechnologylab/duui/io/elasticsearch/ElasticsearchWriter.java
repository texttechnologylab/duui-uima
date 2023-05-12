package org.texttechnologylab.duui.io.elasticsearch;

import co.elastic.clients.elasticsearch.ElasticsearchClient;
import co.elastic.clients.elasticsearch.core.IndexResponse;
import co.elastic.clients.json.jackson.JacksonJsonpMapper;
import co.elastic.clients.transport.ElasticsearchTransport;
import co.elastic.clients.transport.rest_client.RestClientTransport;
import com.fasterxml.jackson.core.io.SerializedString;
import org.apache.http.HttpHost;
import org.apache.http.auth.AuthScope;
import org.apache.http.auth.UsernamePasswordCredentials;
import org.apache.http.client.config.RequestConfig;
import org.apache.http.impl.client.BasicCredentialsProvider;
import org.apache.http.impl.nio.client.HttpAsyncClientBuilder;
import org.apache.http.ssl.SSLContextBuilder;
import org.apache.http.ssl.SSLContexts;
import org.apache.uima.UimaContext;
import org.apache.uima.analysis_engine.AnalysisEngineProcessException;
import org.apache.uima.fit.descriptor.ConfigurationParameter;
import org.apache.uima.jcas.JCas;
import org.apache.uima.json.JsonCasSerializer;
import org.apache.uima.resource.ResourceInitializationException;
import org.dkpro.core.api.io.JCasFileWriter_ImplBase;
import org.elasticsearch.client.RestClient;
import org.elasticsearch.client.RestClientBuilder;
import org.json.JSONObject;

import javax.net.ssl.SSLContext;
import java.io.IOException;
import java.io.StringWriter;
import java.util.concurrent.TimeUnit;

/**
 * ElasticsearchWriter
 * <p>
 * Save the given data in elasticsearch
 *
 * @author Anna-Lena Buccoli
 */
public class ElasticsearchWriter extends JCasFileWriter_ImplBase {

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

    RestClient restClient;
    ElasticsearchClient esClient;

    /**
     * Connection to org.texttechnologylab.duui.io.elasticsearch
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
        esClient = new ElasticsearchClient(transport);

    }

    /**
     * Saves the given JCas into org.texttechnologylab.duui.io.elasticsearch
     *
     * @param aJCas contains the data
     * @throws AnalysisEngineProcessException
     */
    @Override
    public void process(JCas aJCas) throws AnalysisEngineProcessException {

        // Read the data from the JCas
        JsonCasSerializer xcs = new JsonCasSerializer();
        xcs.setOmit0Values(true);
        StringWriter sw = new StringWriter();
        try {
            xcs.serialize(aJCas.getCas(), sw);
        } catch (IOException e) {
            e.printStackTrace();
        }

        JSONObject data = new JSONObject(sw.toString());

        // Save the data
        IndexResponse response = null;
        try {
            response = esClient.index(indexReq -> indexReq
                    .index(index)
                    .document(new SerializedString(data.toString()))
            );
        } catch (IOException e) {
            e.printStackTrace();
        }

        if (!hasNext()) {
            try {
                restClient.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

    }
}
