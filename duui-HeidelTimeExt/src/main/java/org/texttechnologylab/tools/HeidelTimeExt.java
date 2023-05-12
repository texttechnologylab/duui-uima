package org.texttechnologylab.tools;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import com.sun.net.httpserver.HttpServer;
import de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence;
import de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token;
import de.unihd.dbs.uima.annotator.heideltime.HeidelTime;
import de.unihd.dbs.uima.annotator.heideltime.biofid.HeidelTimeBioFID;
import de.unihd.dbs.uima.types.heideltime.Timex3;
import org.apache.uima.UIMAException;
import org.apache.uima.analysis_engine.AnalysisEngine;
import org.apache.uima.cas.impl.XmiCasDeserializer;
import org.apache.uima.cas.impl.XmiCasSerializer;
import org.apache.uima.cas.impl.XmiSerializationSharedData;
import org.apache.uima.fit.factory.AggregateBuilder;
import org.apache.uima.fit.factory.JCasFactory;
import org.apache.uima.fit.factory.TypeSystemDescriptionFactory;
import org.apache.uima.fit.pipeline.SimplePipeline;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;
import org.apache.uima.resource.ResourceInitializationException;
import org.apache.uima.resource.metadata.TypeSystemDescription;
import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;
import org.texttechnologylab.annotation.type.Time;
import org.xml.sax.SAXException;

import java.io.IOException;
import java.io.OutputStream;
import java.io.StringWriter;
import java.net.InetSocketAddress;
import java.nio.charset.Charset;

import static org.apache.uima.fit.factory.AnalysisEngineFactory.createEngineDescription;

public class HeidelTimeExt {

    public static void main(String[] args) throws Exception {
        HttpServer server = HttpServer.create(new InetSocketAddress(9714), 0);
        server.createContext("/v1/communication_layer", new CommunicationLayer());
        server.createContext("/v1/typesystem", new TypesystemHandler());
        server.createContext("/v1/process", new ProcessHandler());
        server.createContext("/v1/details/input_output", new IOHandler());

        server.setExecutor(null); // creates a default executor
        server.start();
        System.out.println(HeidelTimeExt.class.getSimpleName()+" ready");
    }

    static class ProcessHandler implements HttpHandler {
        static JCas jc;
        private static AggregateBuilder pipeline = new AggregateBuilder();
        private static AnalysisEngine pAE = null;

        static {
            try {
                jc = JCasFactory.createJCas();
                pipeline.add(createEngineDescription(HeidelTimeBioFID.class));

            } catch (UIMAException e) {
                e.printStackTrace();
            }
        }

        @Override
        public void handle(HttpExchange t) throws IOException {
            try {
                jc.reset();

                XmiSerializationSharedData sharedData = new XmiSerializationSharedData();

                XmiCasDeserializer.deserialize(t.getRequestBody(), jc.getCas(), true, sharedData);
                pAE = pipeline.createAggregate();
                SimplePipeline.runPipeline(jc, pAE);

                for (Timex3 timex3 : JCasUtil.select(jc, Timex3.class)) {
                    Time nTime = new Time(jc);
                    nTime.setBegin(timex3.getBegin());
                    nTime.setEnd(timex3.getEnd());
                    nTime.setValue(timex3.getTimexType());
                    nTime.setIdentifier(timex3.getTimexValue());
                    nTime.addToIndexes();
                }

                t.sendResponseHeaders(200, 0);
                XmiCasSerializer.serialize(jc.getCas(), null, t.getResponseBody(), false, sharedData);

                t.getResponseBody().close();
            } catch (Exception e) {
                e.printStackTrace();
                t.sendResponseHeaders(404, -1);
            }

            t.getResponseBody().close();
        }
    }

    static class TypesystemHandler implements HttpHandler {
        @Override
        public void handle(HttpExchange t) throws IOException {
            try {
                TypeSystemDescription desc = TypeSystemDescriptionFactory.createTypeSystemDescription();
                StringWriter writer = new StringWriter();
                desc.toXML(writer);
                String response = writer.getBuffer().toString();

                t.sendResponseHeaders(200, response.getBytes(Charset.defaultCharset()).length);

                OutputStream os = t.getResponseBody();
                os.write(response.getBytes(Charset.defaultCharset()));

            } catch (ResourceInitializationException e) {
                e.printStackTrace();
                t.sendResponseHeaders(404, -1);
                return;
            } catch (SAXException e) {
                e.printStackTrace();
            } finally {
                t.getResponseBody().close();
            }

        }
    }

    static class IOHandler implements HttpHandler {
        @Override
        public void handle(HttpExchange t) throws IOException {
            try {
                JSONObject rObject = new JSONObject();
                rObject.put("input", new JSONArray().put(Token.class.getName()).put(Sentence.class.getName()));
                rObject.put("output", new JSONArray().put(Timex3.class.getName()).put(Time.class.getName()));
                String response = rObject.toString();
                t.sendResponseHeaders(200, response.getBytes(Charset.defaultCharset()).length);

                OutputStream os = t.getResponseBody();
                os.write(response.getBytes(Charset.defaultCharset()));

            } catch (JSONException e) {
                e.printStackTrace();
                t.sendResponseHeaders(404, -1);
                return;
            } finally {
                t.getResponseBody().close();
            }

        }
    }

    static class CommunicationLayer implements HttpHandler {
        @Override
        public void handle(HttpExchange t) throws IOException {
            String response = "serial = luajava.bindClass(\"org.apache.uima.cas.impl.XmiCasSerializer\")\n" +
                    "deserial = luajava.bindClass(\"org.apache.uima.cas.impl.XmiCasDeserializer\")" +
                    "function serialize(inputCas,outputStream,params)\n" +
                    "  serial:serialize(inputCas:getCas(),outputStream)\n" +
                    "end\n" +
                    "\n" +
                    "function deserialize(inputCas,inputStream)\n" +
                    "  inputCas:reset()\n" +
                    "  deserial:deserialize(inputStream,inputCas:getCas(),true)\n" +
                    "end";
            t.sendResponseHeaders(200, response.length());
            OutputStream os = t.getResponseBody();
            os.write(response.getBytes());
            os.close();
        }
    }
}
