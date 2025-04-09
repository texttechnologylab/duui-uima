package org.texttechnologylab.tools;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import com.sun.net.httpserver.HttpServer;
import org.apache.uima.UIMAException;
import org.apache.uima.cas.impl.XmiCasDeserializer;
import org.apache.uima.cas.impl.XmiCasSerializer;
import org.apache.uima.cas.impl.XmiSerializationSharedData;
import org.apache.uima.fit.factory.JCasFactory;
import org.apache.uima.fit.factory.TypeSystemDescriptionFactory;
import org.apache.uima.jcas.JCas;
import org.apache.uima.resource.ResourceInitializationException;
import org.apache.uima.resource.metadata.TypeSystemDescription;
import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;
import org.texttechnologylab.annotation.AnnotationComment;
import org.texttechnologylab.annotation.type.Taxon;
import org.texttechnologylab.utilities.helper.FileUtils;
import org.texttechnologylab.utilities.helper.TempFileHandler;
import org.xml.sax.SAXException;

import java.io.*;
import java.net.InetSocketAddress;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;

public class GNFinder {

    public static void main(String[] args) throws Exception {
        HttpServer server = HttpServer.create(new InetSocketAddress(9714), 0);
        server.createContext("/v1/communication_layer", new CommunicationLayer());
        server.createContext("/v1/typesystem", new TypesystemHandler());
        server.createContext("/v1/process", new ProcessHandler());
        server.createContext("/v1/details/input_output", new IOHandler());

        server.setExecutor(null); // creates a default executor
        server.start();
    }

    static class ProcessHandler implements HttpHandler {
        static JCas jc;

        static {
            try {
                jc = JCasFactory.createJCas();
            } catch (UIMAException e) {
                e.printStackTrace();
            }
        }

        @Override
        public void handle(HttpExchange t) throws IOException {
            File tf = null;
            try {
                jc.reset();

                XmiSerializationSharedData sharedData = new XmiSerializationSharedData();

                String body = new String(t.getRequestBody().readAllBytes());
                String[] bodies = body.split("\"}", 2);
                String args = bodies[0].split("args\":\"")[1];
                args = args + "-f compact";

                InputStream casBody = new ByteArrayInputStream(bodies[1].getBytes(StandardCharsets.UTF_8));
                XmiCasDeserializer.deserialize(casBody, jc.getCas(), true, sharedData);

                tf = TempFileHandler.getTempFile("aaa", "bbb");

                FileUtils.writeContent(jc.getDocumentText(), tf);

                Taxon tax = new Taxon(jc);

                Process process = null;
                try {
                    process = Runtime.getRuntime().exec("/gnfinder" + " " + tf.getAbsolutePath() + args);

                    BufferedReader reader = new BufferedReader(new InputStreamReader(
                            process.getInputStream()));
                    String s = null;
                    StringBuilder sb = new StringBuilder();
                    while (true) {
                        try {
                            if ((s = reader.readLine()) == null) break;
                        } catch (IOException e) {
                            e.printStackTrace();
                        }
                        sb.append(s);

                    }

                    JSONObject rObject = new JSONObject(sb.toString());

                    JSONArray tArray = rObject.getJSONArray("names");

                    for (int a = 0; a < tArray.length(); a++) {
                        JSONObject tObject = tArray.getJSONObject(a);

                        int iBegin = tObject.getInt("start");
                        int iEnd = tObject.getInt("end");

                        if (tObject.has("verification")) {

                            JSONObject verification = tObject.getJSONObject("verification");

                            if (verification.has("bestResult")) {

                                JSONObject bestResult = verification.getJSONObject("bestResult");

                                System.out.println(bestResult.toString(1));

                                Taxon nTaxon = new Taxon(jc);
                                nTaxon.setBegin(iBegin);
                                nTaxon.setEnd(iEnd);

                                if (bestResult.has("outlink")) {
                                    nTaxon.setIdentifier(bestResult.getString("outlink"));
                                }
                                if (bestResult.has("currentName")) {
                                    nTaxon.setValue(bestResult.getString("currentName"));
                                }
                                nTaxon.addToIndexes();

                                AnnotationComment ac = new AnnotationComment(jc);
                                ac.setReference(nTaxon);
                                ac.setKey("gnfinder_verification");
                                ac.setValue(bestResult.toString());
                                ac.addToIndexes();
                            }

                            Taxon nTaxon = new Taxon(jc);
                            nTaxon.setBegin(iBegin);
                            nTaxon.setEnd(iEnd);
                            nTaxon.addToIndexes();
                            nTaxon.addToIndexes();

                            AnnotationComment ac = new AnnotationComment(jc);
                            ac.setReference(nTaxon);
                            ac.setKey("gnfinder_verification");
                            ac.setValue(verification.getString("curation"));
                            ac.addToIndexes();

                        }

                    }


                } catch (IOException e) {
                    e.printStackTrace();
                } catch (JSONException e) {
                    e.printStackTrace();
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
                rObject.put("input", new JSONArray());
                rObject.put("output", new JSONArray().put("org.texttechnologylab.annotation.type.Taxon").put("org.texttechnologylab.annotation.AnnotationComment"));
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
            BufferedReader reader = new BufferedReader(new FileReader("./communication.lua"));
            StringBuilder stringBuilder = new StringBuilder();
            String line = null;
            String ls = System.getProperty("line.separator");
            while ((line = reader.readLine()) != null) {
                stringBuilder.append(line);
                stringBuilder.append(ls);
            }
            // delete the last new line separator
            stringBuilder.deleteCharAt(stringBuilder.length() - 1);
            reader.close();

            String response = stringBuilder.toString();

            t.sendResponseHeaders(200, response.length());
            OutputStream os = t.getResponseBody();
            os.write(response.getBytes());
            os.close();
        }
    }
}
