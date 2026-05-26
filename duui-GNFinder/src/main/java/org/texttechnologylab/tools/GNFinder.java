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
import org.texttechnologylab.annotation.GNMetaData;
import org.texttechnologylab.annotation.biofid.gnfinder.Taxon;
import org.texttechnologylab.annotation.biofid.gnfinder.VerifiedTaxon;
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

                    System.out.println("args: " + args + "\n");

                    /* Set Metadata*/
                    JSONObject tMeta = rObject.getJSONObject("metadata");

                    GNMetaData metadata = new GNMetaData(jc);
                    metadata.setDate(tMeta.getString("date"));
                    metadata.setVersion(tMeta.getString("gnfinderVersion"));

                    if(tMeta.has("withUniqueNames"))
                        metadata.setWihUniqueNames(tMeta.getBoolean("withUniqueNames"));
                    else
                        metadata.setWihUniqueNames(false);

                    if(tMeta.has("withAllMatches"))
                        metadata.setWithAllMatches(tMeta.getBoolean("withAllMatches"));
                    else
                        metadata.setWithAllMatches(false);

                    if(tMeta.has("withAmbiguousNames"))
                        metadata.setWithAmbiguousNames(tMeta.getBoolean("withAmbiguousNames"));
                    else
                        metadata.setWithAmbiguousNames(false);

                    metadata.setWithBayes(!args.contains("-n"));  // no Bayes flag

                    if(args.contains("-s")) {  // sources flag
                        String specifiedSources = args.split("-s")[1].split(" -")[0];
                        metadata.setWithSources(specifiedSources);
                    }

                    String specifiedLanguage = null;
                    if(args.contains("-l")) {  // lang(uage) flag
                        specifiedLanguage = args.split("-l")[1].split(" -")[0];
                        metadata.setLang(specifiedLanguage);
                    }

                    metadata.addToIndexes();

                    /* Set Taxons */
                    JSONArray tArray = rObject.getJSONArray("names");

                    for (int a = 0; a < tArray.length(); a++) {
                        JSONObject tObject = tArray.getJSONObject(a);

                        int iBegin = tObject.getInt("start");
                        int iEnd = tObject.getInt("end");
                        String name = tObject.getString("name");
                        short cardinality = (short)tObject.getInt("cardinality");

                        Taxon gnTaxon = new Taxon(jc);
                        gnTaxon.setBegin(iBegin);
                        gnTaxon.setEnd(iEnd);
                        gnTaxon.setValue(name);
                        gnTaxon.setCardinality(cardinality);

                        if(specifiedLanguage != null)
                            gnTaxon.setLanguage(specifiedLanguage);

                        gnTaxon.addToIndexes();

                        if (tObject.has("verification")) {

                            JSONObject verification = tObject.getJSONObject("verification");

                            if (verification.has("bestResult")) {

                                JSONObject bestResult = verification.getJSONObject("bestResult");
                                addVerifiedTaxon(jc, bestResult, iBegin, iEnd, name);

                                /*
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
                                */

                            }else if(verification.has("results")){  // allMatches flag enabled

                                JSONArray results = verification.getJSONArray("results");

                                for (int i = 0; i < results.length(); i++) {
                                    addVerifiedTaxon(jc, results.getJSONObject(i), iBegin, iEnd, name);
                                }

                                /* Add Results */
                                /* First result best? */
                            }

                            /*
                            Taxon nTaxon = new Taxon(jc);
                            nTaxon.setBegin(iBegin);
                            nTaxon.setEnd(iEnd);
                            nTaxon.addToIndexes();


                            AnnotationComment ac = new AnnotationComment(jc);
                            ac.setReference(nTaxon);
                            ac.setKey("gnfinder_verification");
                            ac.setValue(verification.getString("curation"));
                            ac.addToIndexes();
                            */


                        }
                        else{
                            /* Add detected name? */
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

        private void addVerifiedTaxon(JCas jc, JSONObject verified, int begin, int end, String value) throws JSONException {
            VerifiedTaxon verifTaxon = new VerifiedTaxon(jc);
            verifTaxon.setBegin(begin);
            verifTaxon.setEnd(end);
            verifTaxon.setValue(value);

            if (verified.has("outlink")) {
                verifTaxon.setIdentifier(verified.getString("outlink"));
            }
            if (verified.has("currentRecordId")) {
                verifTaxon.setId(verified.getString("currentRecordId"));
            }
            if (verified.has("currentCanonicalFull")) {
                verifTaxon.setCurrentCanonical(verified.getString("currentCanonicalFull"));
            }
            if (verified.has("matchedCanonicalFull")) {
                verifTaxon.setMatchedCanonical(verified.getString("matchedCanonicalFull"));
            }
            if (verified.has("editDistance")) {
                verifTaxon.setVerifEditDistance((short)verified.getInt("editDistance"));
            }
            if (verified.has("matchedName")) {
                verifTaxon.setVerifMatchedName(verified.getString("matchedName"));
            }
            if (verified.has("matchType")) {
                verifTaxon.setVerifMatchType(verified.getString("matchType"));
            }
            if (verified.has("sortScore")) {
                verifTaxon.setVerifSortScore((float)verified.getDouble("sortScore"));
            }

            verifTaxon.addToIndexes();
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
