package org.texttechnologylab.duui.io.hocr;

import de.tudarmstadt.ukp.dkpro.core.api.metadata.type.DocumentMetaData;
import org.apache.uima.UIMAException;
import org.apache.uima.fit.factory.JCasFactory;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;
import org.texttechnologylab.annotation.ocr.OCRDocument;
import org.texttechnologylab.annotation.ocr.OCRPage;
import org.texttechnologylab.annotation.ocr.OCRParagraph;
import org.texttechnologylab.annotation.ocr.OCRToken;
import org.xml.sax.Attributes;
import org.xml.sax.SAXException;
import org.xml.sax.helpers.DefaultHandler;

import javax.xml.parsers.ParserConfigurationException;
import javax.xml.parsers.SAXParser;
import javax.xml.parsers.SAXParserFactory;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

public class HOCRLoader extends DefaultHandler {
    private Path filename;
    private JCas cas;

    private StringBuilder buffer;
    private List<HOCRPage> pages;

    private boolean inPage;
    private HOCRPage currentPage;

    private boolean inParagraph;
    private HOCRParagraph currentParagraph;

    private boolean inWord;
    private HOCRWord currentWord;

    private boolean inCharacter;
    private HOCRCharacter currentCharacter;

    static class HOCRElement {
        int elementCounter;

        boolean isEnd() {
            return elementCounter == 0;
        }

        public void decrElement() {
            elementCounter -= 1;
        }

        public void incrElement() {
            elementCounter += 1;
        }

        public HOCRElement() {
            this.elementCounter = 1;
        }
    }

    static class HOCRPage extends HOCRElement {
        static final String CLASS_TAG = "div";
        static final String CLASS_NAME = "ocr_page";

        String id;
        String title;
        List<HOCRParagraph> paragraphs;

        HOCRPage(Attributes attributes) {
            this.id = attributes.getValue("id");
            this.title = attributes.getValue("title");
            this.paragraphs = new ArrayList<>();
        }

        public void addParagraph(HOCRParagraph paragraph) {
            paragraphs.add(paragraph);
        }

        public List<HOCRParagraph> getParagraphs() {
            return paragraphs;
        }
    }

    static class HOCRParagraph extends HOCRElement {
        static final String CLASS_TAG = "p";
        static final String CLASS_NAME = "ocr_par";

        String id;
        String lang;
        List<HOCRWord> words;

        HOCRParagraph(Attributes attributes) {
            this.id = attributes.getValue("id");
            this.lang = attributes.getValue("lang");
            this.words = new ArrayList<>();
        }

        public void addWord(HOCRWord word) {
            words.add(word);
        }

        public List<HOCRWord> getWords() {
            return words;
        }

        public String getLang() {
            return lang;
        }
    }

    static class HOCRWord extends HOCRElement {
        static final String CLASS_TAG = "span";
        static final String CLASS_NAME = "ocrx_word";

        String id;
        List<HOCRCharacter> characters;

        HOCRWord(Attributes attributes) {
            this.id = attributes.getValue("id");
            this.characters = new ArrayList<>();
        }

        public void addCharacter(HOCRCharacter character) {
            characters.add(character);
        }

        public List<HOCRCharacter> getCharacters() {
            return characters;
        }
    }

    static class HOCRCharacter extends HOCRElement {
        static final String CLASS_TAG = "span";
        static final String CLASS_NAME = "ocrx_cinfo";

        int start;
        String character;

        HOCRCharacter(int start) {
            this.start = start;
            this.character = "";
        }

        public void setCharacter(StringBuilder buffer) {
            this.character = buffer.substring(start);
        }

        public String getCharacter() {
            return character;
        }
    }

    public JCas getCAS() {
        return cas;
    }

    public HOCRLoader(Path filename) throws UIMAException {
        this.filename = filename;
        this.cas = JCasFactory.createJCas();
    }

    public HOCRLoader(JCas cas) throws UIMAException {
        this.cas = cas;
    }

    @Override
    public void startDocument() throws SAXException {
        //cas.reset();

        buffer = new StringBuilder();
        pages = new ArrayList<>();

        inPage = false;
        currentPage = null;

        inParagraph = false;
        currentParagraph = null;

        inWord = false;
        currentWord = null;

        inCharacter = false;
        currentCharacter = null;
    }

    @Override
    public void endDocument() throws SAXException {
        StringBuilder text = new StringBuilder();
        int pageCounter = 0;
        Map<String, Integer> langCounter = new HashMap<>();

        String documentId = "undefined";
        DocumentMetaData meta = null;
        try {
            meta = DocumentMetaData.get(cas);
        }
        catch (IllegalArgumentException ex) {
            // ignore
        }
        if (meta != null) {
            documentId = meta.getDocumentId();
        } else if (filename != null) {
            documentId = filename.getFileName().toString();
        }

        for (HOCRPage page : pages) {
            pageCounter += 1;
            int beginPage = text.length();
            String pageId = documentId + "_" + pageCounter;

            for (HOCRParagraph paragraph : page.getParagraphs()) {
                if (!langCounter.containsKey(paragraph.getLang())) {
                    langCounter.put(paragraph.getLang(), 1);
                }
                else {
                    langCounter.put(paragraph.getLang(), langCounter.get(paragraph.getLang())+1);
                }

                int beginParagraph = text.length();

                for (HOCRWord word : paragraph.getWords()) {
                    int beginWord = text.length();
                    String wordText = word.getCharacters().stream().map(HOCRCharacter::getCharacter).collect(Collectors.joining());
                    text.append(wordText);
                    int endWord = text.length();
                    text.append(" ");
                    OCRToken ocrToken = new OCRToken(cas, beginWord, endWord);
                    ocrToken.addToIndexes();
                }

                int endParagraph = text.length();
                OCRParagraph ocrParagraph = new OCRParagraph(cas, beginParagraph, endParagraph);
                ocrParagraph.addToIndexes();
            }

            // remove trailing space
            int last = text.length()-1;
            text.replace(last, last+1, "");

            int endPage = text.length();
            OCRPage ocrPage = new OCRPage(cas, beginPage, endPage);
            ocrPage.setPageNumber(pageCounter);
            ocrPage.setPageId(pageId);
            ocrPage.addToIndexes();
        }

        String lang = "x-unspecified";
        int langMax = 0;
        for (Map.Entry<String, Integer> entry : langCounter.entrySet()) {
            if (entry.getValue() > langMax) {
                langMax = entry.getValue();
                lang = entry.getKey();
            }
        }

        cas.setDocumentText(text.toString());

        if (meta == null) {
            cas.setDocumentLanguage(lang);

            meta = DocumentMetaData.create(cas);
            meta.setDocumentId(documentId);
            meta.setDocumentTitle(documentId);
            if (filename != null) {
                meta.setDocumentUri(filename.toString());
                meta.setDocumentBaseUri(filename.getParent().toString());
            }
            meta.addToIndexes();
        }
        else {
            if (cas.getDocumentLanguage() == null || cas.getDocumentLanguage().equals("x-unspecified")) {
                cas.setDocumentLanguage(lang);
            }
        }

        OCRDocument ocrDocument = new OCRDocument(cas);
        ocrDocument.setBegin(0);
        ocrDocument.setEnd(cas.getDocumentText().length());
        ocrDocument.setDocumentname(documentId);
        ocrDocument.addToIndexes();
    }

    @Override
    public void startElement(String uri, String localName, String qName, Attributes attributes) throws SAXException {
        if (inWord && qName.equals(HOCRCharacter.CLASS_TAG)) {
            if (currentCharacter != null) {
                currentCharacter.incrElement();
            }
            if (attributes.getValue("class").equals(HOCRCharacter.CLASS_NAME)) {
                inCharacter = true;
                currentCharacter = new HOCRCharacter(buffer.length());
            }
        }

        if (inParagraph && qName.equals(HOCRWord.CLASS_TAG)) {
            if (currentWord != null) {
                currentWord.incrElement();
            }
            if (attributes.getValue("class").equals(HOCRWord.CLASS_NAME)) {
                inWord = true;
                currentWord = new HOCRWord(attributes);
            }
        }

        if (inPage && qName.equals(HOCRParagraph.CLASS_TAG)) {
            if (currentParagraph != null) {
                currentParagraph.incrElement();
            }
            if (attributes.getValue("class").equals(HOCRParagraph.CLASS_NAME)) {
                inParagraph = true;
                currentParagraph = new HOCRParagraph(attributes);
            }
        }

        if (qName.equals(HOCRPage.CLASS_TAG)) {
            if (currentPage != null) {
                currentPage.incrElement();
            }
            if (attributes.getValue("class").equals(HOCRPage.CLASS_NAME)) {
                inPage = true;
                currentPage = new HOCRPage(attributes);
            }
        }
    }

    @Override
    public void endElement(String uri, String localName, String qName) throws SAXException {
        if (qName.equals(HOCRCharacter.CLASS_TAG)) {
            if (currentCharacter != null) {
                currentCharacter.decrElement();
            }
            if (inCharacter && currentCharacter != null) {
                if (currentCharacter.isEnd()) {
                    inCharacter = false;
                    currentCharacter.setCharacter(buffer);
                    currentWord.addCharacter(currentCharacter);
                    currentCharacter = null;
                }
            }
        }

        if (qName.equals(HOCRWord.CLASS_TAG)) {
            if (currentWord != null) {
                currentWord.decrElement();
            }
            if (inWord && currentWord != null) {
                if (currentWord.isEnd()) {
                    inWord = false;
                    currentParagraph.addWord(currentWord);
                    currentWord = null;
                }
            }
        }

        if (qName.equals(HOCRParagraph.CLASS_TAG)) {
            if (currentParagraph != null) {
                currentParagraph.decrElement();
            }
            if (inParagraph && currentParagraph != null) {
                if (currentParagraph.isEnd()) {
                    inParagraph = false;
                    currentPage.addParagraph(currentParagraph);
                    currentParagraph = null;
                }
            }
        }

        if (qName.equals(HOCRPage.CLASS_TAG)) {
            if (currentPage != null) {
                currentPage.decrElement();
            }

            if (inPage && currentPage != null) {
                if (currentPage.isEnd()) {
                    inPage = false;
                    pages.add(currentPage);
                    currentPage = null;
                }
            }
        }
    }

    @Override
    public void characters(char[] ch, int start, int length) throws SAXException {
        if (inCharacter) {
            buffer.append(ch, start, length);
        }
    }

    public static JCas load(Path filename) throws ParserConfigurationException, SAXException, IOException, UIMAException {
        SAXParserFactory factory = SAXParserFactory.newInstance();

        SAXParser saxParser = factory.newSAXParser();
        HOCRLoader handler = new HOCRLoader(filename);

        saxParser.parse(filename.toFile(), handler);

        return handler.getCAS();
    }

    public static void load(InputStream stream, JCas cas) throws ParserConfigurationException, SAXException, IOException, UIMAException {
        SAXParserFactory factory = SAXParserFactory.newInstance();

        SAXParser saxParser = factory.newSAXParser();
        HOCRLoader handler = new HOCRLoader(cas);

        saxParser.parse(stream, handler);
    }

    public static void main(String[] args) throws ParserConfigurationException, IOException, SAXException, UIMAException {
        JCas cas = HOCRLoader.load(Paths.get("src/test/resources/hocr/mobot31753002306394_chocr.html"));

        for (OCRPage page : JCasUtil.select(cas, OCRPage.class)) {
            System.out.println(page.getPageNumber());
            System.out.println("!~!"+page.getCoveredText()+"!~!");
            for (OCRToken token : JCasUtil.selectCovered(OCRToken.class, page)) {
                System.out.println("!~!"+token.getCoveredText()+"!~!");
            }
            break;
        }

        /*try(GZIPOutputStream outputStream = new GZIPOutputStream(Files.newOutputStream(Paths.get("test.xmi.gz")))) {
            XMLSerializer xmlSerializer = new XMLSerializer(outputStream, true);
            xmlSerializer.setOutputProperty(OutputKeys.VERSION, "1.1");
            xmlSerializer.setOutputProperty(OutputKeys.ENCODING, StandardCharsets.UTF_8.toString());
            XmiCasSerializer xmiCasSerializer = new XmiCasSerializer(null);
            xmiCasSerializer.serialize(cas.getCas(), xmlSerializer.getContentHandler());
        }*/
    }
}
