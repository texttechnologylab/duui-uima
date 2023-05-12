package org.hucompute.textimager.uima.ddc.fasttext.service.service;

import java.io.*;
import java.util.Scanner;

public class FastTextProcess {
    // fastText Parameter
    int num_labels;
    String model;
    String language;
    String fasttext;

    // fastText Process
    Process process;
    BufferedWriter in;
    Scanner out;

    boolean is_loaded;

    FastTextProcess(String language, String model, String fasttext, int num_labels) throws FileNotFoundException {
        if (!new File(fasttext).exists()) {
            throw new FileNotFoundException("fastText executable not found at \"" + fasttext + "\"");
        }

        if (!new File(model).exists()) {
            throw new FileNotFoundException("fastText model file not found at \"" + model + "\"");
        }

        if (num_labels < 1) {
            throw new IllegalArgumentException("fastText num_labels must be at leas 1");
        }

        this.language = language;
        this.model = model;
        this.num_labels = num_labels;
        this.fasttext = fasttext;

        this.is_loaded = false;

        System.out.println("[" + this.language + "] Initialized fastText process");
    }

    void start() throws Exception {
        System.out.println("[" + this.language + "] Starting fastText process with location [" + this.fasttext + "], model [" + this.model + "], labels num [" + this.num_labels + "]");

        // Starten mit Predict, - zum lesen von stdin und k (Anzahl Ausgaben)
        ProcessBuilder builder = new ProcessBuilder(fasttext, "predict-prob", model, "-", String.valueOf(num_labels));
        try {
            process = builder.start();

            // get stdin of shell
            in = new BufferedWriter(new OutputStreamWriter(process.getOutputStream()));

            // Stdout
            out = new Scanner(process.getInputStream());

            this.is_loaded = true;

            System.out.println("[" + this.language + "] Successfully started fastText process with location [" + this.fasttext + "], model [" + this.model + "], labels num [" + this.num_labels + "]");

        } catch (Exception e) {
            throw new Exception("Failed to start fastText process", e);
        }
    }

    void exit() {
        if (!isLoaded()) {
            return;
        }

        System.out.println("[" + this.language + "] Exiting fastText process...");

        this.is_loaded = false;

        out.close();
        process.destroy();

        process = null;
        in = null;
        out = null;
    }

    boolean stdin(String text) {
        try {
            in.write(text);
            in.newLine();
            in.flush();
        } catch (IOException e) {
            System.out.println(e.getMessage());
            return false;
        }
        return true;
    }

    String next() {
        return out.next();
    }

    int getNumLabels() {
        return num_labels;
    }

    boolean isLoaded() {
        return is_loaded;
    }
}
