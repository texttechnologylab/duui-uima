#!/bin/bash

# URL des Docker-Endpunkts
URL="http://localhost:9714/v1/process"

# Datei mit Testtexten
TEST_FILE="test_text.json"

echo "Starte Tests"

# Schleife über alle Testtexte
jq -c '.[]' $TEST_FILE | while read t; do
  text=$(echo $t | jq -r '.text')
  expected=$(echo $t | jq -r '.expected')

  # JSON für POST-Request bauen
  payload=$(jq -n --arg txt "$text" '{
    doc_len: 1,
    lang: "ru",
    selections: [
      {
        selection: "1",
        sentences: [
          { text: $txt, begin: 0, end: ($txt | length) }
        ]
      }
    ]
  }')

  # curl Request
  response=$(curl -s -X POST "$URL" -H "Content-Type: application/json" -d "$payload")

  # Ergebnis extrahieren
  label=$(echo $response | jq -r '.results[0][0]')
  score=$(echo $response | jq -r '.factors[0][0]')

  echo "Text: $text"
  echo "Predicted: $label (score: $score), Expected: $expected"
  echo "-----------------------------"
done