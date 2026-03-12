#!/bin/bash

# URL des Docker-Endpunkts
URL="http://localhost:8000/v1/process"

# Datei mit Testtexten
TEST_FILE="test_text.json"

echo "Starte Tests"

# Schleife über alle Testtexte
jq -c '.[]' "$TEST_FILE" | while read t; do
  raw_text=$(echo "$t" | jq -r '.text')
  lang=$(echo "$t" | jq -r '.lang')

  # Wichtig: Prefix "sentence:" für REBEL
  text="sentence: $raw_text"

  # JSON für POST-Request bauen
  payload=$(jq -n --arg txt "$text" --arg l "$lang" '{
    doc_len: 1,
    lang: $l,
    selections: [
      {
        selection: "1",
        sentences: [
          { text: $txt, begin: 0, end: ($txt | length) }
        ]
      }
    ]
  }')

  # Anfrage an Docker-Endpunkt
  response=$(curl -s -X POST "$URL" -H "Content-Type: application/json" -d "$payload")

  # Triplets ausgeben
  relations=$(echo "$response" | jq '.relations[0][0]')

  echo "Text: $raw_text"
  echo "Triplets: $relations"
  echo "-----------------------------"
done

echo "Alle Tests abgeschlossen."