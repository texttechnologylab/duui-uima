#!/bin/bash

# URL des Docker-Endpunkts
URL="http://localhost:9714/v1/process"

# Datei mit Testtexten
TEST_FILE="Test_text.json"

echo "Starte Tests"

jq -c '.[]' "$TEST_FILE" | while read t; do
  text=$(echo "$t" | jq -r '.text')
  expected=$(echo "$t" | jq -r '.expected')

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

  echo "Text: $text"
  echo "Expected: $expected"
  echo "Scores:"

  # jq iteriert sicher über Labels und Scores
  echo "$response" | jq -r '
    .results[0] as $labels |
    .factors[0] as $scores |
    to_entries[] | "\($labels[.key]): \($scores[.key])"
  '

  # Predicted = Label mit höchstem Score
  max_index=$(echo "$response" | jq '
    .factors[0] | to_entries | max_by(.value) | .key
  ')
  pred_label=$(echo "$response" | jq -r ".results[0][$max_index]")
  pred_score=$(echo "$response" | jq -r ".factors[0][$max_index]")

  echo "Predicted (highest score): $pred_label ($pred_score)"
  echo "-----------------------------"
done