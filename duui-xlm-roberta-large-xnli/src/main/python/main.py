from transformers import pipeline
classifier = pipeline("zero-shot-classification",
                      model="joeddav/xlm-roberta-large-xnli")


sequence_to_classify = "Der Algorithmus von Dijkstra (nach seinem Erfinder Edsger W. Dijkstra) ist ein Algorithmus aus der Klasse der Greedy-Algorithmen[1] und löst das Problem der kürzesten Pfade für einen gegebenen Startknoten. Er berechnet somit einen kürzesten Pfad zwischen dem gegebenen Startknoten und einem der (oder allen) übrigen Knoten in einem kantengewichteten Graphen (sofern dieser keine Negativkanten enthält). Für unzusammenhängende ungerichtete Graphen ist der Abstand zu denjenigen Knoten unendlich, zu denen kein Pfad vom Startknoten aus existiert. Dasselbe gilt auch für gerichtete nicht stark zusammenhängende Graphen. Dabei wird der Abstand synonym auch als Entfernung, Kosten oder Gewicht bezeichnet."

candidate_labels = ["Informatik", "Algorithmen", "Schminke", "Krieg", "Liebe", "Wege", "Wissenschaft"]
c = classifier(sequence_to_classify, candidate_labels)
print(c)

