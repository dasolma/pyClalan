CLALAN

Se proporciona la herramienta clalan.py al cual se encarga de descargar el corpus en caso de no disponer de él,
entrenar varios clasificadores probando diferentes preprocesados y configuraciones de parámetros para encontrar
el mejor clasificador posible y posteriormente evalúa el rendimiento mostrando las matrices de confusion sobre
los datos de entrenamiento y los de test.

Los datos entrenamiento y test se almacena en las carpetas "test" y "train". Para cada lenguaje se creará un
archivo [lang].txt donde se almacena el corpus de ese lenguaje. Si el corpus (archivo) de un lenguaje que se
quiere clasificar no existiera se descargaría automaticamente.

Desde el archivo config.py se establecen las configuraciones de los clasificadores y del preprocesado así como el
número de palabras a usar para el entretamiento y para los test.

Uso:

$ python clalan.py -l <languajes> [-i <inputfile>] [-o <outputfile>] [-d] [-t]

donde:

   -d <languages> es uan lista separada por comas de los lenaguajes. Se deben indicar los códigos de lenguaje
              ISO-639-1 (http://es.wikipedia.org/wiki/ISO_639-1).
   -i <inputfile> es el nombre de un clasificador ya existente. Sin no se especifica ninguno se buscaría "class.pk".
            En el caso de no existir se crearían los clasificadores.
   -o <outputfile> es el archivo donde guardar la respuesta. Si no se especifica se usará la salida estandar.

   -d Obliga a descargar el corpus aún existiendo.

   -t Obliga a realizar el entretamiento aún existiendo el clasificador.



RESULTADOS

Los resultados obtenidos son muy buenos con un 91% de fiabilidad sobre el conjunto de test (100.000 palabras). El
clasificador Multinomial Naive Bayes es el que consigue mejores resultados incluso sin necesidad de usar tf-idf.


Usando 1.000.000 de palabras de entrenamiento y 100.000 de test:

Results:

                                                               f1              recall           precision            accuracy
vect->LR                                           0.892128966017      0.892485378548      0.900535651959      0.892485378548
vect->tfidf->svm                                   0.901981766475      0.901942372699      0.907851654748      0.901942372699
vect->tfidf->LR                                    0.894251923718      0.894618232548      0.902004997889      0.894618232548
vect->svm                                          0.892584329509      0.892928046359      0.901073151115      0.892928046359
vect->mNB                                          0.915740566994      0.916952835757       0.91945870916      0.916952835757
vect->tfidf->mNB                                   0.914152005819      0.915517518914      0.918488531648      0.915517518914

Confusion matrix over train train:
             es        pl        be        it
es       179071      6141      5136      9920
pl         2007    119555       599      1994
be          630       598    136864       580
it         4450      2212       529    143744

Confusion matrix over test train:
             es        pl        be        it
es        17264      1578       710      1939
pl          302     17180        67       404
be           66        85     17077       111
it          523       349        57     16836


Se proporciona también un modelo "class2.pk" entrenado con 100.000 palabras únicamente. Aunque lo resultados son
peores, también son bastante buenos (81% de fiabilidad):

Results:

                                                               f1              recall           precision            accuracy
ual->vect->tfidf->LR                               0.772842062868      0.778223681544      0.827076477326      0.778223681544
vect->LR                                           0.734259442031      0.744659288986      0.815908262289      0.744659288986
ual->vect->tfidf->mNB                               0.80542236915      0.811044901158      0.846366342238      0.811044901158
vect->tfidf->svm                                   0.775122068505      0.780466712608      0.827925178754      0.780466712608
vect->tfidf->LR                                    0.772842062868      0.778223681544      0.827076477326      0.778223681544
ual->vect->LR                                      0.734259442031      0.744659288986      0.815908262289      0.744659288986
ual->vect->tfidf->svm                              0.775122068505      0.780466712608      0.827925178754      0.780466712608
ual->vect->svm                                     0.735075428186      0.745348412988      0.816282084053      0.745348412988
ual->vect->mNB                                      0.80687798354      0.812625832692      0.848102602387      0.812625832692
vect->svm                                          0.735075428186      0.745348412988      0.816282084053      0.745348412988
vect->mNB                                           0.80687798354      0.812625832692      0.848102602387      0.812625832692
vect->tfidf->mNB                                    0.80542236915      0.811044901158      0.846366342238      0.811044901158

Confusion matrix over train train:
             es        pl        be        it
es       154562      3017      1322     10942
pl        26412    122554     39237     26725
be          909      1134    103654       985
it         3861      1668       541    119960

Confusion matrix over test train:
             es        pl        be        it
es        14993       211       138      1353
pl         2427     17774      5575      3300
be           80       140     13178       140
it          314       128        61     14195

EL CORPUS

Para la obtención del corpus se ha usado una lista de personajes famosos (train/people.txt). Para obtener los datos de
entrenamiento se recorren los personajes en orden alfabética y para los datos de entrenamiento en orden inversa. Se ha
comprobado que no existe solapamiento en los datos proporcionados.


EL CÓDIGO

- clala.py. Es el script para ser ejecutado desde símbolo de comando explicado anteriormente.
- config.py. Configuraciones de vectorizadores, preprocesado, clasificadores y número de palabras
- dataretriever.py. Funciones para la descarga y lectura de los datos de entretamiento y test.
- pipelines.py. Funciones para la generación de todos los pipelines (sklearn) según la configuración en config.py.
- modelanalizer.py. La clase ModelAnalizer toma la configuración de transformadores, clasificadores y parametros.
              De cada combinación de transformadores + clasificador buscar los parametros de entre los datos y crea
              un modelo de clasificación.

NOTAS

- IMPORTANTE: Para la carga del modelo class.pk se necesita un mínimo de 6Gb de memoria libre.
