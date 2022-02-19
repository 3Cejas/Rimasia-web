# -*- coding: utf-8 -*-
from mmap import mmap
from flask import Flask, render_template, url_for, request
from gensim.models import KeyedVectors
from pyverse import Pyverse
from pyfasttext import FastText

app = Flask(__name__)

 

@app.route('/')
@app.route('/home')
def home():
    return render_template("index.html")



@app.route('/result',methods=['POST', 'GET'])
def result():
    output = request.form.to_dict()
    if not (len(output["palabra"]) == 0):
        palabra = output["palabra"]
        temas = output["temas"].replace(" ", "").split(",")
        KeyedVectors.load('normalized.vec',mmap='r')
        resultado = busca_rima_y_tema(palabra, temas)
        consonante = " ".join(resultado[0])
        asonante = " ".join(resultado[1])
        if len(resultado[0]) == 0:
            consonante= "nada"
        if len(resultado[1]) == 0:
            asonante = "nada"
    return render_template('index.html', palabra = palabra, consonante = consonante, asonante = asonante, temas = ", ".join(temas))




# Función auxiliar que te devuelve una lista de palabras relacionadas con una serie de temas
# y que rimen en consonante y asonante con una palabra dada.
def busca_rima_y_tema(palabra, temas):
    rimas = wordvectors.most_similar_cosmul(positive=temas,topn=1000,negative=[])
    if(len(rimas) == 0):
        print("No se han encontrado resultados")
    else:
        print(rimas)
        verse = Pyverse(palabra)
        res_consonante = []
        res_asonante = []
        for p in rimas:
            if(len(p[0])>1):
                cmp = Pyverse(p[0])
                if(cmp.consonant_rhyme == verse.consonant_rhyme and p[0] != palabra):
                    res_consonante.append(p[0] +",")
                if(cmp.assonant_rhyme == verse.assonant_rhyme and p[0] != palabra):
                    res_asonante.append(p[0] +",")
        print("Consonante: ", res_consonante)
        print("Asonante: ", res_asonante)
    return [res_consonante, res_asonante]


if __name__ == "__main__":
    # Cargamos el word embedding.
    wordvectors_file_vec = 'embeddings-l-model.vec'
    model = KeyedVectors.load_word2vec_format(wordvectors_file_vec)
    model.init_sims(replace=True)
    model.save('normalized.vec')

    wordvectors = KeyedVectors.load('normalized.vec', mmap='r')
    print("WORD EMBEDDING CARGADO")
    app.run(debug=True)