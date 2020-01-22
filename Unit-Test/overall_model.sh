#!/bin/bash

example_models="example";
for model in $example_models; do
    cd ../$model;
    python LanguageModel.py;
    cd -;
done

regular_models="BM25 Vector-Space-Model";
for model in $regular_models; do
    cd ../$model;
    python main.py;
    cd -;
done
