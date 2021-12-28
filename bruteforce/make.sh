#!/bin/bash

clang -Wall -O3 -march=native convert_texmex_to_serialized_array.c -o convert_texmex_to_serialized_array
clang++ -Wall -O3 -march=native -mllvm -polly search.cpp -o search
