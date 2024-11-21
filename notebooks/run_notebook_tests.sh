#!/bin/zsh


FILES=/path/to/*
for f in *.ipynb
do
  echo "Testing $f ..."
  runipy "$f" 2>&1 | awk '/Cell raised uncaught exception/,EOF'
done
