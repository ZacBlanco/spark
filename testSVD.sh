#! /usr/bin/env bash

./build/mvn -Dtest=none -DwildcardSuites=org.apche.spark.mllib.linalg.distributed.fastSVDSuite test
 
