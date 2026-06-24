#!/usr/bin/env bash
# Avvia HDFS (macOS locale). Esegui setup_hadoop_macos.sh la prima volta.
HADOOP_HOME="$(brew --prefix hadoop)/libexec"
"$HADOOP_HOME/sbin/start-dfs.sh"
echo "HDFS Web UI → http://localhost:9870"
