#!/usr/bin/env bash
# Ferma HDFS (macOS locale).
HADOOP_HOME="$(brew --prefix hadoop)/libexec"
"$HADOOP_HOME/sbin/stop-dfs.sh"
