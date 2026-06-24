#!/usr/bin/env bash
# ============================================================
# setup_hadoop_macos.sh
# Installa e avvia Hadoop HDFS nativo su macOS Apple Silicon.
# Eseguibile una sola volta; poi usa start_hdfs.sh / stop_hdfs.sh.
#
# Utilizzo:
#   chmod +x infra/setup_hadoop_macos.sh
#   ./infra/setup_hadoop_macos.sh
# ============================================================
set -euo pipefail

# ── colori per output leggibile ──────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ── 1. Verifica Homebrew ─────────────────────────────────────
command -v brew &>/dev/null || error "Homebrew non trovato. Installa da https://brew.sh"

# ── 2. Installa Hadoop se non presente ──────────────────────
if brew list hadoop &>/dev/null; then
  info "Hadoop già installato: $(hadoop version 2>/dev/null | head -1)"
else
  info "Installo Hadoop via Homebrew…"
  brew install hadoop
fi

# ── 3. Ricava HADOOP_HOME ───────────────────────────────────
HADOOP_HOME="$(brew --prefix hadoop)/libexec"
HADOOP_CONF="$HADOOP_HOME/etc/hadoop"
info "HADOOP_HOME = $HADOOP_HOME"

# ── 4. Ricava JAVA_HOME ─────────────────────────────────────
JAVA_HOME=$(/usr/libexec/java_home 2>/dev/null || true)
if [[ -z "$JAVA_HOME" ]]; then
  # Fallback: prova OpenJDK da brew
  if brew list openjdk &>/dev/null; then
    JAVA_HOME="$(brew --prefix openjdk)"
  elif brew list openjdk@11 &>/dev/null; then
    JAVA_HOME="$(brew --prefix openjdk@11)"
  else
    warn "Java non trovato — installo openjdk@11…"
    brew install openjdk@11
    JAVA_HOME="$(brew --prefix openjdk@11)"
  fi
fi
info "JAVA_HOME = $JAVA_HOME"

# ── 5. Scrivi hadoop-env.sh ─────────────────────────────────
info "Configuro hadoop-env.sh…"
cat > "$HADOOP_CONF/hadoop-env.sh" <<EOF
export JAVA_HOME=${JAVA_HOME}
export HADOOP_HOME=${HADOOP_HOME}
export HADOOP_LOG_DIR=\${HADOOP_HOME}/logs
EOF

# ── 6. Scrivi core-site.xml ─────────────────────────────────
info "Configuro core-site.xml…"
cat > "$HADOOP_CONF/core-site.xml" <<'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<?xml-stylesheet type="text/xsl" href="configuration.xsl"?>
<configuration>
  <property>
    <name>fs.defaultFS</name>
    <value>hdfs://localhost:9000</value>
  </property>
  <property>
    <name>hadoop.http.staticuser.user</name>
    <value>root</value>
  </property>
</configuration>
EOF

# ── 7. Scrivi hdfs-site.xml ─────────────────────────────────
info "Configuro hdfs-site.xml…"
HDFS_DATA="$HOME/.atp_oracle/hdfs/data"
HDFS_NAME="$HOME/.atp_oracle/hdfs/name"
mkdir -p "$HDFS_DATA" "$HDFS_NAME"

cat > "$HADOOP_CONF/hdfs-site.xml" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<?xml-stylesheet type="text/xsl" href="configuration.xsl"?>
<configuration>
  <property>
    <name>dfs.replication</name>
    <value>1</value>
  </property>
  <property>
    <name>dfs.namenode.name.dir</name>
    <value>file://${HDFS_NAME}</value>
  </property>
  <property>
    <name>dfs.datanode.data.dir</name>
    <value>file://${HDFS_DATA}</value>
  </property>
  <property>
    <name>dfs.permissions.enabled</name>
    <value>false</value>
  </property>
  <property>
    <name>dfs.webhdfs.enabled</name>
    <value>true</value>
  </property>
</configuration>
EOF

# ── 8. SSH localhost senza password (richiesto da Hadoop) ────
info "Configuro SSH localhost…"
if ! ssh -o BatchMode=yes -o ConnectTimeout=2 localhost true 2>/dev/null; then
  warn "SSH localhost non disponibile — aggiungo chiave…"
  if [[ ! -f "$HOME/.ssh/id_rsa" ]]; then
    ssh-keygen -t rsa -N "" -f "$HOME/.ssh/id_rsa" -q
  fi
  cat "$HOME/.ssh/id_rsa.pub" >> "$HOME/.ssh/authorized_keys"
  chmod 600 "$HOME/.ssh/authorized_keys"
  # Abilita Remote Login su macOS se necessario
  if ! sudo systemsetup -getremotelogin 2>/dev/null | grep -q "On"; then
    warn "Abilitando Remote Login (necessita password admin)…"
    sudo systemsetup -setremotelogin on || \
      warn "Non riesco ad abilitare Remote Login automaticamente. Vai in:\nSystem Settings → General → Sharing → Remote Login → ON"
  fi
fi

# ── 9. Formatta NameNode (solo la prima volta) ───────────────
if [[ ! -d "$HDFS_NAME/current" ]]; then
  info "Formato il NameNode (prima esecuzione)…"
  "$HADOOP_HOME/bin/hdfs" namenode -format -nonInteractive 2>&1 | grep -E "INFO|WARN|ERROR" | tail -5
else
  info "NameNode già formattato — salto il format."
fi

# ── 10. Avvia HDFS ───────────────────────────────────────────
info "Avvio HDFS…"
"$HADOOP_HOME/sbin/start-dfs.sh"

sleep 5

# ── 11. Crea directory layout su HDFS ───────────────────────
info "Creo la struttura directory su HDFS…"
DIRS=(
  /data/raw/rankings /data/raw/players /data/raw/matches /data/raw/match_stats
  /data/bronze /data/silver /data/gold/features /data/models /data/evaluation
)
for d in "${DIRS[@]}"; do
  "$HADOOP_HOME/bin/hdfs" dfs -mkdir -p "$d" 2>/dev/null || true
done
"$HADOOP_HOME/bin/hdfs" dfs -chmod -R 777 /data

# ── 12. Esporta variabili in ~/.zshrc se non già presenti ───
if ! grep -q "HADOOP_HOME" ~/.zshrc 2>/dev/null; then
  info "Aggiungo HADOOP_HOME a ~/.zshrc…"
  cat >> ~/.zshrc <<EOF

# Hadoop (aggiunto da setup_hadoop_macos.sh)
export HADOOP_HOME=${HADOOP_HOME}
export PATH=\$PATH:\$HADOOP_HOME/bin:\$HADOOP_HOME/sbin
EOF
fi

echo ""
echo -e "${GREEN}✅  HDFS avviato con successo!${NC}"
echo -e "   Web UI  →  http://localhost:9870"
echo -e "   RPC     →  hdfs://localhost:9000"
echo ""
echo -e "Per fermare HDFS:   ${YELLOW}${HADOOP_HOME}/sbin/stop-dfs.sh${NC}"
echo -e "Per riavviare HDFS: ${YELLOW}${HADOOP_HOME}/sbin/start-dfs.sh${NC}"
