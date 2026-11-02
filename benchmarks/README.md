# Benchmark durate pipeline (scrape + feature)

Tempo totale di `atp-scrape run` + `atp-features all`, per backend di storage.
I log completi di ogni run sono nella cartella con la data.

| Data | Backend | Scrape + scrittura raw/silver | `atp-features all` | Totale | Esito | Note |
|---|---|---|---|---|---|---|
| (run precedente) | S3 (AWS) | — | — | **1h 24m 40s** | ok | Sackmann scaricato da GitHub; tennis-data 2026 = 801 match |
| 2026-09-24 16:32 | local | 6m 47s | 1s | 6m 48s | **fallito** | Sackmann (repo GitHub rimosso) e tennis-data (URL cambiato) → 404, 0 match; features fallite per assenza di silver |
| 2026-09-24 16:42 | local | 21s | 1h 7m 27s | **1h 7m 48s** | ok | −16m 52s (−19,9%) vs S3; vedi dettaglio sotto |

## Run 2026-09-24 16:42 (local) — dettaglio fasi

| Ora | Fase | Durata |
|---|---|---|
| 16:42:16 | avvio `atp-scrape run` | |
| 16:42:17 | Step 1 ranking top 200, Step 2 bio (da cache HTML), Step 3 Sackmann (disco locale), Step 4 tennis-data | ~13s |
| 16:42:30 → 16:42:37 | scrittura raw JSON + silver Parquet su disco | ~7s |
| 16:42:37 | fine scrape: rankings=200, players=65.989, matches=130.961, match_stats=196.102 | **21s** |
| 16:42:38 → 16:42:43 | feature: caricamento silver, unificazione ID, Elo | ~5s |
| 16:42:43 → 17:31:53 | feature rolling per giocatore (130.961 match) | **~49m 10s** |
| 17:31:53 → 17:42:17 | H2H | **~10m 24s** |
| 17:42:17 → 17:42:18 | contesto match, join blocchi, interazioni | ~1s |
| 17:42:18 → 17:45:35 | target (parsing scoreline) | **~3m 17s** |
| 17:45:35 → 17:49:57 | feature tornei | **~4m 22s** |
| 17:49:57 → 17:49:58 | masking superficie, randomizzazione ruoli; shape (130.961, 143) | ~1s |
| 17:49:58 → 17:50:00 | scrittura gold + tournament profiles | ~2s |
| 17:50:00 → 17:50:03 | player profiles (200) + player×tournament history | ~3s |
| 17:50:04 | fine `atp-features all` (4047s) | **1h 7m 27s** |

**Totale: 4068s = 1h 7m 48s.**

Output: gold 130.961 × 144 (1986-01-06 → 2026-09-13, `winner_label` medio 0,502);
profili 200 × 37 con `elo`, `elo_hard`, `elo_clay` (stesse colonne del run S3).

Tempo per fase: feature rolling 72%, H2H 15%, tornei 6%, target 5%;
tutte le letture e scritture su disco insieme occupano meno di 30 secondi.

### Differenze rispetto al run S3 (il confronto non è del tutto alla pari)
- **Sackmann** letto da disco (`~/tennis_predictor/data/raw/sackmann/tennis_atp`, copia al 30/12/2024) invece che scaricato da GitHub.
- **Bio dei giocatori** da cache HTML (scritta dal run fallito delle 16:32, TTL 72h): in quel run lo step aveva richiesto ~6 minuti.
- **Più dati 2026**: tennis-data 2026 contiene ora 2.150 match (vs 801 nel run S3).
- Quasi tutto il tempo è CPU nel calcolo delle feature rolling, indipendente dal backend.
