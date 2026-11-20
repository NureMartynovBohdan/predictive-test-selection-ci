import os
import pandas as pd

# Укажи здесь свои пути к распакованным папкам
RTP_DIR = r"./rtp-torrent" 
CASTOR_DIR = r"./CASTOR"

print("=== 1. АУДИТ ПРОЕКТОВ ИЗ CASTOR ===")
castor_audits = os.path.join(CASTOR_DIR, "outputs", "passing_job_operational", "audits")

for fname in ["04_mapping_coverage.csv", "07b_operational_project_support.csv"]:
    fpath = os.path.join(castor_audits, fname)
    if os.path.exists(fpath):
        print(f"\n--- Файл: {fname} ---")
        df = pd.read_csv(fpath)
        print(df.to_string())

print("\n=== 2. СТРУКТУРА ТАБЛИЦЫ КАНДИДАТОВ CASTOR ===")
cand_path = os.path.join(castor_audits, "08_all_job_test_candidates.csv.gz")
if os.path.exists(cand_path):
    df_cand = pd.read_csv(cand_path, nrows=5)
    print("Колонки 08_all_job_test_candidates:", list(df_cand.columns))
    print(df_cand.head(2).to_string())

print("\n=== 3. СТРУКТУРА ИСХОДНИКОВ RTPTorrent (на примере okhttp) ===")
sample_proj = "square@okhttp"
for suffix in [".csv", "-patches.csv", "-offenders.csv"]:
    fpath = os.path.join(RTP_DIR, sample_proj, f"{sample_proj}{suffix}")
    if os.path.exists(fpath):
        df_rtp = pd.read_csv(fpath, nrows=3)
        print(f"\nКолонки {sample_proj}{suffix}:", list(df_rtp.columns))
        print(df_rtp.head(2).to_string())

commits_path = os.path.join(RTP_DIR, "tr_all_built_commits.csv")
if os.path.exists(commits_path):
    df_comm = pd.read_csv(commits_path, nrows=3)
    print("\nКолонки tr_all_built_commits.csv:", list(df_comm.columns))
    print(df_comm.head(2).to_string())