import os
import glob
import pandas as pd

# Ищем папки автоматически относительно папки Code или на уровень выше
BASE_DIRS = [".", ".."]

def find_dir(pattern):
    for base in BASE_DIRS:
        matches = glob.glob(os.path.join(base, pattern))
        for m in matches:
            if os.path.isdir(m):
                return m
    return None

RTP_DIR = find_dir("*rtp*torrent*") or r"./rtp-torrent"
CASTOR_DIR = find_dir("*CASTOR*") or r"./CASTOR_replication_package_v2.0"

print(f"Папка RTPTorrent: {RTP_DIR}")
print(f"Папка CASTOR: {CASTOR_DIR}\n")

# 1. Проверяем готовые таблицы из CASTOR (если папка нашлась)
if os.path.exists(CASTOR_DIR):
    print("=== 1. СВОДКА ИЗ CASTOR ===")
    audits_dir = os.path.join(CASTOR_DIR, "outputs", "passing_job_operational", "audits")
    for fname in ["04_mapping_coverage.csv", "07b_operational_project_support.csv"]:
        fpath = os.path.join(audits_dir, fname)
        if os.path.exists(fpath):
            print(f"\n--- {fname} ---")
            print(pd.read_csv(fpath).to_string())

# 2. Считаем статистику по всем 20 проектам RTPTorrent
print("\n=== 2. ПОЛНЫЙ АУДИТ 20 ПРОЕКТОВ RTPTorrent ===")
commits_map_path = os.path.join(RTP_DIR, "tr_all_built_commits.csv")
df_commits = pd.read_csv(commits_map_path)
# Словарь: tr_job_id -> git_commit_id
job_to_commit = dict(zip(df_commits['tr_job_id'], df_commits['git_commit_id']))

summary = []

for item in sorted(os.listdir(RTP_DIR)):
    proj_dir = os.path.join(RTP_DIR, item)
    main_csv = os.path.join(proj_dir, f"{item}.csv")
    patches_csv = os.path.join(proj_dir, f"{item}-patches.csv")
    
    if not os.path.isdir(proj_dir) or not os.path.exists(main_csv):
        continue
        
    df = pd.read_csv(main_csv, usecols=['travisJobId', 'testName', 'failures', 'errors'])
    df['failed'] = (df['failures'] + df['errors']) > 0
    
    # Группируем по задачам (travisJobId)
    job_stats = df.groupby('travisJobId')['failed'].agg(
        total_tests='count',
        failed_tests='sum'
    ).reset_index()
    
    # Привязываем коммит
    job_stats['commit'] = job_stats['travisJobId'].map(job_to_commit)
    
    # Проверяем наличие диффов (измененных файлов) в -patches.csv
    if os.path.exists(patches_csv):
        df_patches = pd.read_csv(patches_csv, usecols=['sha'])
        commits_with_patches = set(df_patches['sha'].dropna().unique())
    else:
        commits_with_patches = set()
        
    job_stats['has_patch'] = job_stats['commit'].isin(commits_with_patches)
    
    # Оставляем только валидные задачи (есть коммит и есть патч/дифф)
    valid_jobs = job_stats[job_stats['commit'].notna() & job_stats['has_patch']]
    
    total_jobs = len(job_stats)
    valid_jobs_cnt = len(valid_jobs)
    passing_valid_jobs = len(valid_jobs[valid_jobs['failed_tests'] == 0])
    failing_valid_jobs = len(valid_jobs[valid_jobs['failed_tests'] > 0])
    
    # Считаем уникальные коммиты среди валидных задач
    valid_commits_cnt = valid_jobs['commit'].nunique()
    failing_commits_cnt = valid_jobs[valid_jobs['failed_tests'] > 0]['commit'].nunique()
    
    summary.append({
        'Project': item,
        'Total_Jobs': total_jobs,
        'Valid_Jobs(w/diff)': valid_jobs_cnt,
        'Passing_Jobs': passing_valid_jobs,
        'Failing_Jobs': failing_valid_jobs,
        'Valid_Commits': valid_commits_cnt,
        'Failing_Commits': failing_commits_cnt
    })

df_sum = pd.DataFrame(summary).sort_values(by='Failing_Commits', ascending=False)
print(df_sum.to_string(index=False))
df_sum.to_csv("rtp_audit_summary.csv", index=False)
print("\nРезультат сохранен в rtp_audit_summary.csv")