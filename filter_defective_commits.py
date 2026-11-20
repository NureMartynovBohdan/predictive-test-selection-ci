import os
import pandas as pd

RTP_DIR = r"./rtp-torrent"

# Берем топ-8 проектов из нашего первичного аудита
TOP_PROJECTS = [
    "apache@sling",
    "square@okhttp",
    "deeplearning4j@deeplearning4j",
    "CloudifySource@cloudify",
    "jOOQ@jOOQ",
    "SonarSource@sonarqube",
    "l0rdn1kk0n@wicket-bootstrap",
    "facebook@buck"
]

df_commits = pd.read_csv(os.path.join(RTP_DIR, "tr_all_built_commits.csv"))
job_to_commit = dict(zip(df_commits['tr_job_id'], df_commits['git_commit_id']))

results = []

for proj in TOP_PROJECTS:
    proj_dir = os.path.join(RTP_DIR, proj)
    main_csv = os.path.join(proj_dir, f"{proj}.csv")
    patches_csv = os.path.join(proj_dir, f"{proj}-patches.csv")
    
    df = pd.read_csv(main_csv, usecols=['travisJobId', 'testName', 'duration', 'failures', 'errors'])
    df['failed'] = (df['failures'] + df['errors']) > 0
    df['commit'] = df['travisJobId'].map(job_to_commit)
    
    df_patches = pd.read_csv(patches_csv, usecols=['sha'])
    valid_shas = set(df_patches['sha'].dropna().unique())
    
    # Оставляем только записи с валидным коммитом и патчем
    df = df[df['commit'].isin(valid_shas)].copy()
    
    # Порядок коммитов во времени определяем по минимальному travisJobId коммита
    commit_order = df.groupby('commit')['travisJobId'].min().sort_values().index.tolist()
    commit_rank = {c: i for i, c in enumerate(commit_order)}
    df['commit_idx'] = df['commit'].map(commit_rank)
    
    # 1. Агрегация на уровне (commit, testName) и поиск внутрикоммитных flaky-тестов
    # Если на одном коммите тест и падал (max=True), и проходил (min=False) в разных джобах -> это flaky на данном коммите
    agg = df.groupby(['commit_idx', 'commit', 'testName'])['failed'].agg(
        any_fail='max',
        all_fail='min',
        runs='count'
    ).reset_index()
    
    agg['intra_commit_flaky'] = agg['any_fail'] & (~agg['all_fail'])
    
    # Настоящее падение на коммите (без внутрикоммитного мигания):
    # Вариант А (строгий): тест упал во всех запусках этого коммита (all_fail)
    # Вариант Б (мягкий): тест упал хотя бы раз (any_fail), но не мигал внутри коммита
    # Они совпадают! Так как если any_fail=True и intra_commit_flaky=False, то all_fail=True.
    agg['commit_failed_strict'] = agg['all_fail']
    
    # 2. Проверка перехода pass -> fail между последовательными коммитами, где запускался этот тест
    agg = agg.sort_values(['testName', 'commit_idx'])
    agg['prev_strict_fail'] = agg.groupby('testName')['commit_failed_strict'].shift(1, fill_value=False)
    agg['prev_any_fail'] = agg.groupby('testName')['any_fail'].shift(1, fill_value=False)
    
    # Метка 1 (строгая): на этом коммите стабильно упал (all_fail), а на прошлом запуске этого теста НЕ падал
    agg['label_strict_transition'] = agg['commit_failed_strict'] & (~agg['prev_strict_fail'])
    
    # Метка 1 (только переход pass->fail по any_fail, без отсева матричных различий внутри коммита)
    agg['label_any_transition'] = agg['any_fail'] & (~agg['prev_any_fail'])
    
    total_commits = len(commit_order)
    raw_fail_commits = agg[agg['any_fail']]['commit'].nunique()
    no_flaky_fail_commits = agg[agg['commit_failed_strict']]['commit'].nunique()
    transition_any_commits = agg[agg['label_any_transition']]['commit'].nunique()
    transition_strict_commits = agg[agg['label_strict_transition']]['commit'].nunique()
    flaky_pairs_cnt = int(agg['intra_commit_flaky'].sum())
    
    results.append({
        'Project': proj,
        'Valid_Commits': total_commits,
        'Raw_Fail_Commits': raw_fail_commits,
        'Transition_Commits(pass->fail)': transition_any_commits,
        'Strict_Defective(no_matrix_flaky+transition)': transition_strict_commits,
        'Matrix_Flaky_Pairs': flaky_pairs_cnt
    })

df_res = pd.DataFrame(results).sort_values(by='Strict_Defective(no_matrix_flaky+transition)', ascending=False)
print(df_res.to_string(index=False))
df_res.to_csv("rtp_filtered_defective_commits.csv", index=False)