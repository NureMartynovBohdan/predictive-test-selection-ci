# Predictive Test Selection in CI with Change-Level Conformal Guarantees

Магістерське науково-дослідне дослідження, присвячене **предиктивному вибору та приоритизації регресійних тестів у Continuous Integration (CI)** з математичними гарантіями рівня пропуску дефектів за допомогою методів конформного прогнозування (Conformal Prediction).

---

## Мета проекту (Project Goal)

У великих проектах запуск повного набору тестів на кожен комміт у CI займає занадто багато часу та ресурсів. Мета цього проекту — створити та експериментально підтвердити підхід для **предиктивного вибору тестів (Predictive Test Selection / RTS)**, який:
1. вибирає мінімально необхідний піднабір тестів для перевірки кожного комміту;
2. гарантує заданий рівень надійності (контроль Commit Miss Rate ≤ α) через калібровані конформні оцінки;
3. оптимізує черговість виконання тестів за співвідношенням «ризик збою / тривалість» (Cost-aware Prioritization / APFDc).

---

## Архітектура (Architecture)

### Часовий симулятор — `CIHistorySimulator`

Ключовий архітектурний елемент — **часовий симулятор** (`src/simulator.py`), який виступає **єдиним інтерфейсом** між сирими даними RTPTorrent і всіма іншими компонентами системи.

```
                    ┌──────────────────────┐
                    │  CIHistorySimulator   │
                    │  (temporal data layer)│
                    └──────┬───────────────┘
                           │
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
    DecisionContext   TestHistory    GroundTruth
   (до рішення)    (строго минуле) (після рішення)
           │               │               │
           ▼               ▼               ▼
       features.py     features.py    evaluate.py
           │               │               │
           └───────┬───────┘               │
                   ▼                       │
              models.py                    │
           (LightGBM + ISO)               │
                   │                       │
                   ▼                       │
            conformal.py                   │
         (λ, τ calibration)               │
                   │                       │
                   └───────────────────────┘
```

**Принципи:**
- `get_candidate_tests(commit_sha)` формує попередній перелік тестів-кандидатів **строго до запуску CI** на `commit_sha`: з історичних запусків у межах `inventory_window` та на `parent_sha` з `job_order < as_of_order`, об'єднаних із тестовими класами (`*Test.java`, `Test*.java`, `*IT.java` тощо) зі змінених файлів патчу. Метод **ніколи не читає** `self._runs_by_commit[commit_sha]`.
- `DecisionContext` містить **лише** інформацію, відому ДО запуску тестів (хеш, батьківський хеш, змінені файли, список тестів-кандидатів). **Без міток та фактичної тривалості.**
- `get_test_history(test_name, as_of_order)` повертає тільки запуски з `job_order < as_of_order` — суворе запобігання витоку з майбутнього.
- `reveal_ground_truth(sha)` викликається **тільки після** прийняття рішення селектором (або для тренувальних даних). Якщо кандидат із дотестового інвентарю не має запису в логах CI для `sha`, він коректно позначається як ненаблюданий (`label = None`, `label_status = "unobserved_candidate"`) без помилки `KeyError`.

### Запобігання витоку даних (Anti-Leakage)

- Кожен запуск тесту зберігається як окрема спроба з монотонним `job_order` (proxy = `job_id`).
- Пізній перезапуск коміту A (`job_id=100`) не змінює ознаки чи мітки вже пройденого коміту B (`job_id=2`).
- Фактична тривалість поточного тесту (`actual_duration`) **ніколи** не використовується як ознака — тільки історичні оцінки (`duration_median_20`, `duration_p90_20`).

---

## Дані (Data)

### RTPTorrent
Датасет [RTPTorrent v11](https://doi.org/10.5281/zenodo.4046180) (`rtp-torrent-v11.zip`, ~5 GB) містить логи збірок Travis CI для 20 open-source Java проектів (збірки, джоби, результати тестів, тривалості, патчі).

### Прив'язка Job → SHA
- Файл `tr_all_built_commits.csv` забезпечує зв'язок `tr_job_id → git_commit_id`.
- **Коректна обробка дублікатів**: якщо один `job_id` відповідає декільком SHA, зберігається перший запис із попередженням у логах (замість `dict(zip(...))`, яке мовчки перезаписувало дублікати).

### Батьківський коміт (`parent_source`)
- Батьківський SHA виводиться **в першу чергу** з Git DAG (справжній `C^1`-батько через `git rev-list --parents --all` з bare-репозиторію `rtp-torrent/repo/{project}`).
- Якщо коміт відсутній у bare-репозиторії (або репозиторій недоступний), застосовується **хронологічний запасний варіант** (`chronological_fallback`: попередній коміт за `min(job_id)`), а кореневі коміти отримують `"none"`.
- Джерело батька фіксується в полі `parent_source`: `"git_parent"` | `"chronological_fallback"` | `"none"`.
- **Основний режим експерименту (за замовчуванням):** і основний цикл, і `--rolling` залишають **тільки** рядки з підтвердженим `parent_source == "git_parent"`, жорстко виключаючи `chronological_fallback` та `none` із виведенням кількості виключених рядків. Прапор `--include-fallback-parents` вмикає `chronological_fallback` для окремого аналізу чутливості.

### Розмітка міток (Three-State Label Attribution)
Під час роботи з основною політикою `confirmed_new_fail` використовується тризначна розмітка (`1`, `0`, `None`), де неоднозначні або непідтверджені спостереження виключаються з бінарного навчання та оцінки (замість зведення до `0`):

| Статус (`label_status`) | Мітка (`label`) | Пояснення |
|---|---|---|
| `confirmed_new_fail` | `1` | Тест стабільно падає зараз, на батьку підтверджено `pass` |
| `no_new_fail` | `0` | Тест стабільно проходить зараз (нових регресій немає) |
| `already_failing` | `0` | Тест падав і на батьку, і зараз — не нове падіння |
| `ambiguous_current` | `None` | Змішані результати (`pass` і `fail`) на поточному SHA |
| `parent_mixed` | `None` | Поточний результат за неоднозначного (`mixed`) статусу батька |
| `unknown_parent` | `None` | Поточне падіння за невідомого статусу батька |
| `unobserved_candidate` | `None` | Кандидат із дотестового інвентарю відсутній у логах CI для цього SHA |

> **Правило виключення та обліку кандидатів:** усі записи з `label = None` виключаються з бінарного навчання та оцінювання (`filter_valid_pairs`). Натомість тести з фактичного CI ground truth, яких не було в дотестовому переліку `get_candidate_tests()` (`in_candidate_inventory == False`), **не використовуються** під час навчання моделі (`train_lgbm`) і ізотонічної калібрування (`calibrate_isotonic`), отримують примусовий прогноз `0.0` і не можуть бути обрані селектором (`selected_mask = False`), але **враховуються як невиявлені в повному ground truth при конформному калібруванні та обчисленні CMR** (тобто не зникають із оцінювання).

Для аналізу чутливості також підтримуються бінарні політики `--policy strict_all_fail` та `--policy any_fail`.

### Неоднозначні результати на одному SHA (`ambiguous_outcome_same_sha`)
Поле `ambiguous_outcome_same_sha` (зі зворотним аліасом `mixed_outcomes_same_sha`) фіксує ситуацію, коли один і той самий тест мав одночасно і `pass`, і `fail` у межах одного SHA. Такі випадки отримують `label = None` (`ambiguous_current`) і виключаються зі строгого навчання та оцінки CMR. Оскільки RTPTorrent не містить метаданих середовища виконання чи причин інфраструктурних збоїв, ця ознака трактується як сигнал неоднозначності результату на SHA, а не доказ істинної нестабільності (flaky).

---

## Побудова ознак (Feature Engineering)

Всі ознаки обчислюються **виключно** через API `CIHistorySimulator` з даних строго минулого.

| Група | Ознаки |
|---|---|
| **Історія тесту** | `fail_rate_5/20/50`, `new_failure_rate_50`, `new_failure_support`, `last_fail_dist`, `last_observed_status`, `run_count_total`, `duration_median_20`, `duration_p90_20`, `duration_history_known` |
| **Статус батька** | `parent_status_code` (0=pass, 1=fail, 2=mixed, −1=unknown), `parent_result_known` |
| **Характеристики коміту** | `n_changed_files`, `n_prod_files_changed`, `n_test_files_changed`, `has_java`, `has_xml`, `has_build_file` |
| **Зв'язок тест↔файл** | `test_file_changed`, `same_package_changed`, `same_module_changed`, `path_min_dist`, `path_max_overlap`, `co_fail_rate`, `co_fail_support` |

**Важливо:**
- Для тестів без історії: `duration_median_20 = NaN`, `duration_history_known = 0`. LightGBM нативно обробляє `NaN`.
- CamelCase розбиття при токенізації: `FooBarTest → {foo, bar, test}`.
- `co_fail_rate` завжди супроводжується `co_fail_support` (щоб модель відрізняла «0 з 0» від «0 зі 100»).

---

## Моделювання (Modeling)

### 4-фазовий хронологічний спліт
```
┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│  Train   │→ │ ProbCalib│→ │ ConfCalib│→ │   Test   │
│   50%    │  │   15%    │  │   15%    │  │   20%    │
└──────────┘  └──────────┘  └──────────┘  └──────────┘
  LightGBM     Isotonic       Conformal    Незалежна
  навчання    калібрування   калібрування    оцінка
```

- **ProbCalib** використовується тільки для isotonic regression — ніколи для конформних порогів.
- **ConfCalib** використовується тільки для конформного калібрування λ та τ.

### LightGBM
- Бінарна класифікація з `scale_pos_weight` для дисбалансу класів.
- **Без** `.fillna(0)` перед подачею — LightGBM нативно обробляє пропуски.

---

## Конформне прогнозування (Conformal Prediction)

### Global Conformal Threshold
Для кожного дефектного коміту калібрувальної вибірки: S_c = max(p̂_{c,i} | i ∈ Failed(c)).
Поріг λ = S_{(k)}, де k = ⌊α(n+1)⌋ — гарантує CMR ≤ α при **обмінюваності** (exchangeability).

> **Примітка:** класична конформна гарантія спирається на припущення про обмінюваність спостережень. У часовому потоці CI розподіл даних може дрейфувати, тому стійкість порогів λ і τ додатково перевіряється скользячим оцінюванням (`--rolling`).

### Residual Risk (з конформним калібруванням τ)
- Для кожного дефектного калібрувального коміту знаходиться мінімальний префікс j*, що захоплює хоча б одне реальне падіння.
- Записується R(j*−1) — залишковий ризик перед додаванням цього тесту.
- τ = R_{(k)} де k = ⌊α(n+1)⌋ (конформна поправка з нижнім квантилем).

### Детерміноване розв'язання ничьих
Сортування кандидатів за `(-prob, test_name)` — гарантовано однакове і при калібруванні, і при застосуванні.

### Перевірка порожнього префікса
Перед додаванням першого тесту перевіряється `R(0) < τ`: якщо коміт безпечний, обирається 0 тестів.

---

## Метрики (Metrics)

| Метрика | Опис |
|---|---|
| **CMR** | Change Miss Rate — частка дефектних комітів, де жоден обраний тест не впав (`95%` інтервали: Wilson score CI `cmr_ci_low/high` та moving block bootstrap CI `cmr_boot_ci_low/high`) |
| **TTFF** | Time-To-First-Failure — модельний послідовний час до першого виявленого падіння (у секундах, без моделювання паралельних воркерів) |
| **Conditional APFDc** (`conditional_apfdc` / `apfdc`) | `1 − Σ(C_i − t_i/2) / (m·T)` — cost-cognizant APFD, обчислений **по вибраній підмножині тестів** (`T` = сумарна тривалість лише обраних тестів для даного коміту, без кліпування `[0, 1]`) |
| **PR-AUC / PR-AUC Baseline** (`pr_auc_test`, `pr_auc_baseline_test`) | Площа під PR-кривою та її базовий рівень (частка `label == 1` у `test_df`) |
| **Brier Score (raw / iso / baseline)** (`brier_raw_test`, `brier_score_test`, `brier_baseline_test`) | Brier score сирих імовірностей LightGBM, після ізотонічної калібрування та константного прогнозу за часткою позитивів у `train` |
| **% Tests Selected** | Частка обраних тестів (у межах доступних спостережень CI) |
| **% Duration Saved** | Заощаджений час виконання (у межах доступних спостережень CI) |

> **Важливо:** **TTFF** та **Conditional APFDc** є **умовними** метриками — вони обчислюються виключно на **виявлених** дефектних комітах, а знаменник `T` у `Conditional APFDc` береться по вибраному піднабору тестів (не по повному набору). Їх **завжди** слід інтерпретувати разом із **CMR** та бюджетом (`% Tests Selected` / `% Duration Saved`).

---

## Базові методи порівняння (Baselines)

| Метод | Опис |
|---|---|
| `retest_all` | Запуск усіх тестів (100% бюджет, алфавітний порядок) |
| `top_k%` | Топ k% тестів за ймовірністю (k ∈ {10, 30, 50}) |
| `fixed_0.5` | Фіксований поріг p̂ ≥ 0.5 |
| `global_conformal` | Конформний глобальний поріг λ (сортування за p̂) |
| `global_conformal_cost_aware` | Конформний глобальний поріг λ (пріоритизація за `p̂ / duration_median_20`) |
| `residual_risk` | Залишковий ризик з конформним τ (сортування за p̂) |
| `residual_risk_cost_aware` | Залишковий ризик з конформним τ (пріоритизація за `p̂ / duration_median_20`) |
| `naive_residual` | Залишковий ризик без калібрування (τ = α) |
| `random_matched` | Випадковий вибір з покоммітним бюджетом, що збігається з `global_conformal` |
| `recently_failed` | Тести з недавніми падіннями (`fail_rate_20 > 0`, резерв `fail_rate_50 > 0` або `last_fail_dist ≤ 20`) |
| `threshold_target_recall` | Поріг за цільовим recall на валідації |

---

## Структура репозиторію (Repository Structure)

```
.
├── src/                                  # основний модульний код
│   ├── simulator.py                      # CIHistorySimulator (часовий шар даних)
│   ├── features.py                       # генерація ознак через API симулятора
│   ├── conformal.py                      # конформне калібрування (λ, τ)
│   ├── evaluate.py                       # метрики (CMR + CI, TTFF, Conditional APFDc) + бейзлайни
│   ├── models.py                         # LightGBM, isotonic, 4-way і rolling спліти
│   ├── run_experiment.py                 # головний скрипт запуску
│   └── audit_dataset.py                  # аудит датасету
├── tests/                                # модульні тести
│   └── test_all.py                       # pytest: утечки, candidate_tests, git parents, конформне, метрики
├── outputs/                              # результати експериментів
│   ├── dataset_audit.csv                 # аудит датасету
│   └── runs/                             # унікальні папки кожного прогону (<timestamp>_<tag>_seeds<N>/)
│       └── <run_id>/
│           ├── results_raw.csv           # сирі дані по (project, seed, alpha, method)
│           ├── results_table.csv         # агреговані результати (mean ± std)
│           ├── results_rolling.csv       # ковзні вікна (зі status/skip_reason при --rolling)
│           └── figures/                  # графіки (PDF, PNG)
├── rtp-torrent/                          # дані RTPTorrent (не в git)
├── CASTOR_replication_package_v2.0/      # пакет реплікації CASTOR
├── filter_defective_commits.py           # (legacy) очищення комітів
├── audit_all_projects.py                 # (legacy) аудит проектів
├── inspect_data.py                       # швидка перевірка даних
└── README.md                             # цей файл
```

---

## Як запустити (How to Run)

### 1. Вимоги (Requirements)
- Python 3.10+
- Встановлені пакети:
```bash
pip install numpy pandas scikit-learn lightgbm matplotlib scipy pytest
```

### 2. Запуск тестів
```bash
pytest tests/test_all.py -v
```

### 3. Аудит датасету
```bash
python src/audit_dataset.py
```

### 4. Запуск експерименту
```bash
# Швидкий smoke-тест (1 проект, 1 seed; режим git_parent ONLY)
python src/run_experiment.py --smoke

# Smoke-тест зі скользячими (expanding) вікнами
python src/run_experiment.py --smoke --rolling

# Налаштування кількості ковзних вікон (за замовчуванням 3)
python src/run_experiment.py --seeds 5 --rolling --rolling-splits 3

# Конкретний проект з іншою політикою міток або включенням fallback-батьків (аналіз чутливості)
python src/run_experiment.py --project apache@sling --seeds 5 --policy any_fail --include-fallback-parents

# Повний запуск (5 проектів × 20 seeds × 4 alpha)
python src/run_experiment.py
```

**Основні прапори CLI (`src/run_experiment.py`):**
- `--smoke` — швидкий прогін (`l0rdn1kk0n@wicket-bootstrap`, `seeds=1`)
- `--project <id>` — запуск одного конкретного проєкту (напр., `apache@sling`)
- `--seeds <N>` — кількість випадкових seed-прогонів
- `--rolling` / `--rolling-splits <K>` — скользяче (expanding-window) оцінювання по `K` неперетинним тестовим вікнам (за замовчуванням `K=3`)
- `--policy {confirmed_new_fail,strict_all_fail,any_fail}` — політика формування міток (за замовчуванням `confirmed_new_fail`)
- `--include-fallback-parents` — включати `chronological_fallback` батьків замість суворого режиму `git_parent` (тільки для аналізу чутливості)

Результати автоматично зберігаються в унікальну папку `outputs/runs/<timestamp>_<tag>_seeds<N>/` (наприклад, `outputs/runs/2026-11-23_17-52_all5_seeds5/`):
- `results_raw.csv` — один рядок на `(project, seed, alpha, method)`
- `results_table.csv` — `mean ± std` агреговані по `(project, method, alpha)`, включно з `conditional_apfdc_mean`, `pr_auc_baseline_test_mean`, `brier_score_test_mean`, `brier_raw_test_mean`, `brier_baseline_test_mean`, `cmr_ci_low/high`, `cmr_boot_ci_low/high`
- `results_rolling.csv` — результати по ковзних вікнах (`--rolling`): логує як виконані вікна (`status="ok"`), так і пропущені (`status="skipped"`, `skip_reason`), а також `min_def_conf_nonzero` та `sufficient_for_nonzero_threshold` для кожного `alpha`
- `figures/fig1_cmr_vs_alpha.{pdf,png}` — CMR vs α для всіх методів
- `figures/fig2_budget_vs_alpha.{pdf,png}` — Частка обраних тестів vs α
- `figures/fig3_apfdc_alpha10.{pdf,png}` — Conditional APFDc при α=0.10
- `figures/fig4_reliability_diagram.{pdf,png}` — Діаграма надійності ймовірностей
- `figures/fig5_rolling_cmr.{pdf,png}` — Динаміка CMR по ковзних вікнах (тільки з `--rolling`)

---

## Обмеження (Limitations)

1. **Батьківський коміт** визначається з реального Git DAG (bare-репозиторій, `git rev-list --parents --all`, перший батько `C^1` для merge-комітів). За замовчуванням у навчанні та оцінці залишаються **тільки** рядки з `parent_source == "git_parent"`; `chronological_fallback` та `none` виключаються (вмикається лише через `--include-fallback-parents` для аналізу чутливості).
2. **`ambiguous_outcome_same_sha`** (та аліас `mixed_outcomes_same_sha`) є евристичним сигналом неоднозначності результатів на одному SHA (без метаданих OS/JDK чи інфраструктурних збоїв). Такі пари отримують `label = None` (`ambiguous_current`) та виключаються з бінарної оцінки.
3. **CMR** вимірює частку пропущених дефектів відносно тих, що виявляються повним прогоном наявного тестового набору, а не всіх можливих помилок у коді. Пропущені дефекти, яких немає в тестовому наборі взагалі, не враховуються.
4. **TTFF і Conditional APFDc** оцінюють модельний послідовний час виконання (без моделювання паралельних CI-воркерів) і обчислюються умовно лише на виявлених дефектних комітах (де `T` — тривалість вибраного піднабору тестів).
5. **Конформна гарантія** (CMR ≤ α) спирається на припущення про обмінюваність (exchangeability) спостережень та достатнє число дефектних калібрувальних комітів (`n ≥ ⌈1/α⌉ − 1` для ненульового порогу). У часовому потоці CI розподіл може дрейфувати, тому стійкість порогів додатково перевіряється скользячим оцінюванням (`--rolling`).
6. **Результати є попередніми** до повної мультипроектної валідації на всіх 20 проектах.
7. **`lines_added` / `lines_deleted`** відсутні у RTPTorrent (датасет містить лише імена змінених файлів), тому ці ознаки **виключені** з `FEATURE_COLS`.
8. **Актуальні результати** зберігаються виключно в папках `outputs/runs/<timestamp>_<tag>_seeds<N>/` (зокрема повний прогін `outputs/runs/2026-11-23_17-52_all5_seeds5/`); застарілі агреговані CSV у корені `outputs/` видалено.
