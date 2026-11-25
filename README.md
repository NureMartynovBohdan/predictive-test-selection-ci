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

### Неоднозначні результати, збої та інфраструктурні обмеження RTPTorrent
- Поле `ambiguous_outcome_same_sha` (зі зворотним аліасом `mixed_outcomes_same_sha`) фіксує ситуацію, коли один і той самий тест мав одночасно і `pass`, і `fail` у межах одного SHA. Такі випадки отримують `label = None` (`ambiguous_current`) і виключаються зі строгого навчання та оцінки CMR. Оскільки RTPTorrent не містить метаданих середовища виконання чи причин інфраструктурних збоїв, ця ознака трактується як сигнал неоднозначності результату на SHA, а не доказ істинної нестабільності (flaky).
- Датасет містить лише сирі лічильники JUnit `failures` та `errors` (без текстів винятків чи логів збірки), тому не дозволяє достовірно відокремити регресії коду від build/infrastructure збоїв. За повним аудитом 20 проєктів (`outputs/dataset_audit.csv`), частка проблемних прогонів із `errors > 0` коливається від `0.0%` (`facebook@buck`) і `19.0%` (`graylog2-server`) до `95.3%` (`deeplearning4j`) і `99.7%` (`wicket-bootstrap`). Замість штучної евристики обидва типи трактуються однаково в межах суворої політики `confirmed_new_fail`.

### Відбір проєктів (за результатами `outputs/dataset_audit.csv`)
Повний аудит усіх 20 проєктів RTPTorrent за точними фільтрами експерименту (`git_parent ONLY`, тризначна `confirmed_new_fail`) показав, що лише 4 проєкти мають стабільну придатність і для holdout (`ConfCalib ≥ 5`, `Test ≥ 5`), і для неперетинного ковзного оцінювання:
1. **`apache@sling`** — 260 дефектів (`ConfCalib=30`, `Test=53`, `Conf+Test=83`; rolling `3/3`, сумарно `114` тестових дефектів)
2. **`l0rdn1kk0n@wicket-bootstrap`** — 82 дефекти (`ConfCalib=26`, `Test=11`, `Conf+Test=37`; rolling `3/3`, сумарно `49` тестових дефектів)
3. **`SonarSource@sonarqube`** — 80 дефектів (`ConfCalib=14`, `Test=9`, `Conf+Test=23`; rolling `3/3`, сумарно `30` тестових дефектів)
4. **`deeplearning4j@deeplearning4j`** — 40 дефектів (`ConfCalib=10`, `Test=5`, `Conf+Test=15`; rolling `2/3`, сумарно `15` тестових дефектів)

Проєкти **`square@okhttp`** (`ConfCalib=0`, `Test=0`, rolling `0/3`) та **`CloudifySource@cloudify`** (у holdout `ConfCalib=5`, що менше мінімуму для ненульового порога при $\alpha \le 0.15$, і лише `1/3` rolling-вікон) виключено з `BEST_PROJECTS`. Жоден із 20 проєктів RTPTorrent у хронологічному 50/15/15/20 спліті не досягає 150 дефектних змін у `ConfCalib + Test`, тому в результатах додатково звітуються об'єднані rolling-вікна (`results_rolling_pooled.csv`), точний ДИ Клоппера–Пирсона та одностороння 95% верхня межа $U_{95}$.

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

- **ProbCalib** використовується для порівняння та вибору калібратора ймовірностей (**Isotonic Regression** vs **Platt / Sigmoid Scaling**) за допомогою детермінованої крос-валідації по комітах усередині самої `ProbCalib` (без контакту з `ConfCalib` або `Test`), а також можна примусово задати режим `--calibration {auto,isotonic,sigmoid}`.
- **ConfCalib** використовується **виключно** для конформного калібрування λ та τ.

### LightGBM
- Бінарна класифікація з `scale_pos_weight` для сильного дисбалансу класів (тому сирі імовірності зміщені вгору й дають високий `brier_raw_test` до калібрування).
- **Без** `.fillna(0)` перед подачею — LightGBM нативно обробляє пропуски.

---

## Конформне прогнозування (Conformal Prediction)

### Global Conformal Threshold (основний метод)
Для кожного дефектного коміту калібрувальної вибірки: $S_c = \max(\hat{p}_{c,i} \mid i \in \text{Failed}(c))$.
Поріг $\lambda = S_{(k)}$, де $k = \lfloor \alpha (n+1) \rfloor$ — **єдиний метод із формальною скінченновибірковою маргінальною гарантією** $\text{CMR} \le \alpha$ за умови **обмінюваності** (exchangeability) та достатньої кількості дефектних калібрувальних комітів ($n \ge \lceil 1/\alpha \rceil - 1$ для ненульового порога; якщо $n$ замало, логується `calibration_status = "insufficient_for_nonzero_threshold"` і спрацьовує безпечний fallback запуску повного інвентарю).

> **Примітка:** класична конформна гарантія спирається на припущення про обмінюваність спостережень. У часовому потоці CI розподіл даних може дрейфувати, тому стійкість порогів $\lambda$ і $\tau$ додатково перевіряється ковзним оцінюванням (`--rolling`, із підрахунком зваженого `pooled_cmr = sum(M_w) / sum(N_w)` у `results_rolling_pooled.csv`).

### Residual Risk з каліброваним τ (допоміжна евристика)
- Для кожного дефектного калібрувального коміту знаходиться мінімальний префікс $j^*$, що захоплює хоча б одне реальне падіння, та фіксується $R(j^*-1) = 1 - \prod_{i \ge j^*}(1-\hat{p}_{(i)})$.
- $\tau = R_{(k)}$, де $k = \lfloor \alpha(n+1) \rfloor$ (з перевіркою порожнього префікса $R(0) < \tau$).
- Оскільки формула добутку припускає умовну незалежність збоїв між тестами, `residual_risk` класифікується як **евристичний скоринг із конформним калібруванням $\tau$** (`guarantee_type = "heuristic_calibrated_tau"`), а не сувора безмодельна гарантія.

---

## Метрики (Metrics)

| Метрика | Опис |
|---|---|
| **CMR** | Change Miss Rate — частка дефектних комітів, де жоден обраний тест не впав. Для малого числа дефектів обов'язково наводяться **точний біноміальний ДИ Клоппера–Пирсона** (`cmr_exact_ci_low/high`), **одностороння 95% верхня межа** `cmr_upper_95` (при 0 пропусках $1 - 0.05^{1/n}$), ДИ Вілсона та block bootstrap CI |
| **TTFF (`mean`, `median`, `p90`)** | Time-To-First-Failure (`ttff`, `ttff_median`, `ttff_p90`) — модельний послідовний час до першого виявленого падіння у секундах (без моделювання паралельних воркерів) |
| **Conditional APFDc** (`conditional_apfdc` / `apfdc`) | $1 - \sum(C_i - t_i/2) / (m \cdot T)$ — cost-cognizant APFD **по вибраній підмножині тестів** на виявлених дефектних комітах ($T$ = тривалість обраних тестів, без кліпування `[0, 1]`) |
| **PR-AUC / PR-AUC Baseline** (`cand`) | Площа під PR-кривою та її базовий рівень, обчислені **строго на кандидатах** `in_candidate_inventory == True` |
| **Brier Score & BSS** (`cand`) | Brier score для `raw`, `isotonic`, `sigmoid` і константного baseline + **Brier Skill Score** ($\text{BSS} = 1 - \text{BS}_{\text{model}} / \text{BS}_{\text{baseline}}$), строго на `in_candidate_inventory == True` |
| **Budget & Savings** | `% Tests Selected` і `% Duration Saved` по кожному проєкту окремо, а також макро-середнє та **зважене за сумарною тривалістю CI** у `savings_summary.csv` |
| **Розбір причин пропусків** (`miss_analysis.csv`) | Декомпозиція кожного пропущеного дефектного коміту: `missing_from_inventory`, `empty_prefix`, `no_prior_history_low_score`, `score_below_cutoff` |

> **Важливо:** **TTFF** та **Conditional APFDc** є **умовними** метриками — вони обчислюються виключно на **виявлених** дефектних комітах. Їх **завжди** слід інтерпретувати разом із **CMR** (`cmr_exact_ci_high` / `cmr_upper_95`) та бюджетом (`% Duration Saved`).

---

## Базові методи порівняння (Baselines)

| Метод | Опис |
|---|---|
| `retest_all` | Запуск усіх тестів повного набору (100% бюджет, алфавітний порядок) |
| `qtf` | Quickest Test First — повний набір, відсортований за зростанням історичної тривалості `duration_median_20` |
| `latest_fail_cc` | Повний набір, відсортований за нещодавністю збоїв (`last_fail_dist`) та зв'язністю змін коду (`test_file_changed`, `same_package_changed`, `path_max_overlap`, `co_fail_rate`) |
| `global_conformal` / `*_cost_aware` / `*_qtf` | Основний конформний поріг $\lambda$ (порядки за $\hat{p}$, $\hat{p}/\text{dur}$ або QTF) |
| `residual_risk` / `*_cost_aware` | Евристичний залишковий ризик із каліброваним $\tau$ |
| `naive_residual` | Залишковий ризик без калібрування ($\tau = \alpha$) |
| `fixed_0.5` | Фіксований поріг $\hat{p} \ge 0.5$ |
| `top_k%` | Топ k% кандидатів за ймовірністю (k ∈ {10, 30, 50}) |
| `random_matched` | Випадковий вибір кандидатів з покомітним бюджетом `global_conformal` |
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
│   ├── evaluate.py                       # метрики (CMR + exact/Wilson/boot CI, TTFF mean/med/p90, APFDc) + бейзлайни (QTF, LatestFail+CC)
│   ├── models.py                         # LightGBM, Isotonic vs Platt scaling, BSS, 4-way і rolling спліти
│   ├── run_experiment.py                 # головний скрипт запуску
│   └── audit_dataset.py                  # аудит датасету
├── tests/                                # модульні тести
│   └── test_all.py                       # pytest: утечки, candidate_tests, git parents, калібрування, конформне, метрики
├── outputs/                              # результати експериментів
│   ├── dataset_audit.csv                 # аудит датасету (всі 20 проєктів)
│   └── runs/                             # унікальні папки кожного прогону (<timestamp>_<tag>_seeds<N>/)
│       └── <run_id>/
│           ├── results_raw.csv           # сирі дані по (project, seed, alpha, method)
│           ├── results_table.csv         # агреговані результати (mean ± std)
│           ├── results_rolling.csv       # ковзні вікна (зі status/skip_reason при --rolling)
│           ├── results_rolling_pooled.csv# pooled rolling CMR з точним ДИ
│           ├── calibration_summary.csv   # Isotonic vs Platt/Sigmoid, Brier, BSS
│           ├── savings_summary.csv       # покомпонентна та зважена за часом економія
│           ├── miss_analysis.csv         # розбір причин пропусків
│           ├── run_manifest.json         # маніфест відтворюваності
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

# Повний запуск (4 відібраних проєкти × 20 seeds × 4 alpha)
python src/run_experiment.py --rolling
```

**Основні прапори CLI (`src/run_experiment.py`):**
- `--smoke` — швидкий прогін (`l0rdn1kk0n@wicket-bootstrap`, `seeds=1`)
- `--project <id>` — запуск одного конкретного проєкту (напр., `apache@sling`)
- `--seeds <N>` — кількість випадкових seed-прогонів
- `--rolling` / `--rolling-splits <K>` — скользяче (expanding-window) оцінювання по `K` неперетинним тестовим вікнам (за замовчуванням `K=3`)
- `--calibration {auto,isotonic,sigmoid}` — спосіб калібрування ймовірностей (`auto` обирає між Isotonic і Platt за CV на `ProbCalib`)
- `--policy {confirmed_new_fail,strict_all_fail,any_fail}` — політика формування міток (за замовчуванням `confirmed_new_fail`)
- `--include-fallback-parents` — включати `chronological_fallback` батьків замість суворого режиму `git_parent` (тільки для аналізу чутливості)
- `--strict-min-defective <N>` — обов'язкова мінімальна межа дефектних комітів у `ConfCalib + Test`

Результати автоматично зберігаються в унікальну папку `outputs/runs/<timestamp>_<tag>_seeds<N>/`:
- `results_raw.csv` — один рядок на `(project, seed, alpha, method)` (`protocol="holdout_50_15_15_20"`)
- `results_table.csv` — `mean ± std` агреговані по `(project, method, alpha)`, включно з `conditional_apfdc`, `ttff_median/p90`, `pr_auc_baseline_test`, `brier_*`, `bss_*`, `cmr_exact_ci_*`, `cmr_upper_95`, `cmr_ci_*`, `cmr_boot_ci_*`
- `results_rolling.csv` — результати по ковзних вікнах (`--rolling`): логує виконані вікна (`status="ok"`) і пропущені (`status="skipped"`, `skip_reason`), включно з `n_missed_test` і `sufficient_for_nonzero_threshold`
- `results_rolling_pooled.csv` — зважений по комітах підсумок по неперетинних rolling-вікнах (`pooled_cmr = sum(M_w) / sum(N_w)` + Clopper-Pearson і Wilson CI)
- `calibration_summary.csv` — порівняння калібрування (Raw, Isotonic, Platt/Sigmoid, Baseline + BSS) строго на кандидатах
- `savings_summary.csv` — економія по кожному проєкту окремо, макро-середнє та зважений за тривалістю CI підсумок
- `miss_analysis.csv` — покомітний розбір причин кожного пропуску (`global_conformal`, `residual_risk`)
- `run_manifest.json` — маніфест відтворюваності (`run_id`, `git_commit`, `config_hash`, конфігурація сплітів)
- `figures/fig1_cmr_vs_alpha.{pdf,png}` — CMR vs α
- `figures/fig2_budget_vs_alpha.{pdf,png}` — Частка обраних тестів vs α
- `figures/fig3_apfdc_alpha10.{pdf,png}` — Conditional APFDc при α=0.10 (включно з QTF і LatestFail+CC)
- `figures/fig4_reliability_diagram.{pdf,png}` і `fig4_reliability_<project>.png` — Діаграми надійності (Isotonic vs Platt/Sigmoid)
- `figures/fig5_rolling_cmr.{pdf,png}` — Динаміка CMR по ковзних вікнах (`--rolling`)
- `figures/fig6_duration_saved_vs_alpha.{pdf,png}` — Економія часу виконання CI vs α по кожному проєкту

---

## Обмеження (Limitations)

1. **Батьківський коміт** визначається з реального Git DAG (bare-репозиторій, `git rev-list --parents --all`, перший батько `C^1` для merge-комітів). За замовчуванням у навчанні та оцінці залишаються **тільки** рядки з `parent_source == "git_parent"`; `chronological_fallback` та `none` виключаються (вмикається лише через `--include-fallback-parents` для аналізу чутливості).
2. **Нерозрізнюваність flaky / infra / build збоїв:** `ambiguous_outcome_same_sha` фіксує лише одночасні `pass` і `fail` на одному SHA, а поля `failures` vs `errors` у RTPTorrent не мають логів винятків. Такі неоднозначні SHA отримують `label = None` та виключаються.
3. **Обсяг вибірки (150-defect benchmark):** за повним аудитом `outputs/dataset_audit.csv`, жоден із 20 проєктів RTPTorrent у строгому режимі не має ≥ 150 дефектних комітів у одиночному `ConfCalib + Test`. Відібрано 4 найбільш наповнені проєкти, а статистична невизначеність кількісно показана через точні інтервали Клоппера–Пирсона, $U_{95}$ та об'єднану rolling-оцінку (`results_rolling_pooled.csv`).
4. **TTFF і Conditional APFDc** оцінюють модельний послідовний час виконання (без моделювання паралельних CI-воркерів) і обчислюються умовно лише на виявлених дефектних комітах.
5. **Конформна гарантія** ($\text{CMR} \le \alpha$) стосується насамперед `global_conformal`, спирається на обмінюваність та потребує $n \ge \lceil 1/\alpha \rceil - 1$ дефектних комітів для ненульового порога. Для адаптивного `residual_risk` калібрований поріг $\tau$ є евристичним і перевіряється емпірично на `--rolling`.
6. **`lines_added` / `lines_deleted`** відсутні у RTPTorrent (датасет містить лише імена змінених файлів), тому ці ознаки **виключені** з `FEATURE_COLS`.
7. **Результати кожного запуску** ізольовано зберігаються у `outputs/runs/<run_id>/` з повним `run_manifest.json` для відтворюваності.
