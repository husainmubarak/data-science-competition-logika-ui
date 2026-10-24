"""
update_notebook_v7.py
Update notebook description and metadata for v7 10-fold CV pipeline
"""
import json
from pathlib import Path

nb_path = Path('D:/LOGIKA UI/code/DSC0102_JanjiLombaTerakhirTahunIni_Notebook.ipynb')
nb = json.load(open(nb_path, encoding='utf-8'))

# Update markdown description in Cell 0
nb['cells'][0]['source'] = [
    "# DSC LOGIKA UI 2026 \u2014 Advanced Feature Engineering v7 & 10-Fold Stacking Ensemble\n",
    "\n",
    "Notebook ini mengimplementasikan alur pemodelan terstruktur dan reproduktif:\n",
    "1. **Audit & Cutoff Validation:** Menjamin kepatuhan terhadap aturan cut-off (`decision_datetime`).\n",
    "2. **Cross-Domain Feature Engineering (v7):** Ekstraksi sinyal ketepatan waktu pelunasan (`early_days_mean`), e-commerce recency (`days_since_last`), indeks kelaparan kredit (*Credit Hunger Index*), rasio DTI komprehensif, dan *Super Risk Composite*.\n",
    "3. **Multi-Family Diverse Boosting:** LightGBM GBDT, LightGBM Extra Trees, XGBoost Hist, CatBoost Depth 7, dan CatBoost Depth 5.\n",
    "4. **10-Fold Stratified Cross-Validation:** Menurunkan variansi prediksi test hingga 1.41x dan memaksimalkan generalisasi private leaderboard.\n",
    "5. **Stacking & Rank Blending:** Meta-learner logistik dengan skip connections dan optimasi bobot ensemble Dirichlet 10.000 sampel.\n"
]

# Update Cell 1 (imports and config)
nb['cells'][1]['source'] = [
    "from pathlib import Path\n",
    "import json\n",
    "import gc\n",
    "import warnings\n",
    "import joblib\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "from scipy.stats import ks_2samp, rankdata\n",
    "from sklearn.model_selection import StratifiedKFold\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from lightgbm import LGBMClassifier, early_stopping, log_evaluation\n",
    "from xgboost import XGBClassifier\n",
    "from catboost import CatBoostClassifier\n",
    "\n",
    "warnings.filterwarnings('ignore')\n",
    "\n",
    "SEED = 42\n",
    "N_FOLDS = 10\n",
    "DATA_DIR = Path('data-science-competition-logika-ui-2026')\n",
    "OUTPUT_DIR = Path('artifacts')\n",
    "OUTPUT_DIR.mkdir(exist_ok=True)\n",
    "print('Workspace configured. Random SEED:', SEED, '| Folds:', N_FOLDS)\n"
]

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook v7 synced successfully.")
