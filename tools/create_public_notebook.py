from pathlib import Path
import copy
import json


ROOT = Path("/Users/kleberabreu/Desktop/tcc/tcc-copy-note")
DADOS = ROOT / "Dados"

NB_01 = DADOS / "01_coleta_tratamento_feature_engi.ipynb"
NB_02 = DADOS / "02_Modelo_de_Ocupação.ipynb"
NB_OUT = DADOS / "04_Reproducao_Publica_TCC.ipynb"


def load_nb(path: Path):
    return json.loads(path.read_text())


def clone(cell):
    cell = copy.deepcopy(cell)
    if cell["cell_type"] == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def lines(text: str):
    return [line + "\n" for line in text.splitlines()]


def make_markdown(text: str, idx: int):
    return {
        "cell_type": "markdown",
        "id": f"md-{idx:03d}",
        "metadata": {},
        "source": lines(text),
    }


def make_code(text: str, idx: int):
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": f"code-{idx:03d}",
        "metadata": {},
        "outputs": [],
        "source": lines(text),
    }


def cell_source(nb, idx):
    return "".join(nb["cells"][idx]["source"])


nb01 = load_nb(NB_01)
nb02 = load_nb(NB_02)

cell_idx = 0
cells = []


def add_md(text: str):
    global cell_idx
    cells.append(make_markdown(text, cell_idx))
    cell_idx += 1


def add_code(text: str):
    global cell_idx
    cells.append(make_code(text, cell_idx))
    cell_idx += 1


imports_01 = "".join(nb01["cells"][1]["source"][4:]) + "\n\nimport statsmodels.formula.api as smf\nfrom sklearn.metrics import mean_absolute_error, mean_squared_error\n"

load_listings_code = "\n\n".join(
    [
        cell_source(nb01, 3),
        cell_source(nb01, 4),
        cell_source(nb01, 6),
    ]
)

calendar_target_code = "\n\n".join(
    [
        cell_source(nb01, 8),
        cell_source(nb01, 10),
        cell_source(nb01, 11),
    ]
)

merge_and_treat_code = "\n\n".join(
    [
        cell_source(nb01, 13),
        "variaveis_checagem = [\n"
        "    'price',\n"
        "    'review_scores_rating',\n"
        "    'review_scores_accuracy',\n"
        "    'review_scores_cleanliness',\n"
        "    'review_scores_checkin',\n"
        "    'review_scores_communication',\n"
        "    'review_scores_location',\n"
        "    'review_scores_value',\n"
        "    'number_of_reviews',\n"
        "    'reviews_per_month',\n"
        "    'host_total_listings_count',\n"
        "    'total_ocupacy_3',\n"
        "    'ocupacy_m+1',\n"
        "    'ocupacy_m+2',\n"
        "    'ocupacy_m+3',\n"
        "]",
        cell_source(nb01, 15),
        cell_source(nb01, 16),
        cell_source(nb01, 18),
    ]
)

pca_code = "\n\n".join(
    [
        cell_source(nb01, 26).replace(
            '# Resultados a exibir\n{\n    "bartlett_statistic": bartlett_stat,\n    "bartlett_p_value": bartlett_p,\n    "kmo_score": kmo_model,\n    "pca_pipeline_path": "pipeline_quality_score.pkl",\n    "correlacao_imagem": "correlacao_subnotas.png",\n    "Variância explicada": pca.explained_variance_ratio_[0]\n}\n',
            'print({\n    "bartlett_statistic": float(bartlett_stat),\n    "bartlett_p_value": float(bartlett_p),\n    "kmo_score": float(kmo_model),\n    "pca_pipeline_path": "pipeline_quality_score.pkl",\n    "correlacao_imagem": "correlacao_subnotas.png",\n    "variancia_explicada": float(pca.explained_variance_ratio_[0])\n})\n',
        ),
        cell_source(nb01, 29),
    ]
)

model_load_code = "\n\n".join([cell_source(nb02, 1), cell_source(nb02, 2)])
full_model_code = cell_source(nb02, 3)
reduced_model_code = cell_source(nb02, 5)
diagnostics_code = cell_source(nb02, 6)
export_model_code = cell_source(nb02, 8)
mab_classes_code = cell_source(nb02, 11)
stationary_sim_code = cell_source(nb02, 12)
multi_station_code = cell_source(nb02, 15)

tree_compare_code = cell_source(nb02, 17).replace("n_jobs=-1", "n_jobs=1")
tree_compare_code = tree_compare_code.replace(
    "X = df[features_modelo_ml].copy()",
    "X = df[features_modelo_ml].copy()\nX = X.fillna(X.median(numeric_only=True))",
)

add_md(
    "# TCC – Notebook Público de Reprodução\n\n"
    "Este notebook reproduz a análise final do trabalho **“Precificação dinâmica de estúdios por temporada com algoritmos multi-armed bandits”**.\n\n"
    "Objetivos deste material:\n"
    "- reconstruir a base analítica a partir dos arquivos brutos;\n"
    "- reproduzir as tabelas e figuras centrais do TCC;\n"
    "- treinar novamente o modelo final de demanda;\n"
    "- reproduzir as simulações MAB nos cenários estacionário e multi-estação.\n\n"
    "O foco aqui é **reprodutibilidade pública**. Por isso, foram mantidas apenas as células necessárias para seguir a lógica do trabalho final."
)

add_code(
    "import importlib\n"
    "from pathlib import Path\n"
    "import json\n"
    "import warnings\n\n"
    "dependencias = [\n"
    "    'pandas', 'numpy', 'seaborn', 'matplotlib', 'tqdm', 'factor_analyzer',\n"
    "    'joblib', 'statsmodels', 'sklearn'\n"
    "]\n\n"
    "faltantes = []\n"
    "for dependencia in dependencias:\n"
    "    try:\n"
    "        importlib.import_module(dependencia)\n"
    "    except ModuleNotFoundError:\n"
    "        faltantes.append(dependencia)\n\n"
    "if faltantes:\n"
    "    raise ModuleNotFoundError('Dependências ausentes: ' + ', '.join(faltantes))\n\n"
    "BASE_DIR = Path.cwd()\n"
    "warnings.filterwarnings('ignore')\n"
    "print(f'Diretório de execução: {BASE_DIR}')"
)

add_code(imports_01)

add_md(
    "## 1. Ingestão e filtro da amostra\n\n"
    "Primeiro são lidos os arquivos `listings` dos quatro snapshots trimestrais. Em seguida aplica-se o recorte amostral do trabalho: estúdios/apartamentos compactos, `Entire home/apt`, até 3 hóspedes, no máximo 1 quarto, até 90 noites, `superhost` e presença nos quatro períodos."
)
add_code(load_listings_code)

add_md(
    "## 2. Construção da variável-alvo\n\n"
    "A ocupação trimestral prospectiva é derivada dos arquivos `calendar`, usando as janelas `m+1`, `m+2` e `m+3` a partir de cada snapshot."
)
add_code(calendar_target_code)

add_md(
    "## 3. Tratamento e engenharia de variáveis\n\n"
    "Aqui são feitos o merge das bases, o tratamento de extremos por winsorização, transformações logarítmicas e a criação das variáveis de contexto competitivo e sazonalidade."
)
add_code(merge_and_treat_code)

add_md(
    "## 4. Sumários descritivos alinhados ao TCC\n\n"
    "As tabelas abaixo reproduzem os principais números descritivos discutidos no texto final do trabalho."
)
add_code(
    "estatisticas_ocupacao = pd.DataFrame({\n"
    "    'metrica': ['media_dias', 'mediana_dias', 'desvio_padrao_dias', 'ocupacao_zero_pct'],\n"
    "    'valor': [\n"
    "        df_feature_buinding['total_ocupacy_3'].mean(),\n"
    "        df_feature_buinding['total_ocupacy_3'].median(),\n"
    "        df_feature_buinding['total_ocupacy_3'].std(),\n"
    "        (df_feature_buinding['total_ocupacy_3'].eq(0).mean() * 100),\n"
    "    ]\n"
    "})\n\n"
    "bairros_artigo = ['copacabana', 'ipanema', 'leblon', 'centro', 'jacarepaguá', 'barra_da_tijuca', 'botafogo', 'flamengo']\n"
    "tabela_2 = (\n"
    "    df_feature_buinding.groupby('neighbourhood_cleansed')\n"
    "    .agg(\n"
    "        Obs=('listing_id', 'size'),\n"
    "        Preco_mediano=('price', 'median'),\n"
    "        Ocupacao_media=('total_ocupacy_3', 'mean')\n"
    "    )\n"
    ")\n"
    "tabela_2 = tabela_2.loc[[b for b in bairros_artigo if b in tabela_2.index]].copy()\n"
    "outros = df_feature_buinding[~df_feature_buinding['neighbourhood_cleansed'].isin(bairros_artigo)]\n"
    "tabela_2.loc['outros_(24_bairros)'] = {\n"
    "    'Obs': len(outros),\n"
    "    'Preco_mediano': np.nan,\n"
    "    'Ocupacao_media': np.nan,\n"
    "}\n"
    "tabela_2['Pct_amostra'] = tabela_2['Obs'] / len(df_feature_buinding) * 100\n"
    "tabela_2.index = [\n"
    "    idx.replace('_', ' ').title().replace('Da Tijuca', 'da Tijuca').replace('Outros (24 Bairros)', 'Outros (24 bairros)')\n"
    "    for idx in tabela_2.index\n"
    "]\n"
    "tabela_2 = tabela_2[['Obs', 'Pct_amostra', 'Preco_mediano', 'Ocupacao_media']].round(1)\n\n"
    "tabela_3 = (\n"
    "    df_feature_buinding.groupby('estacao')['total_ocupacy_3']\n"
    "    .agg(['mean', 'median', 'std', 'count'])\n"
    "    .rename(columns={'mean': 'media_dias', 'median': 'mediana_dias', 'std': 'desvio_padrao', 'count': 'obs'})\n"
    "    .round(1)\n"
    ")\n\n"
    "print('Estatísticas da variável-alvo:')\n"
    "display(estatisticas_ocupacao.round(2))\n"
    "print('\\nTabela 2 — distribuição geográfica da amostra:')\n"
    "display(tabela_2)\n"
    "print('\\nTabela 3 — ocupação média por estação:')\n"
    "display(tabela_3)"
)

add_md("## 5. Figura 1 — Histogramas antes e depois do tratamento")
add_code(
    "pares = [\n"
    "    {\n"
    "        'label': 'A',\n"
    "        'antes_col': 'price',\n"
    "        'depois_col': 'log_price',\n"
    "        'titulo_antes': 'Preço (original)',\n"
    "        'titulo_depois': 'log(1 + preço) após winsorização',\n"
    "    },\n"
    "    {\n"
    "        'label': 'B',\n"
    "        'antes_col': 'number_of_reviews',\n"
    "        'depois_col': 'log_number_of_reviews',\n"
    "        'titulo_antes': 'Número de avaliações (original)',\n"
    "        'titulo_depois': 'log(1 + nº avaliações)',\n"
    "    },\n"
    "    {\n"
    "        'label': 'C',\n"
    "        'antes_col': 'reviews_per_month',\n"
    "        'depois_col': 'log_reviews_per_month',\n"
    "        'titulo_antes': 'Avaliações por mês (original)',\n"
    "        'titulo_depois': 'log(1 + aval./mês)',\n"
    "    },\n"
    "]\n\n"
    "fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 10))\n"
    "fig.subplots_adjust(hspace=0.45, wspace=0.3)\n\n"
    "for i, par in enumerate(pares):\n"
    "    for j, (col, titulo) in enumerate([\n"
    "        (par['antes_col'], par['titulo_antes']),\n"
    "        (par['depois_col'], par['titulo_depois']),\n"
    "    ]):\n"
    "        ax = axes[i][j]\n"
    "        df_src = df_merged if j == 0 else df_feature_buinding\n"
    "        dados = df_src[col].dropna()\n"
    "        ax.hist(dados, bins=50, color='steelblue', edgecolor='white', linewidth=0.4)\n"
    "        ax.axvline(dados.mean(), color='red', linestyle='--', linewidth=1.5, label='Média')\n"
    "        ax.axvline(dados.median(), color='black', linestyle='-', linewidth=1.5, label='Mediana')\n"
    "        ax.set_title(f\"({par['label']}) {titulo}\", fontsize=10, fontweight='normal', pad=6)\n"
    "        ax.set_xlabel(col, fontsize=8)\n"
    "        ax.set_ylabel('Frequência', fontsize=8)\n"
    "        ax.tick_params(labelsize=8)\n"
    "        ax.grid(False)\n"
    "        if i == 0 and j == 0:\n"
    "            ax.legend(fontsize=8)\n\n"
    "axes[0][0].annotate('Antes do tratamento', xy=(0.5, 1.12), xycoords='axes fraction', ha='center', fontsize=11, fontweight='bold')\n"
    "axes[0][1].annotate('Após winsorização e transformação logarítmica', xy=(0.5, 1.12), xycoords='axes fraction', ha='center', fontsize=11, fontweight='bold')\n"
    "plt.savefig('Histogramas Antes e depois.png', dpi=200, bbox_inches='tight', facecolor='white')\n"
    "plt.savefig('figura1_histogramas_antes_depois.png', dpi=200, bbox_inches='tight', facecolor='white')\n"
    "plt.show()\n"
    "print('Figuras salvas: Histogramas Antes e depois.png e figura1_histogramas_antes_depois.png')"
)

add_md("## 6. PCA para consolidação dos indicadores de qualidade")
add_code(pca_code)

add_code(
    "tabela_4 = pd.DataFrame({\n"
    "    'tipo_nota': sub_scores,\n"
    "    'loading_pc1': pca.components_[0]\n"
    "})\n"
    "tabela_4['loading_abs'] = tabela_4['loading_pc1'].abs()\n"
    "tabela_4 = tabela_4.sort_values('loading_abs', ascending=False)\n"
    "print('Tabela 4 — cargas fatoriais do primeiro componente principal:')\n"
    "display(tabela_4[['tipo_nota', 'loading_pc1']].round(3))"
)

add_md(
    "## 7. Seleção de variáveis e geração da base final\n\n"
    "A base `base_final_para_modelagem.csv` será a entrada da modelagem de demanda e das simulações."
)
add_code(
    "spearman_target = (\n"
    "    dt_pca[[\n"
    "        'log_price', 'discount_vs_median', 'quality_score', 'review_scores_location',\n"
    "        'log_number_of_reviews', 'bairro_mean_ocup', 'log_host_total_listings_count',\n"
    "        'accommodates', 'total_ocupacy_3'\n"
    "    ]]\n"
    "    .corr(method='spearman')['total_ocupacy_3']\n"
    "    .drop('total_ocupacy_3')\n"
    "    .sort_values(ascending=False)\n"
    ")\n"
    "tabela_5 = pd.DataFrame({\n"
    "    'variavel': [\n"
    "        'log_price', 'review_scores_location', 'log_number_of_reviews', 'bairro_mean_ocup',\n"
    "        'estacao_verao', 'estacao_outono', 'estacao_primavera'\n"
    "    ],\n"
    "    'rho_spearman': [\n"
    "        spearman_target.get('log_price', np.nan),\n"
    "        spearman_target.get('review_scores_location', np.nan),\n"
    "        spearman_target.get('log_number_of_reviews', np.nan),\n"
    "        spearman_target.get('bairro_mean_ocup', np.nan),\n"
    "        np.nan, np.nan, np.nan,\n"
    "    ],\n"
    "    'papel': [\n"
    "        'Alavanca do MAB — variável de decisão',\n"
    "        'Localização percebida pelo hóspede',\n"
    "        'Popularidade e visibilidade',\n"
    "        'Proxy de demanda local do bairro',\n"
    "        'Sazonalidade — verão vs. inverno (ref.)',\n"
    "        'Sazonalidade — outono vs. inverno (ref.)',\n"
    "        'Sazonalidade — primavera vs. inverno (ref.)',\n"
    "    ]\n"
    "})\n"
    "display(tabela_5.round(3))\n"
    "print('Base final salva em base_final_para_modelagem.csv')"
)

add_md("## 8. Preparação da modelagem de demanda")
add_code(model_load_code)

add_md("## 9. Modelo completo para diagnóstico de significância")
add_code(full_model_code)

add_md("## 10. Modelo final de demanda")
add_code(reduced_model_code)

add_code(
    "tabela_6 = pd.DataFrame({\n"
    "    'variavel': ols_red.params.index,\n"
    "    'coeficiente': ols_red.params.values,\n"
    "    'erro_padrao': ols_red.bse.values,\n"
    "    'p_valor': ols_red.pvalues.values,\n"
    "}).round(4)\n"
    "print('Tabela 6 — coeficientes do modelo OLS final:')\n"
    "display(tabela_6)"
)

add_md("## 11. Comparação com modelos não lineares")
add_code(tree_compare_code)

add_code(
    "tabela_7 = pd.DataFrame([\n"
    "    {\n"
    "        'Modelo': 'OLS (7 features)',\n"
    "        'R2_in_sample': ols_red.rsquared,\n"
    "        'R2_cross_val': ols_red.rsquared,\n"
        "        'MAE_in_sample_dias': mae_ols * 91,\n"
        "        'MAE_cv_dias': mae_ols * 91,\n"
    "        'Diagnostico': 'Sem overfitting',\n"
    "    },\n"
    "    {\n"
    "        'Modelo': 'Random Forest',\n"
    "        'R2_in_sample': resultados['Random Forest']['R²'],\n"
    "        'R2_cross_val': resultados['Random Forest']['R²_cv'],\n"
    "        'MAE_in_sample_dias': resultados['Random Forest']['MAE_dias'],\n"
    "        'MAE_cv_dias': resultados['Random Forest']['MAE_cv'] * 91,\n"
    "        'Diagnostico': 'Overfitting severo',\n"
    "    },\n"
    "    {\n"
    "        'Modelo': 'Gradient Boosting',\n"
    "        'R2_in_sample': resultados['Gradient Boosting']['R²'],\n"
    "        'R2_cross_val': resultados['Gradient Boosting']['R²_cv'],\n"
    "        'MAE_in_sample_dias': resultados['Gradient Boosting']['MAE_dias'],\n"
    "        'MAE_cv_dias': resultados['Gradient Boosting']['MAE_cv'] * 91,\n"
    "        'Diagnostico': 'Overfitting severo',\n"
    "    },\n"
    "]).round(3)\n"
    "print('Tabela 7 — comparação entre modelos de demanda:')\n"
    "display(tabela_7)"
)

add_md("## 12. Figura 3 — Diagnóstico do modelo OLS")
add_code(diagnostics_code)

add_md("## 13. Exportação do modelo final")
add_code(export_model_code)

add_md("## 14. Figura 4 — Curva de demanda estimada por estação")
add_code(
    "precos = np.arange(100, 1001, 25)\n"
    "cenarios = {\n"
    "    'Inverno': {'estacao_outono': 0, 'estacao_primavera': 0, 'estacao_verao': 0},\n"
    "    'Outono':  {'estacao_outono': 1, 'estacao_primavera': 0, 'estacao_verao': 0},\n"
    "    'Verão':   {'estacao_outono': 0, 'estacao_primavera': 0, 'estacao_verao': 1},\n"
    "}\n"
    "cores = {'Inverno': 'steelblue', 'Outono': 'orange', 'Verão': 'crimson'}\n"
    "fig, ax = plt.subplots(figsize=(9, 5))\n"
    "curva_receita_resumo = []\n"
    "for nome, dummies in cenarios.items():\n"
    "    contexto = {\n"
    "        'review_scores_location': sample['review_scores_location'],\n"
    "        'log_number_of_reviews': sample['log_number_of_reviews'],\n"
    "        'bairro_mean_ocup': sample['bairro_mean_ocup'],\n"
    "        **dummies,\n"
    "    }\n"
    "    occs = [predict_occ(contexto, p) for p in precos]\n"
    "    dias = [o * 91 for o in occs]\n"
    "    receitas = [p * d for p, d in zip(precos, dias)]\n"
    "    idx_max = int(np.argmax(receitas))\n"
    "    curva_receita_resumo.append({\n"
    "        'estacao': nome,\n"
    "        'preco_otimo_no_intervalo': int(precos[idx_max]),\n"
    "        'ocupacao_prevista_dias': float(dias[idx_max]),\n"
    "        'receita_prevista': float(receitas[idx_max]),\n"
    "    })\n"
    "    ax.plot(precos, dias, label=nome, color=cores[nome], lw=2)\n"
    "ax.set_xlabel('Preço (R$)')\n"
    "ax.set_ylabel('Dias ocupados no trimestre (91)')\n"
    "ax.set_title('Curva de demanda estimada por estação')\n"
    "ax.legend()\n"
    "ax.grid(False)\n"
    "plt.tight_layout()\n"
    "plt.savefig('curva_demanda_estimada.png', dpi=150, bbox_inches='tight')\n"
    "plt.show()\n"
    "display(pd.DataFrame(curva_receita_resumo).round(2))"
)

add_md("## 15. Definição dos algoritmos MAB")
add_code(mab_classes_code)

add_md("## 16. Simulação MAB — cenário estacionário (verão)")
add_code(stationary_sim_code)

add_code(
    "tabela_8 = pd.DataFrame([\n"
    "    {\n"
    "        'Algoritmo': nome,\n"
    "        'Receita_media_R$_mil': media_por_algo[nome][-1],\n"
    "        'P10_R$_mil': ic_inf[nome][-1],\n"
    "        'P90_R$_mil': ic_sup[nome][-1],\n"
    "        'Braco_preferido': f\"R$ {precos_bracos[int(np.argmax(contagem_bracos[nome]))]}\",\n"
    "        'Concentracao_pct': contagem_bracos[nome].max() / contagem_bracos[nome].sum() * 100,\n"
    "    }\n"
    "    for nome in resultados_por_algo\n"
    "]).round(1)\n"
    "display(tabela_8)\n\n"
    "cores = {\n"
    "    'ε-Greedy': '#2196F3',\n"
    "    'UCB': '#4CAF50',\n"
    "    'Thompson Sampling': '#FF5722',\n"
    "}\n\n"
    "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n"
    "for nome in resultados_por_algo:\n"
    "    axes[0].plot(media_por_algo[nome], label=nome, color=cores[nome], lw=2)\n"
    "    axes[0].fill_between(range(N_RODADAS), ic_inf[nome], ic_sup[nome], alpha=0.15, color=cores[nome])\n"
    "    axes[1].plot(media_por_algo[nome][:100], label=nome, color=cores[nome], lw=2)\n"
    "    axes[1].fill_between(range(100), ic_inf[nome][:100], ic_sup[nome][:100], alpha=0.15, color=cores[nome])\n"
    "axes[0].set_title('Receita acumulada por algoritmo')\n"
    "axes[0].set_xlabel('Rodada')\n"
    "axes[0].set_ylabel('Receita acumulada (R$ mil)')\n"
    "axes[0].legend()\n"
    "axes[0].grid(False)\n"
    "axes[1].set_title('Zoom — primeiras 100 rodadas')\n"
    "axes[1].set_xlabel('Rodada')\n"
    "axes[1].set_ylabel('Receita acumulada (R$ mil)')\n"
    "axes[1].legend()\n"
    "axes[1].grid(False)\n"
    "plt.tight_layout()\n"
    "plt.savefig('simulacao_mab_receita_estacionario.png', dpi=150, bbox_inches='tight')\n"
    "plt.show()\n\n"
    "fig, axes = plt.subplots(1, 2, figsize=(16, 5))\n"
    "x = np.arange(n_arms)\n"
    "largura = 0.25\n"
    "for idx, nome in enumerate(resultados_por_algo):\n"
    "    total = contagem_bracos[nome].sum()\n"
    "    pcts = contagem_bracos[nome] / total * 100\n"
    "    axes[0].bar(x + idx * largura, pcts, largura, label=nome, color=cores[nome], edgecolor='white')\n"
    "    receita_rodada = np.diff(media_por_algo[nome], prepend=0)\n"
    "    media_movel = np.convolve(receita_rodada, np.ones(50) / 50, mode='valid')\n"
    "    axes[1].plot(media_movel, label=nome, color=cores[nome], lw=2)\n"
    "axes[0].set_xlabel('Preço do braço (R$)')\n"
    "axes[0].set_ylabel('Frequência de escolha (%)')\n"
    "axes[0].set_title('Distribuição de braços selecionados')\n"
    "axes[0].set_xticks(x + largura)\n"
    "axes[0].set_xticklabels([f'R${p}' for p in precos_bracos], rotation=45, ha='right')\n"
    "axes[0].legend()\n"
    "axes[0].grid(False)\n"
    "axes[1].set_xlabel('Rodada')\n"
    "axes[1].set_ylabel('Receita por rodada (R$ mil, média móvel)')\n"
    "axes[1].set_title('Receita por rodada — média móvel de 50 rodadas')\n"
    "axes[1].legend()\n"
    "axes[1].grid(False)\n"
    "plt.tight_layout()\n"
    "plt.savefig('simulacao_mab_bracos_estacionario.png', dpi=150, bbox_inches='tight')\n"
    "plt.show()"
)

add_md("## 17. Simulação MAB — cenário multi-estação")
add_code(multi_station_code)

add_code(
    "queda_pct = {\n"
    "    nome: (media_multi[nome][-1] / media_por_algo[nome][-1] - 1) * 100\n"
    "    for nome in resultados_multi\n"
    "}\n"
    "tabela_9 = pd.DataFrame([\n"
    "    {\n"
    "        'Algoritmo': nome,\n"
    "        'Receita_media_R$_mil': media_multi[nome][-1],\n"
    "        'P10_R$_mil': ic_inf_multi[nome][-1],\n"
    "        'P90_R$_mil': ic_sup_multi[nome][-1],\n"
    "        'Queda_vs_cenario_fixo_pct': queda_pct[nome],\n"
    "    }\n"
    "    for nome in resultados_multi\n"
    "]).round(1)\n"
    "display(tabela_9)\n\n"
    "fig, ax = plt.subplots(figsize=(10, 5))\n"
    "for nome in resultados_multi:\n"
    "    ax.plot(media_multi[nome], label=nome, color=cores[nome], lw=2)\n"
    "    ax.fill_between(range(N_RODADAS_TOTAL), ic_inf_multi[nome], ic_sup_multi[nome], alpha=0.15, color=cores[nome])\n"
    "for i, est in enumerate(ordem_estacoes):\n"
    "    x_pos = i * RODADAS_POR_ESTACAO\n"
    "    ax.axvline(x_pos, ls='--', color='gray', alpha=0.5, lw=1)\n"
    "    ax.text(x_pos + 5, ax.get_ylim()[0] + 0.5, est, fontsize=9, color='gray')\n"
    "ax.set_xlabel('Rodada')\n"
    "ax.set_ylabel('Receita acumulada (R$ mil)')\n"
    "ax.set_title('Receita acumulada — cenário multi-estação')\n"
    "ax.legend()\n"
    "ax.grid(False)\n"
    "plt.tight_layout()\n"
    "plt.savefig('simulacao_mab_receita_multi_estacao.png', dpi=150, bbox_inches='tight')\n"
    "plt.show()\n\n"
    "ts_matrix = np.zeros((len(ordem_estacoes), n_arms))\n"
    "for i, est in enumerate(ordem_estacoes):\n"
    "    total = contagem_multi['Thompson Sampling'][est].sum()\n"
    "    ts_matrix[i] = contagem_multi['Thompson Sampling'][est] / total * 100\n"
    "fig, ax = plt.subplots(figsize=(10, 5))\n"
    "im = ax.imshow(ts_matrix, cmap='YlOrRd', aspect='auto')\n"
    "ax.set_xticks(range(n_arms))\n"
    "ax.set_xticklabels([f'R${p}' for p in precos_bracos], rotation=45, ha='right')\n"
    "ax.set_yticks(range(len(ordem_estacoes)))\n"
    "ax.set_yticklabels(ordem_estacoes)\n"
    "ax.set_title('Thompson Sampling — distribuição de braços por estação (%)')\n"
    "plt.colorbar(im, ax=ax, label='%')\n"
    "for i in range(len(ordem_estacoes)):\n"
    "    for j in range(n_arms):\n"
    "        val = ts_matrix[i, j]\n"
    "        if val > 3:\n"
    "            ax.text(j, i, f'{val:.0f}', ha='center', va='center', fontsize=9, color='white' if val > 30 else 'black')\n"
    "plt.tight_layout()\n"
    "plt.savefig('heatmap_thompson_multi_estacao.png', dpi=150, bbox_inches='tight')\n"
    "plt.show()"
)

add_md("## 18. Resumo final da reprodução")
add_code(
    "resumo_reproducao = {\n"
    "    'amostra': {\n"
    "        'n_imoveis': int(df_feature_buinding['listing_id'].nunique()),\n"
    "        'n_observacoes': int(len(df_feature_buinding)),\n"
    "        'ocupacao_media_dias': float(df_feature_buinding['total_ocupacy_3'].mean()),\n"
    "    },\n"
    "    'pca': {\n"
    "        'kmo': float(kmo_model),\n"
    "        'bartlett_stat': float(bartlett_stat),\n"
    "        'bartlett_p': float(bartlett_p),\n"
    "        'variancia_explicada': float(pca.explained_variance_ratio_[0]),\n"
    "    },\n"
    "    'modelo_final': {\n"
    "        'tipo': 'OLS',\n"
    "        'r2': float(ols_red.rsquared),\n"
    "        'mae_dias': float(mae_ols * 91),\n"
    "        'rmse_dias': float(rmse_ols * 91),\n"
    "    },\n"
    "    'cenario_estacionario': {nome: float(media_por_algo[nome][-1]) for nome in media_por_algo},\n"
    "    'cenario_multi_estacao': {nome: float(media_multi[nome][-1]) for nome in media_multi},\n"
    "    'figuras': [\n"
    "        'Histogramas Antes e depois.png',\n"
    "        'correlacao_subnotas.png',\n"
    "        'diagnostico_modelo_reduzido.png',\n"
    "        'curva_demanda_estimada.png',\n"
    "        'simulacao_mab_receita_estacionario.png',\n"
    "        'simulacao_mab_bracos_estacionario.png',\n"
    "        'simulacao_mab_receita_multi_estacao.png',\n"
    "        'heatmap_thompson_multi_estacao.png',\n"
    "    ],\n"
    "}\n"
    "with open('resultados_reproducao_publica.json', 'w', encoding='utf-8') as fp:\n"
    "    json.dump(resumo_reproducao, fp, ensure_ascii=False, indent=2)\n"
    "print(json.dumps(resumo_reproducao, ensure_ascii=False, indent=2))"
)

metadata = nb02.get("metadata", {})
metadata.setdefault(
    "kernelspec",
    {"display_name": "Python 3", "language": "python", "name": "python3"},
)
metadata.setdefault("language_info", {"name": "python", "version": "3.x"})

notebook = {
    "cells": cells,
    "metadata": metadata,
    "nbformat": 4,
    "nbformat_minor": 5,
}

NB_OUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1))
print(f"Notebook criado em: {NB_OUT}")
