# Precificação Dinâmica de Estúdios por Temporada com Multi-Armed Bandits

Repositório público de reprodução do TCC sobre precificação dinâmica de estúdios por temporada no Rio de Janeiro com algoritmos Multi-Armed Bandits.

O material principal para reprodução está no notebook [Dados/04_Reproducao_Publica_TCC.ipynb](Dados/04_Reproducao_Publica_TCC.ipynb), organizado para seguir a lógica final do trabalho e gerar os principais resultados, tabelas e figuras.

## Conteúdo

- `Dados/04_Reproducao_Publica_TCC.ipynb`: notebook principal de reprodução.
- `Dados/tcc_referencia_final.pdf`: texto final do TCC.
- `Dados/listings_*.csv.gz` e `Dados/calendar_*.csv.gz`: dados brutos usados no pipeline.
- `Dados/base_final_para_modelagem.csv`: base analítica final.
- `Dados/modelo_ocupacao_final.pkl`: modelo OLS final exportado.
- `Dados/resultados_reproducao_publica.json`: resumo numérico gerado pelo notebook.
- `Dados/*.png`: figuras centrais reproduzidas no trabalho.
- `tools/create_public_notebook.py`: script que gera o notebook público.
- `tools/validate_public_notebook.py`: script de validação automatizada.

## Estrutura

```text
.
├── README.md
├── requirements.txt
├── Dados
│   ├── 04_Reproducao_Publica_TCC.ipynb
│   ├── tcc_referencia_final.pdf
│   ├── listings_*.csv.gz
│   ├── calendar_*.csv.gz
│   ├── base_final_para_modelagem.csv
│   ├── modelo_ocupacao_final.pkl
│   ├── resultados_reproducao_publica.json
│   └── figuras geradas
└── tools
    ├── create_public_notebook.py
    └── validate_public_notebook.py
```

## Como reproduzir

1. Crie um ambiente Python 3.10+.
2. Instale as dependências:

```bash
pip install -r requirements.txt
```

3. Abra o notebook `Dados/04_Reproducao_Publica_TCC.ipynb`.
4. Execute o notebook de ponta a ponta com o diretório de trabalho apontando para `Dados/`.

Opção via terminal:

```bash
cd Dados
jupyter nbconvert --to notebook --execute 04_Reproducao_Publica_TCC.ipynb --output 04_Reproducao_Publica_TCC.executed.ipynb
```

## O que o notebook faz

1. Lê os quatro snapshots trimestrais de `listings` e `calendar`.
2. Reconstrói a amostra balanceada de 571 imóveis e 2.284 observações.
3. Deriva a ocupação trimestral prospectiva.
4. Aplica tratamento, winsorização, transformações logarítmicas e engenharia de variáveis.
5. Consolida qualidade percebida por PCA.
6. Treina o modelo final de demanda por OLS.
7. Compara OLS com Random Forest e Gradient Boosting.
8. Simula os algoritmos `ε-Greedy`, `UCB` e `Thompson Sampling` em dois cenários:
   cenário estacionário e cenário multi-estação.

## Principais artefatos gerados

- `Histogramas Antes e depois.png`
- `correlacao_subnotas.png`
- `diagnostico_modelo_reduzido.png`
- `curva_demanda_estimada.png`
- `simulacao_mab_receita_estacionario.png`
- `simulacao_mab_bracos_estacionario.png`
- `simulacao_mab_receita_multi_estacao.png`
- `heatmap_thompson_multi_estacao.png`
- `resultados_reproducao_publica.json`

## Fonte dos dados

Os dados brutos utilizados derivam do projeto Inside Airbnb para a cidade do Rio de Janeiro, com snapshots de junho/2024, setembro/2024, dezembro/2024 e março/2025.

## Observações

- Os arquivos `reviews_*.csv.gz` não foram incluídos porque não são usados no pipeline final do notebook público.
- O repositório mantém a base final e o modelo exportado para permitir auditoria dos resultados sem necessidade de rerodar todo o ETL.
