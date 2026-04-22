# Precificação Dinâmica de Estúdios por Temporada com Multi-Armed Bandits

Repositório público de reprodução do TCC sobre precificação dinâmica de estúdios por temporada no Rio de Janeiro com algoritmos Multi-Armed Bandits.

O material principal está no notebook [Dados/04_Reproducao_Publica_TCC.ipynb](Dados/04_Reproducao_Publica_TCC.ipynb), organizado para seguir a lógica final do trabalho e reproduzir os principais resultados, tabelas e figuras do estudo.

## Visão Geral

O trabalho investiga o uso de algoritmos Multi-Armed Bandits para precificação dinâmica de estúdios de aluguel por temporada. A análise utiliza dados públicos do Inside Airbnb para o Rio de Janeiro, com quatro snapshots trimestrais entre junho de 2024 e março de 2025.

O pipeline público deste repositório:

- reconstrói a amostra analítica a partir dos dados brutos;
- gera a variável de ocupação trimestral prospectiva;
- consolida indicadores de qualidade via PCA;
- estima a função de demanda por OLS;
- compara OLS com Random Forest e Gradient Boosting;
- simula `ε-Greedy`, `UCB` e `Thompson Sampling` em cenários estacionário e multi-estação.

## Acesso Rápido

- [Dados/reproducao_publica_tcc.ipynb](Dados/reproducao_publica_tcc.ipynb): notebook principal de reprodução.
- [Dados/resultados_reproducao_publica.json](Dados/resultados_reproducao_publica.json): resumo numérico final gerado pelo notebook.
- `Dados/base_final_para_modelagem.csv`: base analítica final.
- `Dados/modelo_ocupacao_final.pkl`: modelo OLS final exportado.

## Estrutura

```text
.
├── README.md
├── requirements.txt
├── Dados
│   ├── reproducao_publica_tcc.ipynb
│   ├── listings_*.csv.gz
│   ├── calendar_*.csv.gz
│   ├── base_final_para_modelagem.csv
│   ├── modelo_ocupacao_final.pkl
│   ├── resultados_reproducao_publica.json
│   └── figuras geradas
```

## Requisitos

- Python 3.10+
- Jupyter Notebook ou JupyterLab

Instalação das dependências:

```bash
pip install -r requirements.txt
```

## Como Executar

### Opção 1: Jupyter

1. Abra o notebook `Dados/reproducao_publica_tcc.ipynb`.
2. Configure o diretório de execução para `Dados/`.
3. Execute o notebook de ponta a ponta.

### Opção 2: terminal

```bash
cd Dados
jupyter nbconvert --to notebook --execute reproducao_publica_tcc.ipynb --output reproducao_publica_tcc.executed.ipynb
```

## O Que Esperar da Reprodução

- Amostra final de `571` imóveis e `2.284` observações.
- Base final salva em `Dados/base_final_para_modelagem.csv`.
- Modelo OLS final salvo em `Dados/modelo_ocupacao_final.pkl`.
- Resumo final salvo em `Dados/resultados_reproducao_publica.json`.
- Geração das principais figuras do trabalho.

Resultados centrais esperados:

- `R²` do modelo OLS final em torno de `0.10`.
- Cenário estacionário:
  `UCB` > `ε-Greedy` > `Thompson Sampling` em receita acumulada.
- Cenário multi-estação:
  `UCB` mantém maior receita absoluta, mas `Thompson Sampling` apresenta menor queda relativa entre cenários.

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

## Observações Sobre o Repositório

- Os arquivos `reviews_*.csv.gz` não foram incluídos porque não são usados no pipeline final do notebook público.
- O repositório mantém a base final e o modelo exportado para permitir auditoria dos resultados sem necessidade de rerodar todo o ETL.
