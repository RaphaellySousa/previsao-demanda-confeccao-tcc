# Previsão de demanda em indústria de confecção com data analytics e aprendizagem de máquina

**Universidade de São Paulo – Escola Superior de Agricultura "Luiz de Queiroz" (USP/Esalq)**  
MBA em Data Science e Analytics – Trabalho de Conclusão de Curso (2026)

- **Autora:** Raphaelly Sousa Silva
- **Orientador:** Fabio Lima

## Sobre o trabalho

O TCC estruturou um procedimento de previsão inicial de demanda, por coleção (Inverno, Verão e
Alto Verão), para uma indústria de confecção com duas unidades (Matriz e Filial) e duas marcas.
Foram usados dados internos de 2023 a 2025: os modelos foram treinados com 2023 e 2024 e
testados em 2025. Foram comparados dois modelos de referência (baseline do ano anterior e média
de 2023–2024) e quatro modelos de aprendizagem de máquina (Ridge, Lasso, Random Forest e
Gradient Boosting), avaliados por MAE, RMSE e MAPE.

## O que tem neste repositório

Somente os códigos em Python e as instruções de execução. **O arquivo Excel com a base de dados
original da empresa não foi publicado**, por ser informação interna. As marcas foram
anonimizadas como A e B e nenhum dado de cliente ou representante foi utilizado.

```
scripts/
├── 01_preparar_dados.py         leitura do Excel, limpeza, diagnóstico e bases analíticas
├── 02_analise_exploratoria.py   tabelas e gráficos da análise exploratória (G01–G15)
└── 03_modelos_previsao.py       modelos, métricas, análise de erros e gráficos (G16–G24)
requirements.txt                 bibliotecas usadas
```

## Como executar (Anaconda + Spyder)

1. Instalar as bibliotecas (já vêm no Anaconda):
   ```
   pip install -r requirements.txt
   ```
2. Abrir os scripts no Spyder e rodar na ordem, com **F5**:
   - `01_preparar_dados.py` (precisa do Excel `Base de dados/RESUMOS 2023 - 2025.xlsx`, que não está aqui);
   - `02_analise_exploratoria.py` (usa as bases geradas pelo script 01);
   - `03_modelos_previsao.py`.
3. Os gráficos aparecem na aba **Plots** do Spyder e são salvos na pasta `graficos/`.
   As tabelas de resultado vão para `resultados/resultados_modelos.xlsx`.

**O script 03 roda sozinho, sem o Excel**, porque as tabelas agregadas por ano, unidade, marca e
coleção (36 linhas de peças e 18 de faturamento) estão escritas dentro do próprio código.
Assim é possível reproduzir os resultados dos modelos apresentados no TCC.

## Observações

- Validação temporal: treino com 2023–2024 e teste com 2025 (sem divisão aleatória).
- Parâmetros fixos: Ridge (alfa = 1), Lasso (alfa = 0,001), Random Forest (500 árvores, semente 0)
  e Gradient Boosting (semente 42). Com eles, os resultados saem iguais aos do TCC.
- O Excel só tinha as abas de 2024 e 2025; os totais de 2023 vieram do relatório consolidado.
