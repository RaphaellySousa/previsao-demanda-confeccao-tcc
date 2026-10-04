# -*- coding: utf-8 -*-
"""
===============================================================================
UNIVERSIDADE DE SÃO PAULO
Escola Superior de Agricultura "Luiz de Queiroz" (USP/Esalq)
MBA em Data Science e Analytics

Trabalho de Conclusão de Curso (TCC)
Título:     Previsão de demanda em indústria de confecção com data analytics
            e aprendizagem de máquina
Autora:     Raphaelly Sousa Silva
Orientador: Fabio Lima
Ano:        2026

Script 3 de 3 - 03_modelos_previsao.py
===============================================================================

O que este script faz:
    Aqui está a parte principal do TCC: os modelos de previsão.
    Eu treino os modelos com 2023 e 2024 e vejo o quanto eles erram ao
    prever 2025 (que eu já sei quanto vendeu). Assim dá para comparar qual
    modelo é melhor.

    Base: unidade x marca x coleção x ano (36 linhas)
      - treino: 2023 e 2024 (24 linhas)
      - teste:  2025 (12 linhas)
      - o que eu quero prever: peças vendidas

    Modelos que testei:
      - Baseline (t-1): repete o valor do ano anterior (o mais simples possível)
      - Baseline (média 2023-2024): média dos dois anos anteriores
      - Ridge e Lasso: regressões lineares
      - Random Forest e Gradient Boosting: modelos de árvore

    Métricas: MAE, RMSE e MAPE (as fórmulas estão no TCC).

    No final repito tudo usando o FATURAMENTO no lugar das peças, para ver se
    o resultado muda (análise de sensibilidade).

Os dados estão dentro do próprio código:
    As tabelas agregadas são pequenas (36 linhas de peças e 18 de faturamento),
    então coloquei elas aqui embaixo, na parte 1. Assim este script roda
    sozinho, sem precisar do Excel e sem rodar o script 01 antes. Se o script
    01 já tiver sido rodado, eu confiro se os números dele batem com os daqui.

Gráficos (entre parênteses, o número da figura no TCC):
    G16 métricas por modelo                G17 observado x previsto
    G18 dispersão observado x previsto     G19 MAPE por coleção (Figura 9)
    G20 erro por segmento (Figura 10)      G21 erro % / viés (Figura 11)
    G22 importância das variáveis (Figura 12)
    G23 MAPE peças x faturamento           G24 faturamento observado x previsto

Como rodar no Spyder:
    F5. As tabelas aparecem no console, os gráficos na aba "Plots", e tudo
    fica salvo nas pastas "graficos" e "resultados".
"""


# =============================================================================
# %% 0. BIBLIOTECAS E PASTAS
# =============================================================================
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Modelos do scikit-learn
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

try:
    PASTA_PROJETO = Path(__file__).resolve().parent.parent
except NameError:
    PASTA_PROJETO = Path.cwd().parent

PASTA_DADOS = PASTA_PROJETO / "dados"
PASTA_GRAFICOS = PASTA_PROJETO / "graficos"
PASTA_RESULTADOS = PASTA_PROJETO / "resultados"
PASTA_GRAFICOS.mkdir(exist_ok=True)
PASTA_RESULTADOS.mkdir(exist_ok=True)

COLECOES = ["Inverno", "Verão", "Alto Verão"]
ANO_TESTE = 2025          # o ano que eu "escondo" do modelo para testar
FONTE = "Fonte: dados internos da empresa (2023-2025). Elaboração própria."

plt.rcParams.update({"figure.figsize": (10, 6), "font.size": 11,
                     "axes.grid": True, "grid.alpha": 0.3})
pd.options.display.float_format = "{:,.2f}".format
pd.set_option("display.width", 150)
pd.set_option("display.max_columns", 20)


def salvar(fig, nome):
    """Coloca a fonte, salva o gráfico em PNG e mostra no Spyder."""
    fig.text(0.01, 0.005, FONTE, fontsize=8, style="italic")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(PASTA_GRAFICOS / f"{nome}.png", dpi=300)
    plt.show()


# =============================================================================
# %% 1. DADOS (DENTRO DO PRÓPRIO CÓDIGO)
# =============================================================================
# Peças vendidas por ano, unidade, marca e coleção.
# São os mesmos números da Tabela 2 do TCC. 2023 veio do relatório
# consolidado; 2024 e 2025 vieram do Excel (script 01).
# IMPORTANTE: deixei as linhas nesta ordem de propósito (ano, marca,
# unidade, coleção), porque a ordem muda um pouco o Random Forest.
DADOS_PECAS = [
    # ano,  unidade, marca, coleção,      peças
    (2023, "MATRIZ", "A", "Inverno",     347057),
    (2023, "MATRIZ", "A", "Verão",       143211),
    (2023, "MATRIZ", "A", "Alto Verão",  402691),
    (2023, "FILIAL", "A", "Inverno",      75076),
    (2023, "FILIAL", "A", "Verão",        32095),
    (2023, "FILIAL", "A", "Alto Verão",   86392),
    (2023, "MATRIZ", "B", "Inverno",     224666),
    (2023, "MATRIZ", "B", "Verão",       139579),
    (2023, "MATRIZ", "B", "Alto Verão",  257677),
    (2023, "FILIAL", "B", "Inverno",      48387),
    (2023, "FILIAL", "B", "Verão",        33236),
    (2023, "FILIAL", "B", "Alto Verão",   59664),
    (2024, "MATRIZ", "A", "Inverno",     408158),
    (2024, "MATRIZ", "A", "Verão",       196085),
    (2024, "MATRIZ", "A", "Alto Verão",  409891),
    (2024, "FILIAL", "A", "Inverno",      63917),
    (2024, "FILIAL", "A", "Verão",        38789),
    (2024, "FILIAL", "A", "Alto Verão",   81646),
    (2024, "MATRIZ", "B", "Inverno",     252424),
    (2024, "MATRIZ", "B", "Verão",       169554),
    (2024, "MATRIZ", "B", "Alto Verão",  287905),
    (2024, "FILIAL", "B", "Inverno",      50033),
    (2024, "FILIAL", "B", "Verão",        34028),
    (2024, "FILIAL", "B", "Alto Verão",   57676),
    (2025, "MATRIZ", "A", "Inverno",     441665),
    (2025, "MATRIZ", "A", "Verão",       203488),
    (2025, "MATRIZ", "A", "Alto Verão",  458132),
    (2025, "FILIAL", "A", "Inverno",      75853),
    (2025, "FILIAL", "A", "Verão",        45261),
    (2025, "FILIAL", "A", "Alto Verão",   94863),
    (2025, "MATRIZ", "B", "Inverno",     287726),
    (2025, "MATRIZ", "B", "Verão",       165586),
    (2025, "MATRIZ", "B", "Alto Verão",  305466),
    (2025, "FILIAL", "B", "Inverno",      54433),
    (2025, "FILIAL", "B", "Verão",        36420),
    (2025, "FILIAL", "B", "Alto Verão",   63556),
]

# Faturamento (R$) por ano, unidade e coleção (mesmos números da Tabela 1)
DADOS_FATURAMENTO = [
    # ano,  unidade, coleção,      faturamento (R$)
    (2023, "MATRIZ", "Inverno",    40590223.86),
    (2023, "MATRIZ", "Verão",      16756951.33),
    (2023, "MATRIZ", "Alto Verão", 36529623.84),
    (2023, "FILIAL", "Inverno",    13650584.10),
    (2023, "FILIAL", "Verão",       7250573.79),
    (2023, "FILIAL", "Alto Verão", 15741419.97),
    (2024, "MATRIZ", "Inverno",    39830631.42),
    (2024, "MATRIZ", "Verão",      20256262.81),
    (2024, "MATRIZ", "Alto Verão", 38413090.91),
    (2024, "FILIAL", "Inverno",    13204822.10),
    (2024, "FILIAL", "Verão",       7572097.72),
    (2024, "FILIAL", "Alto Verão", 14777104.94),
    (2025, "MATRIZ", "Inverno",    43912831.56),
    (2025, "MATRIZ", "Verão",      21999749.89),
    (2025, "MATRIZ", "Alto Verão", 47497891.47),
    (2025, "FILIAL", "Inverno",    15614232.90),
    (2025, "FILIAL", "Verão",       9478620.12),
    (2025, "FILIAL", "Alto Verão", 19355844.53),
]

# Transformo as listas em tabelas do pandas
pecas = pd.DataFrame(DADOS_PECAS, columns=["ano", "unidade", "marca", "colecao", "pecas"])
pecas["pecas"] = pecas["pecas"].astype(float)
fat = pd.DataFrame(DADOS_FATURAMENTO, columns=["ano", "unidade", "colecao", "faturamento"])

# Se o script 01 já foi rodado, confiro se os CSVs dele batem com os dados daqui.
# (Se eu mudar alguma coisa no tratamento, este aviso me mostra.)
arq_csv = PASTA_DADOS / "base_agregada_pecas.csv"
if arq_csv.exists():
    csv = pd.read_csv(arq_csv)
    iguais = np.allclose(csv["pecas"].values, pecas["pecas"].values)
    print("Conferência com o CSV do script 01:",
          "os números batem." if iguais else "ATENÇÃO, os números são diferentes!")
else:
    print("O script 01 ainda não foi rodado, mas tudo bem: uso os dados deste arquivo.")


# =============================================================================
# %% 2. MÉTRICAS DE ERRO
# =============================================================================
# As três métricas do TCC. Quanto menor, melhor.
def mae(real, previsto):
    # Erro absoluto médio: em média, quantas peças o modelo erra
    return np.mean(np.abs(real - previsto))


def rmse(real, previsto):
    # Raiz do erro quadrático médio: parecido com o MAE, mas "pune" mais os erros grandes
    return np.sqrt(np.mean((real - previsto) ** 2))


def mape(real, previsto):
    # Erro percentual médio: o erro em % do valor real.
    # Obs.: o MAPE dá problema quando o valor real é zero (divisão por zero),
    # mas aqui o menor valor é 36.420 peças, então não precisei tratar isso.
    return np.mean(np.abs((real - previsto) / real)) * 100


def tabela_metricas(df, alvo, modelos):
    """Monta uma tabela com MAE, RMSE e MAPE de cada modelo."""
    linhas = []
    for m in modelos:
        linhas.append({"Modelo": m,
                       "MAE": mae(df[alvo], df[m]),
                       "RMSE": rmse(df[alvo], df[m]),
                       "MAPE (%)": mape(df[alvo], df[m])})
    return pd.DataFrame(linhas)


# =============================================================================
# %% 3. FUNÇÃO QUE TREINA OS MODELOS E FAZ A PREVISÃO
# =============================================================================
# Fiz uma função porque uso a mesma coisa duas vezes: para peças e para faturamento.
def prever(base, alvo, chaves, ano_teste=ANO_TESTE):
    """Treina com os anos antes de ano_teste e prevê ano_teste."""
    # Separação no tempo: treino = passado, teste = 2025.
    # Não pode ser aleatório, senão o modelo "veria o futuro".
    treino = base[base["ano"] < ano_teste].copy()
    teste = base[base["ano"] == ano_teste].copy()

    # --- Baselines (os modelos de referência) ---
    # t-1: a previsão é o que vendeu no ano anterior
    ano_anterior = treino[treino["ano"] == ano_teste - 1].set_index(chaves)[alvo]
    # média: a previsão é a média dos dois anos anteriores
    media_2_anos = (treino[treino["ano"] >= ano_teste - 2]
                    .groupby(chaves)[alvo].mean())
    teste["Baseline (t-1)"] = teste.set_index(chaves).index.map(ano_anterior)
    teste["Baseline (média 2023-2024)"] = teste.set_index(chaves).index.map(media_2_anos)

    # --- Variáveis de entrada dos modelos ---
    # Os modelos não entendem texto ("MATRIZ", "Verão"...), então usei
    # one-hot encoding (get_dummies): cada categoria vira uma coluna de 0 e 1.
    # O ano entra como número para o modelo tentar pegar o crescimento.
    tudo = pd.concat([treino, teste])
    X = pd.get_dummies(tudo[chaves], dtype=float)
    X["ano"] = tudo["ano"].astype(float)
    X_treino, X_teste = X.iloc[:len(treino)], X.iloc[len(treino):]
    y_treino = treino[alvo]

    # --- Modelos de aprendizagem de máquina ---
    # Usei parâmetros fixos. Não dava para fazer busca de parâmetros porque
    # com 3 anos eu teria que usar o próprio 2025, e aí o teste não valeria.
    # O random_state fixo é para o resultado sair sempre igual.
    modelos = {
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.001, max_iter=10000),
        "Random Forest": RandomForestRegressor(n_estimators=500, random_state=0),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    }
    for nome, modelo in modelos.items():
        modelo.fit(X_treino, y_treino)          # aprende com 2023 e 2024
        teste[nome] = modelo.predict(X_teste)   # prevê 2025

    return teste, modelos, X.columns


NOMES = ["Baseline (t-1)", "Baseline (média 2023-2024)", "Ridge", "Lasso",
         "Random Forest", "Gradient Boosting"]
# Uma cor fixa para cada modelo em todos os gráficos
CORES = {"Baseline (t-1)": "#2c3e50", "Baseline (média 2023-2024)": "#95a5a6",
         "Ridge": "#e67e22", "Lasso": "#f1c40f",
         "Random Forest": "#27ae60", "Gradient Boosting": "#2980b9"}


# =============================================================================
# PARTE A - PEÇAS (o alvo principal do TCC)
# =============================================================================
# %% 4. TREINANDO E PREVENDO
# =============================================================================
CHAVES = ["unidade", "marca", "colecao"]
print("Linhas de treino:", (pecas["ano"] < ANO_TESTE).sum(),
      "| linhas de teste:", (pecas["ano"] == ANO_TESTE).sum())
print("Menor valor real em 2025:", pecas.loc[pecas["ano"] == ANO_TESTE, "pecas"].min())

teste, modelos, colunas_X = prever(pecas, "pecas", CHAVES)


# =============================================================================
# %% 5. TABELA DE DESEMPENHO (TABELA 3 DO TCC)
# =============================================================================
resultado = tabela_metricas(teste, "pecas", NOMES)
print("\n=== DESEMPENHO - PEÇAS, TESTE EM 2025 ===")
print(resultado.to_string(index=False))

# Confiro se os números batem com os que escrevi no TCC
TEXTO = {"Baseline (t-1)": 9.17, "Baseline (média 2023-2024)": 11.94,
         "Ridge": 36.53, "Lasso": 36.97, "Random Forest": 9.48,
         "Gradient Boosting": 9.53}
print("\nConferindo o MAPE com o texto do TCC:")
for m, v in TEXTO.items():
    calc = resultado.set_index("Modelo").loc[m, "MAPE (%)"]
    print(f"  {m:<28} código = {calc:6.2f}% | texto = {v:6.2f}% ->",
          "OK" if abs(calc - v) < 0.01 else "VERIFICAR")


# =============================================================================
# %% 6. OBSERVADO X PREVISTO POR SEGMENTO (TABELA 4 DO TCC)
# =============================================================================
teste["segmento"] = teste["unidade"].str.title() + " - " + teste["marca"]
tabela2 = teste[["unidade", "marca", "colecao", "pecas", "Baseline (t-1)",
                 "Gradient Boosting", "Random Forest"]].copy()
tabela2 = tabela2.rename(columns={"pecas": "Observado 2025"})
print("\n=== OBSERVADO x PREVISTO (2025) ===")
print(tabela2.round(0).to_string(index=False))


# =============================================================================
# %% 7. ERRO SEPARADO POR COLEÇÃO E POR SEGMENTO
# =============================================================================
# Separei porque um erro médio baixo pode esconder erros grandes justamente
# nos picos (Inverno e Alto Verão), que é onde errar custa mais caro.
por_colecao = pd.concat([tabela_metricas(d, "pecas", NOMES).assign(Coleção=c)
                         for c, d in teste.groupby("colecao")])
por_segmento = pd.concat([tabela_metricas(d, "pecas", NOMES).assign(Segmento=s)
                          for s, d in teste.groupby("segmento")])
print("\n=== DESEMPENHO POR COLEÇÃO ===")
print(por_colecao.pivot_table(index="Coleção", columns="Modelo",
                              values=["MAE", "MAPE (%)"]).loc[COLECOES].round(2))
print("\n=== DESEMPENHO POR UNIDADE E MARCA ===")
print(por_segmento.pivot_table(index="Segmento", columns="Modelo",
                               values=["MAE", "MAPE (%)"]).round(2))


# =============================================================================
# %% 8. IMPORTÂNCIA DAS VARIÁVEIS (RANDOM FOREST)
# =============================================================================
# Mostra quais variáveis o Random Forest mais usou. Não quer dizer que uma
# coisa "causa" a outra, é só uma pista. Achei interessante que o "ano"
# quase não tem importância, por isso o modelo não acompanha o crescimento.
importancia = (pd.Series(modelos["Random Forest"].feature_importances_, index=colunas_X)
               .sort_values(ascending=True))
print("\n=== IMPORTÂNCIA DAS VARIÁVEIS (Random Forest) ===")
print(importancia.sort_values(ascending=False))


# =============================================================================
# %% G16 - MÉTRICAS POR MODELO
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 6))
for ax, met in zip(axes, ["MAE", "RMSE", "MAPE (%)"]):
    ax.barh(resultado["Modelo"], resultado[met],
            color=[CORES[m] for m in resultado["Modelo"]])
    for i, v in enumerate(resultado[met]):
        ax.text(v, i, f" {v:,.1f}" if met == "MAPE (%)" else f" {v:,.0f}",
                va="center", fontsize=9)
    ax.set_title(met)
    ax.set_xlabel("Peças" if met != "MAPE (%)" else "%")
    ax.invert_yaxis()
    ax.set_xlim(0, resultado[met].max() * 1.25)
    if met != "MAPE (%)":
        ax.xaxis.set_major_formatter(lambda v, p: f"{v/1000:,.0f} mil".replace(",", "."))
axes[1].set_yticklabels([])
axes[2].set_yticklabels([])
fig.suptitle("Desempenho dos modelos na previsão de peças - teste 2025")
salvar(fig, "G16_metricas_modelos")


# =============================================================================
# %% G17 - OBSERVADO X PREVISTO (BARRAS LADO A LADO)
# =============================================================================
rotulos = (teste["segmento"] + "\n" + teste["colecao"]).values
x = np.arange(len(teste))
larg = 0.27     # largura de cada barra
fig, ax = plt.subplots(figsize=(15, 6))
ax.bar(x - larg, teste["pecas"], larg, label="Observado 2025", color="#7f7f7f")
ax.bar(x, teste["Baseline (t-1)"], larg, label="Baseline (t-1)", color=CORES["Baseline (t-1)"])
ax.bar(x + larg, teste["Gradient Boosting"], larg, label="Gradient Boosting",
       color=CORES["Gradient Boosting"])
ax.set_xticks(x)
ax.set_xticklabels(rotulos, fontsize=8)
ax.set_title("Peças em 2025: observado x previsto, por segmento e coleção")
ax.set_xlabel("Segmento (unidade - marca) / coleção")
ax.set_ylabel("Peças")
ax.yaxis.set_major_formatter(lambda v, p: f"{v/1000:,.0f} mil".replace(",", "."))
ax.legend()
salvar(fig, "G17_observado_previsto")


# =============================================================================
# %% G18 - DISPERSÃO OBSERVADO X PREVISTO
# =============================================================================
# Se o modelo acertasse tudo, os pontos ficariam em cima da linha tracejada.
# Pontos abaixo da linha = o modelo previu menos do que vendeu.
fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharex=True, sharey=True)
limite = teste["pecas"].max() * 1.1
for ax, m in zip(axes, ["Baseline (t-1)", "Ridge", "Random Forest", "Gradient Boosting"]):
    ax.scatter(teste["pecas"], teste[m], color=CORES[m], s=50)
    ax.plot([0, limite], [0, limite], "k--", linewidth=1, label="Previsão perfeita")
    ax.set_title(f"{m}\nMAPE = {mape(teste['pecas'], teste[m]):.2f}%")
    ax.set_xlabel("Peças observadas")
    ax.xaxis.set_major_formatter(lambda v, p: f"{v/1000:,.0f} mil".replace(",", "."))
    ax.yaxis.set_major_formatter(lambda v, p: f"{v/1000:,.0f} mil".replace(",", "."))
axes[0].set_ylabel("Peças previstas")
axes[0].legend(loc="upper left")
fig.suptitle("Observado x previsto em 2025 (pontos abaixo da linha = subestimação)")
salvar(fig, "G18_dispersao_observado_previsto")


# =============================================================================
# %% G19 - MAPE POR COLEÇÃO (FIGURA 9)
# =============================================================================
t = por_colecao.pivot_table(index="Coleção", columns="Modelo",
                            values="MAPE (%)").loc[COLECOES, NOMES]
fig, ax = plt.subplots(figsize=(12, 6))
t.plot(kind="bar", ax=ax, color=[CORES[m] for m in NOMES], width=0.85)
ax.set_title("MAPE por coleção e modelo - teste 2025")
ax.set_xlabel("Coleção")
ax.set_ylabel("MAPE (%)")
ax.tick_params(axis="x", rotation=0)
ax.legend(title="Modelo", bbox_to_anchor=(1.02, 1), loc="upper left")
salvar(fig, "G19_mape_por_colecao")


# =============================================================================
# %% G20 - MAPE E MAE POR UNIDADE E MARCA (FIGURA 10)
# =============================================================================
ordem_seg = ["Matriz - A", "Matriz - B", "Filial - A", "Filial - B"]
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for ax, met in zip(axes, ["MAPE (%)", "MAE"]):
    t = por_segmento.pivot_table(index="Segmento", columns="Modelo",
                                 values=met).loc[ordem_seg, NOMES]
    t.plot(kind="bar", ax=ax, color=[CORES[m] for m in NOMES], width=0.85, legend=False)
    ax.set_title(f"{met} por unidade e marca")
    ax.set_xlabel("Segmento (unidade - marca)")
    ax.set_ylabel(met if met != "MAE" else "MAE (peças)")
    ax.tick_params(axis="x", rotation=0)
axes[1].legend(title="Modelo", bbox_to_anchor=(1.02, 1), loc="upper left")
fig.suptitle("Erro por segmento - teste 2025")
salvar(fig, "G20_erro_por_segmento")


# =============================================================================
# %% G21 - ERRO PERCENTUAL DE CADA PREVISÃO (FIGURA 11)
# =============================================================================
# Erro % = (previsto - real) / real. Se der negativo, o modelo previu menos
# do que vendeu (subestimou). Quase todas as barras ficaram negativas.
erro = pd.DataFrame({m: (teste[m] - teste["pecas"]) / teste["pecas"] * 100
                     for m in ["Baseline (t-1)", "Gradient Boosting"]})
erro.index = rotulos
fig, ax = plt.subplots(figsize=(15, 6))
erro.plot(kind="bar", ax=ax, color=[CORES["Baseline (t-1)"], CORES["Gradient Boosting"]])
ax.axhline(0, color="black", linewidth=1)
ax.set_title("Erro percentual da previsão de 2025 (valores negativos = subestimação)")
ax.set_xlabel("Segmento (unidade - marca) / coleção")
ax.set_ylabel("Erro (%)")
ax.tick_params(axis="x", rotation=0, labelsize=8)
ax.legend(title="Modelo")
salvar(fig, "G21_erro_percentual_vies")


# =============================================================================
# %% G22 - IMPORTÂNCIA DAS VARIÁVEIS NO RANDOM FOREST (FIGURA 12)
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(importancia.index, importancia.values, color=CORES["Random Forest"])
ax.set_title("Importância das variáveis no Random Forest (peças)")
ax.set_xlabel("Importância relativa")
ax.set_ylabel("Variável")
salvar(fig, "G22_importancia_variaveis_rf")


# =============================================================================
# PARTE B - FATURAMENTO (análise de sensibilidade)
# =============================================================================
# %% 9. MESMOS MODELOS, AGORA PREVENDO O FATURAMENTO
# =============================================================================
# Aqui a base é menor (unidade x coleção, sem marca), porque o faturamento
# não está separado por marca no relatório consolidado.
teste_fat, _, _ = prever(fat, "faturamento", ["unidade", "colecao"])
resultado_fat = tabela_metricas(teste_fat, "faturamento", NOMES)
print("\n=== DESEMPENHO - FATURAMENTO, TESTE EM 2025 ===")
print(resultado_fat.to_string(index=False))
print("\nObservado x previsto (R$):")
print(teste_fat[["unidade", "colecao", "faturamento", "Baseline (t-1)",
                 "Random Forest"]].round(2).to_string(index=False))


# =============================================================================
# %% G23 - MAPE DE PEÇAS X MAPE DE FATURAMENTO
# =============================================================================
comp = pd.DataFrame({"Peças": resultado.set_index("Modelo")["MAPE (%)"],
                     "Faturamento": resultado_fat.set_index("Modelo")["MAPE (%)"]}).loc[NOMES]
fig, ax = plt.subplots(figsize=(12, 6))
comp.plot(kind="bar", ax=ax, color=["#2980b9", "#c0392b"], width=0.8)
for barras in ax.containers:
    ax.bar_label(barras, fmt="%.1f", fontsize=9)
ax.set_title("MAPE por modelo: peças x faturamento - teste 2025")
ax.set_xlabel("Modelo")
ax.set_ylabel("MAPE (%)")
ax.tick_params(axis="x", rotation=15)
ax.legend(title="Alvo")
salvar(fig, "G23_mape_pecas_vs_faturamento")


# =============================================================================
# %% G24 - FATURAMENTO: OBSERVADO X PREVISTO
# =============================================================================
rot = (teste_fat["unidade"].str.title() + "\n" + teste_fat["colecao"]).values
x = np.arange(len(teste_fat))
fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(x - larg, teste_fat["faturamento"], larg, label="Observado 2025", color="#7f7f7f")
ax.bar(x, teste_fat["Baseline (t-1)"], larg, label="Baseline (t-1)",
       color=CORES["Baseline (t-1)"])
ax.bar(x + larg, teste_fat["Random Forest"], larg, label="Random Forest",
       color=CORES["Random Forest"])
ax.set_xticks(x)
ax.set_xticklabels(rot)
ax.set_title("Faturamento em 2025: observado x previsto")
ax.set_xlabel("Unidade / coleção")
ax.set_ylabel("Faturamento (R$)")
ax.yaxis.set_major_formatter(lambda v, p: f"R$ {v/1e6:,.0f} mi")
ax.legend()
salvar(fig, "G24_faturamento_observado_previsto")


# =============================================================================
# %% 10. SALVANDO TODAS AS TABELAS NUM EXCEL
# =============================================================================
# Assim eu consigo copiar as tabelas direto para o Word do TCC.
with pd.ExcelWriter(PASTA_RESULTADOS / "resultados_modelos.xlsx") as arq:
    resultado.round(2).to_excel(arq, sheet_name="desempenho_pecas", index=False)
    tabela2.round(0).to_excel(arq, sheet_name="observado_previsto", index=False)
    por_colecao.round(2).to_excel(arq, sheet_name="por_colecao", index=False)
    por_segmento.round(2).to_excel(arq, sheet_name="por_segmento", index=False)
    importancia.sort_values(ascending=False).to_excel(arq, sheet_name="importancia_rf")
    resultado_fat.round(2).to_excel(arq, sheet_name="desempenho_faturamento", index=False)
print("\nPronto! Resultados salvos em:", PASTA_RESULTADOS / "resultados_modelos.xlsx")
