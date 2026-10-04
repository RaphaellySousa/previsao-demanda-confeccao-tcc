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

Script 2 de 3 - 02_analise_exploratoria.py
===============================================================================

O que este script faz:
    Aqui eu faço a análise exploratória, ou seja, olho os dados com gráficos
    antes de partir para os modelos. Meu orientador pediu mais gráficos, então
    fiz um para cada coisa que eu afirmo no texto. Entre parênteses está o
    número da figura no TCC (os que não têm número ficaram só no código).

    G01  peças por coleção e ano ......................... (Figura 2)
    G02  participação % das coleções nas peças
    G03  faturamento total por ano e por unidade
    G04  faturamento por unidade e coleção
    G05  participação % do faturamento por coleção ....... (Figura 1)
    G06  peças por unidade e marca
    G07  sequência das coleções por segmento ............. (Figura 3)
    G08  variação de um ano para o outro por segmento .... (Figura 4)
    G09  preço médio por peça ............................ (Figura 13)
    G10  categorias que mais vendem por unidade .......... (Figura 5)
    G11  mapa de calor categoria x coleção ............... (Figura 6 = filial)
    G12  curva ABC das categorias ........................ (Figura 7)
    G13  tecidos que mais vendem
    G14  lançamentos x produtos continuados .............. (Figura 8)
    G15  boxplot das vendas por perfil de produto

Antes de rodar:
    É preciso ter rodado o 01_preparar_dados.py, porque este script lê as
    bases que ele salva na pasta "dados".

Como rodar no Spyder:
    F5. Os gráficos aparecem na aba "Plots" e também são salvos em PNG
    (300 dpi, boa resolução para colocar no Word) na pasta "graficos".
"""


# =============================================================================
# %% 0. BIBLIOTECAS, PASTAS E PADRÃO DOS GRÁFICOS
# =============================================================================
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt   # biblioteca principal dos gráficos
import seaborn as sns             # usei só para os mapas de calor e o boxplot

try:
    PASTA_PROJETO = Path(__file__).resolve().parent.parent
except NameError:
    PASTA_PROJETO = Path.cwd().parent

PASTA_DADOS = PASTA_PROJETO / "dados"
PASTA_GRAFICOS = PASTA_PROJETO / "graficos"
PASTA_GRAFICOS.mkdir(exist_ok=True)

# Cada coleção tem sempre a mesma cor em todos os gráficos, para ficar
# mais fácil de comparar uma figura com a outra.
COLECOES = ["Inverno", "Verão", "Alto Verão"]
CORES_COL = {"Inverno": "#1f77b4", "Verão": "#ff7f0e", "Alto Verão": "#d62728"}
FONTE = "Fonte: dados internos da empresa (2023-2025). Elaboração própria."

# Tamanho padrão das figuras e grade clarinha no fundo
plt.rcParams.update({"figure.figsize": (10, 6), "font.size": 11,
                     "axes.grid": True, "grid.alpha": 0.3})
pd.options.display.float_format = "{:,.2f}".format
pd.set_option("display.width", 150)


def salvar(fig, nome):
    """Coloca a fonte embaixo do gráfico, salva em PNG e mostra no Spyder.
    Fiz essa função para não ter que repetir as mesmas linhas toda hora."""
    fig.text(0.01, 0.005, FONTE, fontsize=8, style="italic")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(PASTA_GRAFICOS / f"{nome}.png", dpi=300)
    plt.show()


def milhares(x, pos=None):
    """Deixa o eixo mais fácil de ler: 1500000 vira '1.500 mil'."""
    return f"{x/1000:,.0f} mil".replace(",", ".")


def milhoes(x, pos=None):
    """Para valores em reais: 45000000 vira 'R$ 45 mi'."""
    return f"R$ {x/1e6:,.0f} mi"


# =============================================================================
# %% 1. LEITURA DAS BASES QUE O SCRIPT 01 SALVOU
# =============================================================================
pecas = pd.read_csv(PASTA_DADOS / "base_agregada_pecas.csv")
fat = pd.read_csv(PASTA_DADOS / "base_agregada_faturamento.csv")
det = pd.read_csv(PASTA_DADOS / "base_detalhada_2024_2025.csv")
ANOS = sorted(pecas["ano"].unique())
print("Anos na base:", ANOS)


# =============================================================================
# %% G01 - PEÇAS POR COLEÇÃO E ANO (FIGURA 2 DO TCC)
# =============================================================================
# Pergunta: qual coleção vende mais? A ordem se repete todo ano?
tab = pecas.pivot_table(index="ano", columns="colecao", values="pecas",
                        aggfunc="sum")[COLECOES]
print("\nPeças por coleção e ano:\n", tab)

fig, ax = plt.subplots()
tab.plot(kind="bar", ax=ax, color=[CORES_COL[c] for c in COLECOES], width=0.8)
# Escrevo o valor em cima de cada barra
for barras in ax.containers:
    ax.bar_label(barras, labels=[f"{v/1000:,.0f} mil".replace(",", ".") for v in barras.datavalues],
                 fontsize=9)
ax.set_title("Volume de vendas (peças) por coleção - 2023 a 2025")
ax.set_xlabel("Ano")
ax.set_ylabel("Peças vendidas")
ax.yaxis.set_major_formatter(milhares)
ax.tick_params(axis="x", rotation=0)
ax.legend(title="Coleção")
salvar(fig, "G01_pecas_por_colecao_ano")


# =============================================================================
# %% G02 - PARTICIPAÇÃO (%) DE CADA COLEÇÃO NAS PEÇAS
# =============================================================================
# Pergunta: o peso de cada coleção muda de um ano para o outro?
part = tab.div(tab.sum(axis=1), axis=0) * 100
print("\nParticipação % das coleções nas peças:\n", part)

fig, ax = plt.subplots()
part.plot(kind="bar", stacked=True, ax=ax, color=[CORES_COL[c] for c in COLECOES])
for barras in ax.containers:
    ax.bar_label(barras, labels=[f"{v:.1f}%" for v in barras.datavalues],
                 label_type="center", color="white", fontweight="bold")
ax.set_title("Participação das coleções no volume anual de peças")
ax.set_xlabel("Ano")
ax.set_ylabel("Participação (%)")
ax.set_ylim(0, 100)
ax.tick_params(axis="x", rotation=0)
ax.legend(title="Coleção", bbox_to_anchor=(1.02, 1), loc="upper left")
salvar(fig, "G02_participacao_colecoes_pecas")


# =============================================================================
# %% G03 - FATURAMENTO TOTAL POR ANO E QUANTO É DA MATRIZ
# =============================================================================
# No texto eu digo que a matriz tem uns 72% do faturamento. Este gráfico mostra isso.
fat_uni = fat.pivot_table(index="ano", columns="unidade", values="faturamento",
                          aggfunc="sum")[["MATRIZ", "FILIAL"]]
print("\nFaturamento por unidade e ano:\n", fat_uni)
print("Participação % da matriz:\n", fat_uni["MATRIZ"] / fat_uni.sum(axis=1) * 100)

fig, ax = plt.subplots()
fat_uni.plot(kind="bar", stacked=True, ax=ax, color=["#2c3e50", "#95a5a6"])
total = fat_uni.sum(axis=1)
for i, ano in enumerate(fat_uni.index):
    pct = fat_uni.loc[ano, "MATRIZ"] / total[ano] * 100
    ax.text(i, total[ano] * 1.01, f"R$ {total[ano]/1e6:,.1f} mi\n(matriz {pct:.0f}%)",
            ha="center", fontsize=10)
ax.set_title("Faturamento total por ano e por unidade")
ax.set_xlabel("Ano")
ax.set_ylabel("Faturamento (R$)")
ax.yaxis.set_major_formatter(milhoes)
ax.set_ylim(0, total.max() * 1.15)
ax.tick_params(axis="x", rotation=0)
ax.legend(title="Unidade")
salvar(fig, "G03_faturamento_total_unidade")


# =============================================================================
# %% G04 - FATURAMENTO POR UNIDADE E COLEÇÃO (MESMOS DADOS DA TABELA 1)
# =============================================================================
# Fiz um gráfico do lado do outro: matriz à esquerda e filial à direita.
fig, axes = plt.subplots(1, 2, figsize=(13, 6))
for ax, uni in zip(axes, ["MATRIZ", "FILIAL"]):
    t = fat[fat["unidade"] == uni].pivot_table(index="ano", columns="colecao",
                                               values="faturamento")[COLECOES]
    t.plot(kind="bar", ax=ax, color=[CORES_COL[c] for c in COLECOES], width=0.8)
    ax.set_title(uni.title())
    ax.set_xlabel("Ano")
    ax.set_ylabel("Faturamento (R$)")
    ax.yaxis.set_major_formatter(milhoes)
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="Coleção", fontsize=9)
fig.suptitle("Faturamento por unidade e coleção - 2023 a 2025")
salvar(fig, "G04_faturamento_unidade_colecao")


# =============================================================================
# %% G05 - PARTICIPAÇÃO (%) DO FATURAMENTO POR COLEÇÃO EM CADA UNIDADE (FIGURA 1)
# =============================================================================
# Pergunta: a divisão entre as coleções é estável? (no texto: 43%/18%/39% etc.)
pf = fat.pivot_table(index=["unidade", "ano"], columns="colecao",
                     values="faturamento")[COLECOES]
pf = pf.div(pf.sum(axis=1), axis=0) * 100
pf = pf.loc[["MATRIZ", "FILIAL"]]
print("\nParticipação % do faturamento por coleção:\n", pf)

fig, ax = plt.subplots(figsize=(11, 6))
pf.plot(kind="bar", stacked=True, ax=ax, color=[CORES_COL[c] for c in COLECOES])
for barras in ax.containers:
    ax.bar_label(barras, labels=[f"{v:.0f}%" for v in barras.datavalues],
                 label_type="center", color="white", fontweight="bold")
ax.set_xticklabels([f"{u.title()}\n{a}" for u, a in pf.index], rotation=0)
ax.set_title("Participação das coleções no faturamento anual, por unidade")
ax.set_xlabel("Unidade / ano")
ax.set_ylabel("Participação (%)")
ax.set_ylim(0, 100)
ax.legend(title="Coleção", bbox_to_anchor=(1.02, 1), loc="upper left")
salvar(fig, "G05_participacao_colecoes_faturamento")


# =============================================================================
# %% G06 - PEÇAS POR UNIDADE E MARCA
# =============================================================================
# Pergunta: a Marca A vende mais que a B em todos os anos?
# Criei a coluna "segmento" juntando unidade e marca (ex.: "Matriz - Marca A").
pecas["segmento"] = pecas["unidade"].str.title() + " - Marca " + pecas["marca"]
seg = pecas.pivot_table(index="ano", columns="segmento", values="pecas", aggfunc="sum")
seg = seg[["Matriz - Marca A", "Matriz - Marca B", "Filial - Marca A", "Filial - Marca B"]]
print("\nPeças por segmento:\n", seg)

fig, ax = plt.subplots(figsize=(11, 6))
seg.plot(kind="bar", ax=ax, width=0.8,
         color=["#08519c", "#6baed6", "#a50f15", "#fb6a4a"])
for barras in ax.containers:
    ax.bar_label(barras, labels=[f"{v/1000:,.0f}".replace(",", ".") for v in barras.datavalues],
                 fontsize=8)
ax.set_title("Volume de peças por unidade e marca (valores em mil peças)")
ax.set_xlabel("Ano")
ax.set_ylabel("Peças vendidas")
ax.yaxis.set_major_formatter(milhares)
ax.tick_params(axis="x", rotation=0)
ax.legend(title="Segmento")
salvar(fig, "G06_pecas_unidade_marca")


# =============================================================================
# %% G07 - COLEÇÕES EM SEQUÊNCIA, POR SEGMENTO (FIGURA 3)
# =============================================================================
# Coloquei as 9 coleções uma atrás da outra (Inv/23, Ver/23, AV/23, ...)
# para ver o "desenho" que se repete todo ano. Este gráfico também mostra
# por que eu não usei ARIMA ou Holt-Winters: cada segmento tem só 9 pontos.
pecas["periodo"] = pecas["colecao"].map({"Inverno": "Inv", "Verão": "Ver",
                                         "Alto Verão": "AV"}) + "/" + pecas["ano"].astype(str).str[2:]
pecas["ordem"] = pecas["ano"] * 10 + pecas["colecao"].map({c: i for i, c in enumerate(COLECOES)})
serie = pecas.pivot_table(index=["ordem", "periodo"], columns="segmento", values="pecas")
serie = serie[seg.columns]

fig, ax = plt.subplots(figsize=(12, 6))
for s, cor in zip(serie.columns, ["#08519c", "#6baed6", "#a50f15", "#fb6a4a"]):
    ax.plot(range(len(serie)), serie[s], marker="o", label=s, color=cor, linewidth=2)
# Linhas tracejadas separando os anos
for x in [2.5, 5.5]:
    ax.axvline(x, color="grey", linestyle="--", linewidth=1)
ax.set_xticks(range(len(serie)))
ax.set_xticklabels(serie.index.get_level_values("periodo"))
ax.set_title("Peças vendidas em sequência de coleções, por segmento")
ax.set_xlabel("Coleção / ano (Inv = Inverno, Ver = Verão, AV = Alto Verão)")
ax.set_ylabel("Peças vendidas")
ax.yaxis.set_major_formatter(milhares)
ax.legend(title="Segmento", bbox_to_anchor=(1.02, 1), loc="upper left")
salvar(fig, "G07_sequencia_colecoes_segmento")


# =============================================================================
# %% G08 - VARIAÇÃO (%) EM RELAÇÃO À MESMA COLEÇÃO DO ANO ANTERIOR (FIGURA 4)
# =============================================================================
# Pergunta: todos os segmentos cresceram igual? (Spoiler: não, a Filial A
# caiu em 2024 e subiu muito em 2025.)
pecas_ord = pecas.sort_values("ano")
pecas_ord["variacao_%"] = (pecas_ord.groupby(["segmento", "colecao"])["pecas"]
                           .pct_change() * 100)
var = pecas_ord.dropna(subset=["variacao_%"]).pivot_table(
    index="segmento", columns=["ano", "colecao"], values="variacao_%")
var = var.reindex(seg.columns)
var = var[[(a, c) for a in ANOS[1:] for c in COLECOES]]
print("\nVariação % em relação à mesma coleção do ano anterior:\n", var)

fig, ax = plt.subplots(figsize=(12, 5))
# Mapa de calor: verde = cresceu, vermelho = caiu
sns.heatmap(var, annot=True, fmt=".1f", cmap="RdYlGn", center=0, ax=ax,
            cbar_kws={"label": "Variação (%)"})
ax.grid(False)
ax.set_xticklabels([f"{c}\n{a}" for a, c in var.columns], rotation=0)
ax.set_title("Variação das vendas (%) em relação à mesma coleção do ano anterior")
ax.set_xlabel("Coleção / ano")
ax.set_ylabel("Segmento")
salvar(fig, "G08_variacao_anual_segmento")


# =============================================================================
# %% G09 - PREÇO MÉDIO POR PEÇA (FIGURA 13)
# =============================================================================
# Dividi o faturamento pelas peças. Serve para explicar por que prever o
# faturamento é mais difícil: o preço médio também muda de um ano para outro.
pecas_uni = pecas.groupby(["ano", "unidade", "colecao"], as_index=False)["pecas"].sum()
preco = fat.merge(pecas_uni, on=["ano", "unidade", "colecao"])
preco["preco_medio"] = preco["faturamento"] / preco["pecas"]
print("\nPreço médio (R$ por peça):\n",
      preco.pivot_table(index=["unidade", "ano"], columns="colecao", values="preco_medio")[COLECOES])

fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
for ax, uni in zip(axes, ["MATRIZ", "FILIAL"]):
    t = preco[preco["unidade"] == uni].pivot_table(index="ano", columns="colecao",
                                                   values="preco_medio")[COLECOES]
    for c in COLECOES:
        ax.plot(t.index, t[c], marker="o", label=c, color=CORES_COL[c], linewidth=2)
    ax.set_title(uni.title())
    ax.set_xlabel("Ano")
    ax.set_xticks(ANOS)
    ax.set_ylabel("R$ por peça")
    ax.legend(title="Coleção")
fig.suptitle("Preço médio implícito (faturamento / peças) por unidade e coleção")
salvar(fig, "G09_preco_medio_implicito")


# =============================================================================
# DAQUI PARA BAIXO uso a base detalhada (só 2024 e 2025), que tem categoria
# e tecido de cada produto.
# =============================================================================
FONTE = "Fonte: dados internos da empresa (2024-2025). Elaboração própria."


# =============================================================================
# %% G10 - CATEGORIAS QUE MAIS VENDEM EM CADA UNIDADE (FIGURA 5)
# =============================================================================
# Pergunta: o que a matriz vende é diferente do que a filial vende?
fig, axes = plt.subplots(1, 2, figsize=(14, 7))
for ax, uni in zip(axes, ["MATRIZ", "FILIAL"]):
    t = (det[det["unidade"] == uni].pivot_table(index="categoria", columns="ano",
                                                 values="pecas", aggfunc="sum")
         .fillna(0))
    # Pego só as 10 maiores para o gráfico não ficar poluído
    t = t.loc[t.sum(axis=1).sort_values(ascending=False).index[:10]]
    print(f"\nTop 10 categorias - {uni}:\n", t)
    t.iloc[::-1].plot(kind="barh", ax=ax, color=["#9ecae1", "#08519c"])
    ax.set_title(uni.title())
    ax.set_xlabel("Peças vendidas")
    ax.set_ylabel("Categoria")
    ax.xaxis.set_major_formatter(milhares)
    ax.legend(title="Ano")
fig.suptitle("Dez categorias de maior volume por unidade - 2024 e 2025")
salvar(fig, "G10_top_categorias_unidade")


# =============================================================================
# %% G11 - MAPA DE CALOR CATEGORIA X COLEÇÃO (FIGURA 6 É O DA FILIAL)
# =============================================================================
# Pergunta: em qual coleção cada categoria vende mais? Aqui aparece que as
# jaquetas da filial vendem quase só no Inverno.
for uni in ["MATRIZ", "FILIAL"]:
    d = det[det["unidade"] == uni].copy()
    d["periodo"] = d["colecao"] + "\n" + d["ano"].astype(str)
    t = d.pivot_table(index="categoria", columns="periodo", values="pecas",
                      aggfunc="sum").fillna(0)
    ordem = [f"{c}\n{a}" for a in [2024, 2025] for c in COLECOES]
    t = t[[o for o in ordem if o in t.columns]]
    t = t.loc[t.sum(axis=1).sort_values(ascending=False).index[:12]]

    fig, ax = plt.subplots(figsize=(11, 7))
    sns.heatmap(t / 1000, annot=True, fmt=".1f", cmap="Blues", ax=ax,
                cbar_kws={"label": "Mil peças"})
    ax.grid(False)
    ax.set_title(f"{uni.title()}: peças vendidas por categoria e coleção (mil peças)")
    ax.set_xlabel("Coleção / ano")
    ax.set_ylabel("Categoria")
    ax.tick_params(axis="x", rotation=0)
    salvar(fig, f"G11_heatmap_categoria_colecao_{uni.lower()}")


# =============================================================================
# %% G12 - CURVA ABC DAS CATEGORIAS EM 2025 (FIGURA 7)
# =============================================================================
# A ideia da curva ABC (Pareto) é ver quantas categorias fazem a maior parte
# das vendas. Classe A = até 80% acumulado, B = até 95%, C = o resto.
abc = (det[det["ano"] == 2025].groupby("categoria")["pecas"].sum()
       .sort_values(ascending=False).to_frame())
abc["% acumulado"] = abc["pecas"].cumsum() / abc["pecas"].sum() * 100
abc["classe"] = np.where(abc["% acumulado"] <= 80, "A",
                         np.where(abc["% acumulado"] <= 95, "B", "C"))
print("\nCurva ABC das categorias (2025):\n", abc)

fig, ax = plt.subplots(figsize=(12, 6))
cores_abc = abc["classe"].map({"A": "#08519c", "B": "#6baed6", "C": "#c6dbef"})
ax.bar(abc.index, abc["pecas"], color=cores_abc)
ax.set_ylabel("Peças vendidas")
ax.yaxis.set_major_formatter(milhares)
ax.set_xlabel("Categoria")
ax.tick_params(axis="x", rotation=60)
# Segundo eixo (à direita) com a linha do % acumulado
ax2 = ax.twinx()
ax2.plot(abc.index, abc["% acumulado"], color="#d62728", marker="o")
ax2.axhline(80, color="grey", linestyle="--")
ax2.set_ylim(0, 105)
ax2.set_ylabel("Participação acumulada (%)")
ax2.grid(False)
ax.set_title("Curva ABC das categorias - peças vendidas em 2025 (matriz + filial)")
salvar(fig, "G12_curva_abc_categorias")


# =============================================================================
# %% G13 - TECIDOS QUE MAIS VENDEM
# =============================================================================
tec = (det.pivot_table(index="tecido", columns="ano", values="pecas", aggfunc="sum")
       .fillna(0))
tec = tec.loc[tec.sum(axis=1).sort_values(ascending=False).index[:12]]
print("\nTecidos de maior volume:\n", tec)

fig, ax = plt.subplots(figsize=(11, 7))
tec.iloc[::-1].plot(kind="barh", ax=ax, color=["#9ecae1", "#08519c"])
ax.set_title("Doze grupos de tecido de maior volume - 2024 e 2025")
ax.set_xlabel("Peças vendidas")
ax.set_ylabel("Tecido")
ax.xaxis.set_major_formatter(milhares)
ax.legend(title="Ano")
salvar(fig, "G13_tecidos_volume")


# =============================================================================
# %% G14 - LANÇAMENTOS X CONTINUADAS EM CADA COLEÇÃO (FIGURA 8)
# =============================================================================
# Quis saber quanto das vendas vem de produto novo (lançamento). Produto
# novo não tem histórico, então nenhum modelo baseado no passado consegue
# prever ele direito. Isso é uma limitação que discuto no TCC.
orig = pd.read_csv(PASTA_DADOS / "base_origem_produto.csv")
orig["origem_produto"] = orig["origem_produto"].str.title()
t = orig.pivot_table(index=["ano", "colecao"], columns="origem_produto",
                     values="pecas", aggfunc="sum")
t = t.div(t.sum(axis=1), axis=0) * 100
t = t.loc[[(a, c) for a in [2024, 2025] for c in COLECOES]]
print("\nParticipação % por origem do produto:\n", t)

fig, ax = plt.subplots(figsize=(11, 6))
t.plot(kind="bar", stacked=True, ax=ax, color=["#2ca02c", "#ff7f0e", "#7f7f7f"])
for barras in ax.containers:
    ax.bar_label(barras, labels=[f"{v:.0f}%" if v > 4 else "" for v in barras.datavalues],
                 label_type="center", color="white", fontweight="bold")
ax.set_xticklabels([f"{c}\n{a}" for a, c in t.index], rotation=0)
ax.set_title("Participação das peças por origem do produto em cada coleção")
ax.set_xlabel("Coleção / ano")
ax.set_ylabel("Participação nas peças (%)")
ax.set_ylim(0, 100)
ax.legend(title="Origem", bbox_to_anchor=(1.02, 1), loc="upper left")
salvar(fig, "G14_lancamentos_continuadas")


# =============================================================================
# %% G15 - BOXPLOT DAS VENDAS POR PERFIL DE PRODUTO (OUTLIERS)
# =============================================================================
# Perfil = unidade x marca x categoria x tecido x coleção.
# Usei escala logarítmica porque tem perfil com 50 peças e perfil com quase
# 200 mil, e na escala normal a caixa ficava achatada.
d = det[det["pecas"] > 0].copy()
print("\nResumo das peças por perfil:\n",
      d.groupby("colecao")["pecas"].describe()[["count", "mean", "50%", "max"]])

fig, ax = plt.subplots(figsize=(11, 6))
sns.boxplot(data=d, x="colecao", y="pecas", hue="ano", order=COLECOES,
            palette=["#9ecae1", "#08519c"], ax=ax)
ax.set_yscale("log")
ax.set_title("Distribuição das peças vendidas por perfil de produto (escala log)")
ax.set_xlabel("Coleção")
ax.set_ylabel("Peças por perfil (escala logarítmica)")
ax.legend(title="Ano")
salvar(fig, "G15_boxplot_perfis")

print("\nPronto! Gráficos salvos em:", PASTA_GRAFICOS)
