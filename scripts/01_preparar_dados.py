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

Script 1 de 3 - 01_preparar_dados.py
===============================================================================

O que este script faz:
    Esta é a primeira etapa do meu trabalho. Aqui eu leio o relatório de
    vendas da empresa ("RESUMOS 2023 - 2025.xlsx"), arrumo os nomes, confiro
    se os dados têm problemas e monto as bases que vou usar nos outros dois
    scripts. No final eu confiro se os totais batem com as Tabelas 1 e 2 do
    TCC.

Bases que saem daqui (pasta "dados"):
    - base_detalhada_2024_2025.csv  -> ano x coleção x unidade x marca x
                                       categoria x tecido (usada nos gráficos)
    - base_agregada_pecas.csv       -> ano x unidade x marca x coleção
    - base_agregada_faturamento.csv -> ano x unidade x coleção
    - base_origem_produto.csv       -> lançamento, continuada e sobras

Uma observação importante:
    O Excel só tem as abas de 2024 e 2025. Os totais de 2023 eu digitei a
    partir do relatório consolidado (são os mesmos valores das Tabelas 1 e 2
    do TCC). Eles ficam marcados na coluna "origem".

Como rodar no Spyder:
    Abrir este arquivo e apertar F5. Os resultados aparecem no console.
"""


# =============================================================================
# %% 0. BIBLIOTECAS E PASTAS
# =============================================================================
# pathlib serve para montar os caminhos das pastas sem depender de "C:\..."
# e pandas é a biblioteca que eu uso para trabalhar com as tabelas.
from pathlib import Path
import pandas as pd

# Descobre onde está a pasta do projeto (uma pasta acima de "scripts").
# O try/except é porque, se eu rodar só uma célula no Spyder, às vezes
# a variável __file__ não existe.
try:
    PASTA_PROJETO = Path(__file__).resolve().parent.parent
except NameError:
    PASTA_PROJETO = Path.cwd().parent

# O Excel fica na pasta "Base de dados", dentro da pasta do projeto:
#   TCC 6.0 - Codigo simples/
#     Base de dados/RESUMOS 2023 - 2025.xlsx
#     scripts/01_preparar_dados.py
# Assim, se eu copiar a pasta do projeto para outro computador, o Excel vai junto.
# (Se não achar ali, o script também procura na pasta de cima, por garantia.)
NOME_EXCEL = "RESUMOS 2023 - 2025.xlsx"
lugares = [PASTA_PROJETO / "Base de dados" / NOME_EXCEL,
           PASTA_PROJETO.parent / "Base de dados" / NOME_EXCEL]
ARQUIVO_RESUMOS = next((c for c in lugares if c.exists()), lugares[0])

# Só para os números aparecerem mais fáceis de ler no console
pd.options.display.float_format = "{:,.2f}".format
pd.set_option("display.width", 150)

# Pasta onde vou salvar as bases tratadas (cria se não existir)
PASTA_SAIDA = PASTA_PROJETO / "dados"
PASTA_SAIDA.mkdir(exist_ok=True)

# Ordem das coleções dentro do ano
COLECOES = ["Inverno", "Verão", "Alto Verão"]

# No Excel as marcas aparecem com código. Para não identificar a empresa,
# troquei pelos nomes "A" (a que vende mais) e "B".
MARCAS = {4: "A", 2: "B"}

print("Arquivo de dados:", ARQUIVO_RESUMOS)
print("O arquivo existe?", ARQUIVO_RESUMOS.exists())


# =============================================================================
# %% 1. LEITURA DAS ABAS DO EXCEL
# =============================================================================
xls = pd.ExcelFile(ARQUIVO_RESUMOS)
print("Abas do arquivo:", xls.sheet_names)

lista = []
for aba in xls.sheet_names:
    df = pd.read_excel(xls, sheet_name=aba)
    # Cada aba tem uns quadros de resumo do lado esquerdo. A tabela que me
    # interessa (uma linha por produto) começa na coluna "Empresa", então
    # eu corto tudo o que vem antes dela.
    inicio = list(df.columns).index("Empresa")
    df = df.iloc[:, inicio:]
    df["ano"] = int(aba)          # o nome da aba é o ano
    lista.append(df)

# Junta 2024 e 2025 numa tabela só
bruto = pd.concat(lista, ignore_index=True)
print("\nQuantidade de linhas lidas por ano:")
print(bruto["ano"].value_counts().sort_index())


# =============================================================================
# %% 2. PADRONIZAÇÃO DAS COLUNAS
# =============================================================================
# O Excel tem duas colunas chamadas "Categoria". O pandas renomeia a
# segunda para "Categoria.1". Eu usei a segunda porque ela traz o nome do
# produto mesmo (ex.: JAQUETA), enquanto a primeira junta vários produtos
# (ex.: AGASALHO).
base = pd.DataFrame({
    "ano":            bruto["ano"],
    "unidade":        bruto["Empresa"].str.strip().str.upper(),
    "marca":          bruto["Marca"].map(MARCAS),
    "temporada":      bruto["Temporada (Coleção Venda)"].str.strip().str.upper(),
    "categoria":      bruto["Categoria.1"].str.strip().str.upper(),
    "tecido":         bruto["Tecido"].str.strip().str.upper(),
    "origem_produto": bruto["Origem"].str.strip().str.upper(),
    "referencia":     bruto["Referência"],
    "pecas":          bruto["Quantidade"],
    "faturamento":    bruto["ROB"],     # ROB = receita operacional bruta (R$)
})

# A temporada vem como "INVERNO 2024". Eu tiro o ano e deixo só o nome da
# coleção, com a primeira letra maiúscula: "Inverno", "Verão", "Alto Verão".
base["colecao"] = (base["temporada"].str.replace(r"\s*\d{4}", "", regex=True)
                   .str.title().str.replace("Verao", "Verão"))
base = base.drop(columns="temporada")

# Dicionário de categorias (é o "dicionário" que cito no TCC).
# Percebi que em 2025 a matriz passou a lançar jaquetas, blusões, coletes e
# tricôs como "AGASALHO", mas em 2024 cada um tinha o seu nome. Se eu não
# juntasse, parecia que esses produtos tinham sumido de um ano para o outro.
# Só juntei o que eu tinha certeza de que era a mesma coisa.
DICIONARIO_CATEGORIAS = {
    "JAQUETA": "AGASALHO/JAQUETA",
    "BLUSAO": "AGASALHO/JAQUETA",
    "COLETE": "AGASALHO/JAQUETA",
    "TRICOT LEVE": "AGASALHO/JAQUETA",
    "AGASALHO": "AGASALHO/JAQUETA",
    "BERMUDA LINHO": "BERMUDA",
}
print("\nCategorias por ano ANTES de juntar os nomes (peças):")
print(base.pivot_table(index="categoria", columns="ano", values="pecas",
                       aggfunc="sum").fillna(0).round(0))
base["categoria"] = base["categoria"].replace(DICIONARIO_CATEGORIAS)


# =============================================================================
# %% 3. VERIFICANDO A QUALIDADE DOS DADOS
# =============================================================================
# Antes de fazer qualquer análise eu quis saber se os dados tinham
# problemas. Esses números estão descritos na seção de Implementação.
print("\n=== DIAGNÓSTICO DE QUALIDADE ===")
print("Valores vazios em cada coluna:")
print(base.isna().sum())

print("\nLinhas com quantidade igual ou menor que zero:", (base["pecas"] <= 0).sum())
print("Linhas com quantidade quebrada (com vírgula):", (base["pecas"] % 1 != 0).sum())
print("Linhas sem faturamento (ROB vazio):", base["faturamento"].isna().sum())
print("Marcas que não consegui identificar:", base["marca"].isna().sum())
print("Coleções encontradas:", sorted(base["colecao"].unique()))

# O que eu decidi fazer com cada problema:
#  - ROB vazio vira 0, porque o produto não teve receita naquela linha;
#  - quantidades quebradas (aparecem em 2025) ficam como estão na base
#    detalhada e só arredondo no total, igual nas tabelas do TCC;
#  - NÃO apaguei nenhuma linha, porque cada linha é um produto diferente.
base["faturamento"] = base["faturamento"].fillna(0)

# Conferi se tinha produto repetido na mesma coleção e unidade
print("\nProdutos repetidos (mesmo ano, unidade, coleção e referência):",
      base.duplicated(["ano", "unidade", "colecao", "referencia"]).sum())


# =============================================================================
# %% 4. BASE DETALHADA (PERFIL DE PRODUTO = CATEGORIA X TECIDO)
# =============================================================================
# Essa base eu uso nos gráficos de categoria e tecido do script 02.
detalhada = (base.groupby(["ano", "colecao", "unidade", "marca",
                           "categoria", "tecido"], as_index=False)
             [["pecas", "faturamento"]].sum())
detalhada.to_csv(PASTA_SAIDA / "base_detalhada_2024_2025.csv",
                 index=False, encoding="utf-8-sig")
print("\nBase detalhada (perfis de produto):", detalhada.shape)

# Peças por origem do produto (lançamento, continuada, sobras), usada no G14
origem = (base.groupby(["ano", "colecao", "origem_produto"], as_index=False)
          ["pecas"].sum())
origem.to_csv(PASTA_SAIDA / "base_origem_produto.csv", index=False, encoding="utf-8-sig")


# =============================================================================
# %% 5. VALORES DE 2023 (DIGITADOS DO RELATÓRIO CONSOLIDADO)
# =============================================================================
# Como o Excel não tem a aba de 2023, eu digitei aqui os totais por
# unidade, marca e coleção. Se um dia a aba 2023 aparecer, é só colocar no
# Excel que o script passa a usar ela no lugar destes números.
pecas_2023 = pd.DataFrame({
    "ano": 2023,
    "unidade": ["MATRIZ"] * 3 + ["FILIAL"] * 3 + ["MATRIZ"] * 3 + ["FILIAL"] * 3,
    "marca":   ["A"] * 6 + ["B"] * 6,
    "colecao": COLECOES * 4,
    "pecas": [347057, 143211, 402691,     # Matriz - Marca A
              75076, 32095, 86392,        # Filial - Marca A
              224666, 139579, 257677,     # Matriz - Marca B
              48387, 33236, 59664],       # Filial - Marca B
})
fat_2023 = pd.DataFrame({
    "ano": 2023,
    "unidade": ["MATRIZ"] * 3 + ["FILIAL"] * 3,
    "colecao": COLECOES * 2,
    "faturamento": [40590223.86, 16756951.33, 36529623.84,
                    13650584.10, 7250573.79, 15741419.97],
})
pecas_2023["origem"] = "Tabela do TCC"
fat_2023["origem"] = "Tabela do TCC"


# =============================================================================
# %% 6. BASES AGREGADAS (SÃO AS QUE ENTRAM NOS MODELOS)
# =============================================================================
# Somo as peças por ano, unidade, marca e coleção
pecas = base.groupby(["ano", "unidade", "marca", "colecao"],
                     as_index=False)["pecas"].sum()
pecas["pecas"] = pecas["pecas"].round(0)
pecas["origem"] = "RESUMOS"

# E o faturamento por ano, unidade e coleção
fat = base.groupby(["ano", "unidade", "colecao"],
                   as_index=False)["faturamento"].sum()
fat["faturamento"] = fat["faturamento"].round(2)
fat["origem"] = "RESUMOS"

# Junto 2023 só se ele ainda não estiver no Excel
if 2023 not in pecas["ano"].values:
    pecas = pd.concat([pecas_2023, pecas], ignore_index=True)
    fat = pd.concat([fat_2023, fat], ignore_index=True)

# Deixo as linhas sempre na mesma ordem (ano, marca, unidade, coleção).
# Descobri que isso é importante: a ordem das linhas muda um pouco o
# resultado do Random Forest, então com a ordem fixa o resultado sempre
# sai igual.
ordem_uni = {"MATRIZ": 0, "FILIAL": 1}
ordem_col = {c: i for i, c in enumerate(COLECOES)}
pecas = (pecas.assign(u=pecas["unidade"].map(ordem_uni), c=pecas["colecao"].map(ordem_col))
         .sort_values(["ano", "marca", "u", "c"]).drop(columns=["u", "c"])
         .reset_index(drop=True))
fat = (fat.assign(u=fat["unidade"].map(ordem_uni), c=fat["colecao"].map(ordem_col))
       .sort_values(["ano", "u", "c"]).drop(columns=["u", "c"]).reset_index(drop=True))

pecas.to_csv(PASTA_SAIDA / "base_agregada_pecas.csv", index=False, encoding="utf-8-sig")
fat.to_csv(PASTA_SAIDA / "base_agregada_faturamento.csv", index=False, encoding="utf-8-sig")


# =============================================================================
# %% 7. CONFERINDO COM AS TABELAS 1 E 2 DO TCC
# =============================================================================
print("\n=== TABELA 2 DO TCC - peças por unidade, marca e coleção ===")
tab2 = pecas.pivot_table(index=["ano", "unidade", "marca"], columns="colecao",
                         values="pecas", aggfunc="sum")[COLECOES]
tab2["Total"] = tab2.sum(axis=1)
print(tab2.astype(int).to_string())

print("\n=== TABELA 1 DO TCC - faturamento (R$) por unidade e coleção ===")
tab1 = fat.pivot_table(index=["ano", "unidade"], columns="colecao",
                       values="faturamento", aggfunc="sum")[COLECOES]
tab1["Total"] = tab1.sum(axis=1)
print(tab1.round(2).to_string())

# Totais por ano que eu cito no texto
print("\nFaturamento total por ano:")
print(fat.groupby("ano")["faturamento"].sum().round(2))
print("\nPeças totais por ano:")
print(pecas.groupby("ano")["pecas"].sum())

# Confiro automaticamente os anos que vieram do Excel (2024 e 2025).
# Se aparecer "DIFERENTE", tem alguma coisa errada na leitura.
tcc_2024_2025 = {
    (2024, "MATRIZ", "A"): 1014134, (2024, "FILIAL", "A"): 184352,
    (2024, "MATRIZ", "B"): 709883,  (2024, "FILIAL", "B"): 141737,
    (2025, "MATRIZ", "A"): 1103285, (2025, "FILIAL", "A"): 215977,
    (2025, "MATRIZ", "B"): 758778,  (2025, "FILIAL", "B"): 154409,
}
print("\nConferência dos totais (Excel x Tabela 2 do TCC):")
for chave, valor_tcc in tcc_2024_2025.items():
    valor_dados = tab2.loc[chave, "Total"]
    situacao = "OK" if abs(valor_dados - valor_tcc) <= 2 else "DIFERENTE"
    print(f"  {chave}: dados = {valor_dados:,.0f} | TCC = {valor_tcc:,.0f} -> {situacao}")

print("\nPronto! Arquivos salvos em:", PASTA_SAIDA)
