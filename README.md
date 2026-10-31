# Análise exploratória de churn

Miniprojeto da disciplina de **Sistemas Inteligentes — CIn/UFPE**, voltado à análise exploratória de uma base de churn. Os resultados orientarão o pré-processamento e projetos futuros com algoritmos supervisionados e não supervisionados.

O trabalho atual compreende apenas a contextualização e a compreensão inicial da base. As demais partes da EDA estão planejadas no notebook e aguardam revisão. Ainda não há pré-processamento nem modelos implementados.

## Estrutura

```text
.
├── analise_base_churn.ipynb
├── base_churn_5000_VF2_aluno.xlsx - Dados_Sinteticos_VF2.csv
├── requirements.txt
├── .gitignore
└── README.md
```

- `analise_base_churn.ipynb`: roteiro da EDA, análises implementadas, saídas e interpretações.
- `base_churn_5000_VF2_aluno.xlsx - Dados_Sinteticos_VF2.csv`: arquivo original, preservado. Seu formato real é CSV, apesar de conter `.xlsx` no nome.
- `requirements.txt`: dependências diretas, nas versões utilizadas na validação inicial.
- `.gitignore`: impede a inclusão do ambiente virtual, arquivos temporários e configurações locais em novos commits.

## Execução no VS Code

O notebook foi validado com **Python 3.10.12**. Os comandos abaixo devem ser executados na raiz do projeto, em um terminal Linux/macOS.

Se o ambiente `.venv` já estiver configurado, utilize-o. Em uma nova cópia do projeto, crie-o uma única vez:

```bash
python3 -m venv .venv
```

Instale as dependências nesse ambiente:

```bash
.venv/bin/python -m pip install -r requirements.txt
```

No VS Code, utilize as extensões Python e Jupyter. Abra `analise_base_churn.ipynb`, selecione o ambiente `.venv` como kernel e execute todas as células na ordem apresentada. Mantenha a raiz do projeto como diretório de trabalho do notebook.

No Windows, o executável correspondente é `.venv\Scripts\python.exe`.

## Leitura e preservação dos dados

O notebook localiza o CSV por um caminho relativo:

```python
caminho_base = Path("base_churn_5000_VF2_aluno.xlsx - Dados_Sinteticos_VF2.csv")
```

Assim, a leitura independe do nome do usuário ou da localização em que o projeto foi clonado. Mantenha o nome original do arquivo e sua posição ao lado do notebook.

A leitura utiliza UTF-8, separador de campos `,` e vírgula decimal. A base contém **5.020 linhas e 10 colunas**. `data_ocorrencia_churn` foi identificada como origem provável da variável-alvo; sua definição definitiva depende da confirmação das regras temporais e do significado dos campos vazios.

As análises são realizadas em memória, sem sobrescrever a base original. Nesta etapa, não há imputação, remoção de registros ou outliers, codificação, normalização ou treinamento de modelos.

## Roteiro da EDA

1. Introdução e contextualização — implementada.
2. Carregamento e compreensão inicial — implementada.
3. Qualidade e consistência dos dados — planejada.
4. Análise univariada e investigação de valores extremos — planejada.
5. Compreensão da variável-alvo — planejada.
6. Análise bivariada — planejada.
7. Correlações entre atributos — planejada.
8. Investigações complementares justificadas pelos resultados — planejada.
9. Síntese e encaminhamentos para o pré-processamento — planejada.

## Referências de organização e análise

- [CIN0024 — Churn Analysis System, branch feature/churn-eda-preprocessing](https://github.com/llucasEmanuel/CIN0024-churn-analysis-system/tree/feature/churn-eda-preprocessing).
- [Projeto Lígia — Grupo 10](https://github.com/victoriapessoabm/PS_Ligia_Grupo_10).

As referências orientam a estrutura do trabalho. As interpretações do notebook são baseadas no arquivo local; resultados e regras dos outros projetos não são assumidos como válidos para esta base.
