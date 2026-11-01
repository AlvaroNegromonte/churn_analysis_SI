# Análise exploratória de churn

Miniprojeto da disciplina de **Sistemas Inteligentes — CIn/UFPE**, voltado à análise exploratória de uma base de churn. Os resultados orientarão o pré-processamento e projetos futuros com algoritmos supervisionados e não supervisionados.

A EDA está concluída nas Partes 1–9: contexto, inspeção, qualidade, consolidação temporal exploratória, distribuições, alvo, associações, correlações e implicações para as etapas futuras. As derivações ficam em memória, com rastreabilidade. Ainda não há pré-processamento definitivo nem modelos implementados.

## Referência principal e regras temporais

As Aulas 1 e 2 do professor **Germano Vasconcelos (CIn/UFPE)**, conforme as definições fornecidas no projeto, orientam as decisões. Os repositórios externos listados ao final são referências secundárias de organização. Os PDFs das aulas não estão incluídos neste workspace.

- Corte: **21/09/2026**; observação: **21/11/2026**.
- Alvo: cancelamento em **(21/09/2026, 20/11/2026]**, correspondente a 60 dias. A observação ocorre 61 dias após o corte e não amplia o horizonte.
- Fotografia: um cliente por corte, com permanência e preditores conhecidos até esse momento.
- Modelo principal futuro: pelo menos **60 dias de histórico anterior**. Clientes novos permanecem na exploração, com análise separada e possibilidade futura de Early Churn.
- Suporte: janelas **(corte − N dias, corte]**, para 30, 60 e 90 dias. Conforme a Aula 1, campo vazio representa zero chamados registrados. Datas inválidas, informação indisponível e histórico comprometido permanecem desconhecidos; datas futuras não entram nos preditores.
- Maturação histórica e cobertura futura são condições distintas. Recall será prioritário, considerando falsos positivos, custos e capacidade operacional.

**Premissa didática do exercício:** acompanhamento completo até 21/11/2026 para todos os clientes elegíveis, confirmado pelo usuário e não comprovado independentemente pelo CSV. Ela permite atribuir `churn_60d = 0` à ausência de cancelamento no horizonte e a eventos posteriores a 20/11, respeitando a elegibilidade. O notebook testa os limites usando a própria função do alvo. Em novas bases, a cobertura precisa ser revalidada.

Na visão exploratória há 5.000 clientes distintos, 4.990 elegíveis, 4.976 na coorte principal e 14 novos. Dez cancelamentos anteriores à aquisição são sinalizados como incompatíveis com o vínculo. Na coorte principal, são 1.763 cancelamentos e 3.213 permanências, com taxa de churn de 35,43%. Todos os 4.990 elegíveis têm desfecho conhecido no horizonte; os casos com cronologia incompatível continuam sem rótulo utilizável.

## Ajustes técnicos da EDA

- Os domínios da Aula 2 são critérios de validade: 12 mensalidades, 8 contagens de atraso e 80/90 grafias de contrato/pagamento violam suas regras na visão de 5.000 clientes. Ausência de informação é contabilizada separadamente; nenhuma violação foi corrigida ou removida definitivamente.
- Os 1.048 campos vazios de suporte consolidados passam a zero. Outros 33 históricos têm cronologia inconsistente e permanecem com contagem desconhecida, sem excluir os clientes.
- A mediana de suporte em 90 dias na fotografia elegível passa de 2 para 1; a correlação usa 4.798 casos completos, antes 3.804. Coortes e taxas de churn não mudam.
- Mensalidade, contrato, pagamento, atrasos e indicadores são tratados como informações disponíveis no corte por premissa do exercício. O CSV não tem histórico de atualização que comprove essa disponibilidade independentemente.
- Os gráficos têm zoom de exibição com quantidades fora do eixo anotadas. Estatísticas mantêm todos os valores, incluindo violações; continuam seis figuras.

## Estrutura

```text
.
├── analise_base_churn.ipynb
├── base_churn_5000_VF2_aluno.xlsx - Dados_Sinteticos_VF2.csv
├── requirements.txt
├── AGENTS.md
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

A leitura utiliza UTF-8, separador de campos `,` e vírgula decimal. A base contém **5.020 linhas e 10 colunas**. `data_ocorrencia_churn` foi identificada como origem provável da variável-alvo; sua aplicação usa as regras temporais acima e utiliza a cobertura futura confirmada para interpretar os campos de churn vazios.

As análises são realizadas em memória, sem sobrescrever a base original. Nesta etapa, não há imputação, remoção de registros ou outliers, codificação, normalização ou treinamento de modelos.

## Roteiro da EDA

1. Introdução e contextualização — implementada.
2. Carregamento e compreensão inicial — implementada.
3. Qualidade e consistência dos dados — implementada.
4. Análise univariada e investigação de valores extremos — implementada.
5. Compreensão da variável-alvo — implementada.
6. Análise bivariada — implementada.
7. Correlações entre atributos — implementada.
8. Investigações complementares justificadas pelos resultados — implementada.
9. Síntese e encaminhamentos para o pré-processamento — implementada.

## Referências secundárias de organização e análise

- [CIN0024 — Churn Analysis System, branch feature/churn-eda-preprocessing](https://github.com/llucasEmanuel/CIN0024-churn-analysis-system/tree/feature/churn-eda-preprocessing).
- [Projeto Lígia — Grupo 10](https://github.com/victoriapessoabm/PS_Ligia_Grupo_10).

Essas referências orientam apenas a estrutura do trabalho; em caso de dúvida metodológica, prevalecem as definições das aulas fornecidas no projeto. As interpretações do notebook são baseadas no arquivo local; resultados e regras dos outros projetos não são assumidos como válidos para esta base.
