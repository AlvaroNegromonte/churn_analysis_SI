# Análise e preparação de dados de churn

Miniprojeto da disciplina de **Sistemas Inteligentes — CIn/UFPE**, voltado à análise exploratória de uma base de churn. Os resultados orientarão o pré-processamento e projetos futuros com algoritmos supervisionados e não supervisionados.

A EDA está concluída nas Partes 1–9 e seu notebook foi preservado como registro da primeira etapa. A preparação tem um nível determinístico e componentes aprendidos somente no treino. A etapa de imputação compara mediana e KNN dentro do treino e preenche categóricos por moda em visões separadas. Nenhum classificador de churn ou algoritmo de agrupamento foi treinado.

## Executar o pré-processamento

Utilize a `.venv` existente, na raiz do projeto:

```bash
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m scripts.run_preprocessing
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/test_preprocessing.py -q
```

A descoberta automática de plugins é desativada nos testes para evitar interferência dos plugins ROS presentes no ambiente; isso não exige outro ambiente nem dependências ROS. O processamento funciona sem executar ou abrir o notebook. Também pode ser chamado por `scripts/run_preprocessing.py`, com opções `--input` e `--output-dir`.

Os sete arquivos gerados em `data/processed/` incluem as bases canônica (**4.990 clientes**), principal (**4.976**) e novos (**14**), mais auditoria de todas as **5.020 linhas originais**, alterações, eventos e hashes. Reexecuções substituem apenas esses arquivos gerados. O CSV original permanece intacto.

Na base canônica, as **12 mensalidades e 8 contagens de atraso inválidas viram ausências auditadas**, sem eliminar clientes. As 170 ocorrências categóricas equivalentes são padronizadas. Conflitos de identificador são segregados, e os 10 cancelamentos anteriores à aquisição permanecem na auditoria. São preservadas 132 mensalidades, 43 contagens de atraso e 33 históricos de suporte desconhecidos, sem imputação estatística nesta etapa.

Leia [as decisões, a comparação com os projetos de referência, os critérios temporais e o esquema das saídas](docs/preprocessing_decisions.md).

## Dois níveis de preparação

- **Nível A — determinístico:** `scripts/preprocessing.py`, coordenado por `prepare_data()`. Produz uma base com esquema estável e proveniência, categorias canônicas e ausências que dependem de escolhas futuras. Não usa medianas/modas nem escalas aprendidas.
- **Nível B — aprendido:** `scripts/ml_preprocessing.py`, com `build_ml_preprocessor()`. Retorna pipeline **não ajustado**, com mediana ou KNN nos numéricos, moda categórica por padrão, one-hot e escala opcional. O ajuste deve acontecer exclusivamente no treino, inclusive dentro dos folds de validação de modelos futuros.

`select_features()` retorna somente os dez preditores autorizados, sem identificadores, datas de cancelamento, coorte ou alvo, inclusive no uso não supervisionado. `supervised_data()` também extrai e valida `churn`. A base limpa ainda **não é uma matriz numérica ajustada para um algoritmo específico**; há exemplos e limitações no documento de decisões.

## Imputação aplicada, sem modificar a EDA ou a base canônica

```bash
.venv/bin/python -m scripts.run_imputation
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/test_preprocessing.py tests/test_imputation.py -q
```

O comando prepara os dados em memória e separa a coorte principal em treino/teste 80/20 estratificado, com semente 42. Compara mediana e KNN (k=5) em três folds internos do treino, ocultando temporariamente valores conhecidos. O teste não participa da escolha. Ambos os métodos usam moda nos categóricos e binários. Os parâmetros finais são ajustados apenas no treino; teste e novos recebem `transform`.

As saídas adicionais ficam em `data/processed/imputation/`: visões preenchidas de treino, teste, novos e de todos os elegíveis, auditoria por célula, métricas de validação, máscaras e relatório JSON. Cada preditor tem uma flag `imputed_*`. Os NA da base canônica continuam representando informação desconhecida; somente a visão derivada contém estimativas. Os dez vínculos segregados não são recuperados por invenção de datas.

O relatório registra o método escolhido e a comparação antes/depois. As contagens estimadas respeitam inteiros, domínios e a ordem 30 ≤ 60 ≤ 90 dias, sem alterar contagens observadas. A escolha por mascaramento artificial é uma evidência exploratória, não prova de recuperação dos valores reais. Leia [o protocolo de imputação](docs/imputation_decisions.md).

Nesta cópia Windows, sem `.venv`, o usuário autorizou uma exceção para validação com o Python já instalado e dependências temporárias em `.validation_deps/`. Essa pasta é ignorada pelo Git. Não foi criado outro ambiente virtual. O procedimento recomendado para o ambiente original continua usando `.venv/bin/python`.

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

## Registro dos ajustes técnicos da EDA

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
├── scripts/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── ml_preprocessing.py
│   ├── imputation.py
│   ├── run_preprocessing.py
│   └── run_imputation.py
├── tests/
│   ├── test_preprocessing.py
│   └── test_imputation.py
├── docs/
│   ├── preprocessing_decisions.md
│   └── imputation_decisions.md
├── data/
│   └── processed/              # sete arquivos determinísticos e subpasta imputation/
├── requirements.txt
├── AGENTS.md
├── .gitignore
└── README.md
```

- `analise_base_churn.ipynb`: roteiro da EDA, análises implementadas, saídas e interpretações.
- `base_churn_5000_VF2_aluno.xlsx - Dados_Sinteticos_VF2.csv`: arquivo original, preservado. Seu formato real é CSV, apesar de conter `.xlsx` no nome.
- `requirements.txt`: dependências diretas, nas versões validadas, incluindo scikit-learn e pytest para a segunda etapa.
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

Na EDA, as análises foram realizadas em memória, sem sobrescrever a base original. A preparação posterior gera arquivos separados em `data/processed/`, com decisões auditadas. Imputação estatística, codificação e eventual escalonamento são componentes de um pipeline que deverá ser ajustado somente após separar treino/teste; nenhum modelo foi treinado.

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


