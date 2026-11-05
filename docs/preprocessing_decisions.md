# Decisões de tratamento e pré-processamento

**Atualização da imputação:** este documento descreve a entrega determinística original. A aplicação posterior do Nível B, autorizada pelo usuário, é detalhada em [imputation_decisions.md](imputation_decisions.md). Agora a fábrica usa moda categórica por padrão, admite mediana ou KNN e o comando `scripts.run_imputation` gera partições preenchidas sem modificar as saídas originais. As afirmações abaixo sobre ausência de split ou de ajuste referem-se ao comando determinístico `scripts.run_preprocessing` e à entrega anterior a essa ampliação.

Este documento registra a passagem da EDA para o processamento reutilizável. As Aulas 1 e 2 do professor **Germano Vasconcelos** são a referência principal, conforme as regras fornecidas no projeto. Os PDFs não estão no workspace. O notebook foi lido integralmente, com atenção às Partes 3, 8 e 9, e permanece como registro histórico da EDA, sem alterações nesta etapa.

## 1. Referências examinadas antes da implementação

Os dois repositórios foram clonados em diretórios temporários externos ao projeto e somente consultados. Não foi copiado código de implementação.

| Referência consultada | Evidência observada | Aplicação neste projeto |
|---|---|---|
| [Churn, branch `feature/churn-eda-preprocessing`, commit `8eb2af1`](https://github.com/llucasEmanuel/CIN0024-churn-analysis-system/tree/8eb2af1ff074542713645d7b4b44e94baffdc227) | `scripts/preprocessing.py` tem **zero bytes**. O README não documenta uma etapa de preparação concluída. Não há testes ou saídas processadas versionadas nesse snapshot. | Não atribuímos ao grupo um pipeline modular que ainda não consta da branch. |
| Mesmo repositório, `notebooks/EDA.ipynb`, células 20, 26, 32, 34, 38, 40 e 44 | Conversão controlada de datas; visão provisória deduplicada/elegível; alvo em 60 dias; split estratificado 70/30; números inválidos convertidos em ausentes na visão de EDA de treino; meses completos, suporte em 90 dias e agrupamentos categóricos. O próprio fechamento do notebook aponta pré-processamento como próxima etapa. | Aproveitamos a separação entre fonte e visão analítica. Aqui adicionamos auditoria, coortes, conflitos e políticas explícitas. Não repetimos a borda inferior inclusiva dos 90 dias desse notebook: preservamos `(corte − N, corte]`, já documentado na nossa EDA. |
| [Lígia, commit `64b00f0`, `src/preprocessing.py`](https://github.com/victoriapessoabm/PS_Ligia_Grupo_10/blob/64b00f0032cb44ce5e5f20e13e1597e325a6f054/src/preprocessing.py) | Funções para leitura, limpeza determinística, grupos de atributos, `ColumnTransformer`, pipelines, split estratificado e coordenador que devolve pré-processador sem ajuste. Numéricos usam mediana/escala; categóricos usam imputação/one-hot. Treinamento fica em outro módulo. | Adotamos a separação de responsabilidades. O Nível A não depende de scikit-learn, do notebook ou de treinamento. O Nível B permite escolher escalonamento, em vez de impô-lo a todos os modelos. |

As observações acima se referem aos commits consultados, não a versões futuras dos outros grupos. A referência de churn foi acessível por Git, embora o navegador de consulta não tenha aberto a URL da branch.

## 2. Arquitetura e contratos

- `scripts/preprocessing.py`: esquema, leitura, deduplicação/conflitos, datas, categorias, domínios, elegibilidade, atributos temporais, alvo, validação e exportação. `prepare_data()` coordena o **Nível A**, retorna `PreparationResult` e não escreve arquivos nem faz `fit`.
- `scripts/ml_preprocessing.py`: lista positiva de preditores, extração supervisionada e fábrica `build_ml_preprocessor()`. O **Nível B** retorna um pipeline **não ajustado**.
- `scripts/run_preprocessing.py`: interface de execução; chama somente o Nível A, gera os artefatos e confere o hash da fonte.
- `tests/test_preprocessing.py`: regras, bordas temporais, auditoria, idempotência, resultados reais, reprodutibilidade, novos registros e ausência de vazamento no ajuste.

`prepare_data()` recebe um caminho CSV ou um DataFrame no **esquema bruto**. São aceitas as dez colunas originais e, opcionalmente, `data_corte` e `historico_suporte_incompleto`. Colunas adicionais são recusadas, para que um alvo previamente adicionado não interfira nem mesmo na deduplicação. Não recebe a própria tabela canônica: ela tem outro contrato, sem o histórico bruto de chamados. As normalizações determinísticas são idempotentes; repetir a execução a partir da mesma fonte produz as mesmas tabelas e arquivos. Os testes verificam ambos os aspectos, sem confundir reprocessamento de fonte com ingestão de saída.

`load_data()` usa UTF-8 e `keep_default_na=False`. Isso preserva a diferença entre `""` (ausência de evento, para suporte/cancelamento) e marcadores de informação desconhecida. Ao fornecer um DataFrame, use vazio literal para ausência de evento: `None`, `pd.NA` e marcadores como `?` são tratados conservadoramente como informação indisponível. Não passe uma leitura padrão que tenha apagado essa distinção.

O processamento representa **um corte por execução**. Se houver coluna `data_corte` na entrada, todas as datas devem corresponder ao parâmetro da execução. Múltiplos cortes não são ignorados ou misturados. Em uma ampliação futura, processar cada fotografia e planejar a avaliação por tempo/cliente.

## 3. Regras temporais

| Regra | Critério implementado |
|---|---|
| Corte | 21/09/2026, sem horário. A inclusão do dia corresponde à granularidade diária do exercício. |
| Horizonte | `(21/09/2026, 20/11/2026]`, exatamente 60 dias. |
| Observação | 21/11/2026, 61 dias após o corte. Não amplia o horizonte do alvo. |
| Cobertura futura | `assume_full_followup=True` é **premissa didática**, não comprovação independente do CSV. É possível desativá-la; desfechos desconhecidos ficam ausentes e `supervised_data()` recusa seu uso como rótulos. |
| Elegibilidade | Aquisição válida até o corte; cliente não cancelado antes/no corte; sem cronologia de cancelamento incompatível nem data essencial desconhecida. |
| Churn | Evento observado em `(corte, fim]` recebe 1. Permanência na janela recebe 0 com cobertura suficiente. Evento após o horizonte também recebe 0 sob essa cobertura, inclusive se registrado após a observação. No corte, o cliente é inelegível. |
| Maturação | Dias completos entre aquisição e corte: `>=60` para principal, `<60` para novos. Histórico anterior e acompanhamento futuro são condições distintas. |
| Permanência mensal | Meses completos de calendário, descontando um mês se o dia de aquisição ainda não foi alcançado. Não divide dias por 30. |
| Suporte 30 dias | `(22/08/2026, 21/09/2026]`: primeiro dia incluído 23/08. |
| Suporte 60 dias | `(23/07/2026, 21/09/2026]`: primeiro dia incluído 24/07. |
| Suporte 90 dias | `(23/06/2026, 21/09/2026]`: primeiro dia incluído 24/06. |
| Disponibilidade dos atributos de perfil | Mensalidade, contrato, pagamento, atrasos e indicadores são considerados disponíveis no corte por premissa do exercício. O CSV não traz histórico de atualização para comprovar isso independentemente. |

Nenhuma contagem preditora utiliza data posterior ao corte ou a data de cancelamento. Aquisição é usada para permanência/validade; cancelamento somente para elegibilidade, rótulo e auditoria. Datas brutas, identificador, coorte, status de qualidade e proveniência são metadados e ficam fora da matriz de ML.

## 4. Tratamentos escolhidos e evidência real

Os denominadores são explícitos: fonte com **5.020 linhas**, visão sem cópias com **5.000 clientes**, canônica com **4.990 elegíveis**, principal com **4.976** e novos com **14**. Contagens de ocorrências podem se sobrepor; não devem ser somadas como clientes distintos.

| Problema identificado na EDA | Tratamento escolhido | Justificativa | Quantidade real afetada | Consequência para modelagem |
|---|---|---|---|---|
| Duplicatas completas | Manter a primeira cópia na visão processada; rastrear todas as origens. | Não duplicar peso nem permitir cópias entre treino/teste. | 20 cópias excedentes, 40 linhas envolvidas. | 5.000 clientes distintos após essa etapa; fonte intacta. |
| Identificadores com valores conflitantes | Segregar todas as versões do identificador; não selecionar a mais recente nem combinar campos. | Não há regra confiável para resolver conflitos. | 0 casos reais; comportamento testado com casos sintéticos. | Evita criar um cliente fictício por consolidação arbitrária. |
| Grafias de contrato e pagamento | Mapeamentos explícitos após remover espaços externos, caixa e acentuação para comparação. | Valores equivalentes já demonstrados na EDA. | 80 ocorrências de contrato e 90 de pagamento, em 168 linhas distintas da visão sem cópias. | Domínios canônicos sem inventar categorias. |
| Categoria desconhecida | Converter para ausência e auditar o original. | Não escolher uma modalidade válida arbitrariamente. | 0 desconhecidas reais. Permanecem 25 contratos e 20 pagamentos ausentes nos elegíveis. | Imputação/encoding definidos posteriormente no Nível B. |
| Mensalidade fora de R$ 10–500 | Converter para ausência, sem excluir cliente. | Violação explícita da Aula 2; não é winsorização por IQR. | 12 valores: 8 de −85 e 4 de 9.999; somados aos 120 vazios, 132 ausentes na canônica. | Mediana, se escolhida, será aprendida apenas no treino. |
| Atrasos fora de inteiros 0–6 | Converter para ausência; rejeitar também frações. | Contagem e domínio da Aula 2. | 8 valores 99; com 35 vazios, 43 ausentes na canônica. | Não altera o alvo nem elimina clientes. |
| Indicadores fora de {0,1} | Converter para ausência, com auditoria. | Domínio binário. | 0 casos reais. | Eventual moda é aprendida no treino. |
| Datas em formatos diferentes | Aceitar apenas DD/MM/AAAA, AAAA-MM-DD e objetos de data já válidos, sem aproximar. | Preservar os dois formatos presentes. | 15 aquisições ISO na fonte; **0 datas impossíveis** encontradas. | Datas essenciais inválidas ou indisponíveis seriam segregadas, não convertidas em ausência de evento. |
| Churn antes da aquisição | Segregar o registro e preservar a fonte/auditoria. | Não sustenta vínculo nem rótulo temporal confiável. | 10 clientes. | Fora da canônica e das coortes; não recebem churn zero. |
| Cliente já cancelado no corte ou ainda não adquirido | Registrar inelegibilidade e manter o original no ledger. | Fora da população sob risco naquele corte. | Além dos 10 casos cronologicamente inválidos, 0 casos reais adicionais. | Não confundir inelegibilidade com classe zero. |
| Vazio literal de chamados | Zero eventos registrados nas janelas. | Semântica didática da Aula 1, compatível com os vazios reais do CSV. | 1.053 vazios na fonte; 1.048 após cópias; 1.047 nos elegíveis (1.045 principal e 2 novos). | Zeros são valores válidos, não imputação estatística. |
| Chamados antes da aquisição | Contagens de todo o histórico desse cliente ficam ausentes; preservar datas/parcela válida para auditoria. | Não comprovar completude a partir de um histórico inconsistente. | 52 eventos em 33 clientes elegíveis: 23 principal e 10 novos. | Os clientes são mantidos; três contagens de suporte ficam desconhecidas. |
| Data de chamado inválida, marcador desconhecido ou perda de histórico comprovada | Ausência nas contagens e código específico; não zero. | Ausência de informação não é ausência de evento. | 0 datas impossíveis ou perdas comprovadas reais; casos testados. | API aceita sinalizador opcional `historico_suporte_incompleto`. |
| Chamados após o corte | Excluir somente do cálculo preditor e preservar o evento na auditoria. | Impedir vazamento temporal. | 0 eventos reais; bordas testadas. | Alterar eventos futuros não muda preditores. |
| Permanência fora de 0–120 meses | Segregar para auditoria, sem truncar nem inventar aquisição. | A data define elegibilidade, exposição e coorte. | 0 casos reais; regra testada. | Não corrigir cronologia por aproximação. |
| Suporte fora de 0–50 eventos | Converter a contagem inválida para ausência, sem apagar eventos. | Domínio didático; também aplicável às janelas menores, que são subconjuntos. | 0 violações reais; 51 eventos sintéticos testados. | Nenhum corte por IQR ou winsorização. |
| Clientes novos | Manter visão própria, sem misturá-los à coorte principal. | Política de maturação das aulas. | 14 clientes, com 31 dias de permanência; 9 cancelamentos no horizonte. | Early Churn exige desenho próprio; 14 casos não sustentam conclusão robusta sobre treinamento. |

Nenhuma linha é removida por simplesmente conter ausência ou um extremo estatístico. A diferença de contagens é contabilizada: **5.020 = 20 cópias + 10 segregados + 4.976 principal + 14 novos**. Os alvos dos elegíveis somam **1.772 positivos e 3.218 negativos**; na principal, **1.763 e 3.213**, respectivamente. A taxa principal permanece **35,43%**.

## 5. Saídas, esquema e rastreabilidade

Os sete arquivos na raiz de `data/processed/` são as saídas determinísticas abaixo. A subpasta `imputation/` contém as visões aprendidas após a divisão treino/teste, documentadas separadamente.

| Arquivo | Conteúdo |
|---|---|
| `churn_canonical.csv` | 4.990 elegíveis; preditores limpos, ausências preservadas, metadados, coorte e alvo. |
| `churn_main.csv` | 4.976 clientes maturados. |
| `churn_new_customers.csv` | 14 clientes novos, preservados separadamente. |
| `audit_rows.csv` | Todas as 5.020 linhas originais, com destino e códigos. Inclui as cópias e os segregados. |
| `audit_issues.csv` | Problema/mudança por linha, campo, ação, valor original e motivo. Valores derivados inválidos constam também no detalhe. |
| `audit_support_events.csv` | Uma linha por data de chamado dos elegíveis sem cópias: texto, interpretação, origem e limites de inclusão. |
| `audit_summary.json` | Contagens por destino/código, ausências, parâmetros, proveniência, versões, hash da implementação e hashes dos CSVs. Sem timestamps de execução ou caminhos da máquina, para permitir comparação byte a byte. |

`source_rows` identifica as linhas originais de cada cliente, incluindo as cópias, com o primeiro registro no número 2. Os CSVs processados usam ponto decimal e datas ISO. Ausências são gravadas vazias e retomadas como ausências na leitura padrão desses arquivos canônicos. A semântica de vazio de evento se aplica ao **esquema bruto**, não à reingestão da saída como se fosse fonte.

O histórico bruto permanece em `audit_rows.csv`, não nos preditores. Em `audit_support_events.csv`, `included_Nd` expressa a condição de inclusão de **cada evento**. Se o histórico do cliente estiver comprometido, a contagem final fica ausente mesmo que existam parcelas válidas: não somar essas flags e chamar o resultado de histórico completo. Consulte também `status_suporte` e `audit_issues.csv`.

O esquema ordenado é definido por `CANONICAL_COLUMNS`; `schema_version` está no relatório. `FEATURE_COLUMNS` é a lista positiva de **dez** preditores. Metadados como `data_ocorrencia_churn`, `id_cliente`, `source_rows` e o próprio `churn` nunca entram no `ColumnTransformer`.

Não use apenas `data.drop(columns="churn")` para formar X: a tabela canônica contém datas e metadados de auditoria. Use `select_features()` ou `supervised_data()` para impedir que essas informações entrem acidentalmente no modelo.

## 6. Nível B e prevenção de vazamento

O comando determinístico não faz split nem ajuste. A aplicação autorizada em `scripts.run_imputation` usa uma divisão estratificada 80/20 e validação interna no treino, descritas em `imputation_decisions.md`. Com múltiplos cortes, será necessário considerar tempo e compartilhamento de clientes. O desenho não deve ser escolhido pelo desempenho no teste.

`build_ml_preprocessor()` devolve pipeline sem ajuste:

- Numéricos: mediana por padrão ou `numeric_imputation="knn"`, ajustados exclusivamente no treino. Se um atributo numérico/binário estiver totalmente ausente no treino, o ajuste falha com mensagem explícita, em vez de inventar estatística.
- Categóricos: por padrão, `categorical_imputation="most_frequent"` aprende moda somente no treino e recusa colunas inteiramente ausentes, pois não há moda definida. O marcador `[ausente]` permanece como alternativa explícita com `categorical_imputation="constant"`. A codificação usa `OneHotEncoder(handle_unknown="ignore")`.
- Binários: moda somente no treino; permanecem separados de escalonamento contínuo.
- Escalonamento: `none` por padrão; `standard` ou `minmax` são alternativas explícitas. Não há pipeline universal imposto a Árvores de Decisão e métodos de distância.
- `select_features()` usa a mesma lista positiva para uso supervisionado e não supervisionado. `supervised_data()` também valida o alvo; recusa desfechos desconhecidos.

Exemplo opcional de uso futuro, **após escolher os conjuntos**:

```python
from sklearn.model_selection import train_test_split
from scripts.preprocessing import prepare_data
from scripts.ml_preprocessing import supervised_data, build_ml_preprocessor

result = prepare_data("base_churn_5000_VF2_aluno.xlsx - Dados_Sinteticos_VF2.csv")
X, y = supervised_data(result.main)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
preprocessor = build_ml_preprocessor(scaling="none")
X_train_ready = preprocessor.fit_transform(X_train)
X_test_ready = preprocessor.transform(X_test)
```

O exemplo não é executado pelo script determinístico e não treina classificador. A execução aplicada do Nível B fica em outro comando, `scripts.run_imputation`. Para agrupamento, usar `select_features(result.canonical)` e escolher uma estratégia de ajuste/avaliação adequada, sem incluir o alvo; não reutilizar um ajuste global do notebook. Em validação cruzada, o pré-processador deverá integrar o pipeline de cada fold/modelo.

Para preparar futuros registros sem informação de cancelamento, `prepare_data(frame, include_target=False)` aceita a ausência dessa coluna, mantém `churn` desconhecido e permite extrair os mesmos dez preditores. A população ativa no corte deve ser fornecida/validada pela origem; a função não comprova sozinha a situação contratual de um cliente sem esse histórico.

As decisões de imputação, escala, seleção de janelas redundantes e tratamento de perdas históricas permanecem revisáveis para cada algoritmo. Recall, falsos positivos e custos operacionais orientarão a avaliação futura. Nenhum modelo de classificação ou agrupamento foi implementado.

Referências técnicas da implementação: [prevenção de vazamento no scikit-learn 1.7](https://scikit-learn.org/1.7/common_pitfalls.html#data-leakage), [ColumnTransformer](https://scikit-learn.org/1.7/modules/generated/sklearn.compose.ColumnTransformer.html), [SimpleImputer](https://scikit-learn.org/1.7/modules/generated/sklearn.impute.SimpleImputer.html) e [OneHotEncoder](https://scikit-learn.org/1.7/modules/generated/sklearn.preprocessing.OneHotEncoder.html). As regras do problema continuam sendo as das aulas.

## 7. Execução e validação

Na raiz do repositório, usando o ambiente existente:

```bash
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m scripts.run_preprocessing
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/test_preprocessing.py -q
```

O último comando desativa a descoberta de plugins externos, pois este ambiente também expõe plugins ROS que não pertencem ao projeto. Não exige instalar dependências ROS nem criar outro ambiente. O pipeline também pode ser executado como `.venv/bin/python scripts/run_preprocessing.py`, sem notebook aberto.

As execuções substituem apenas os sete arquivos gerados com nomes definidos; não apagam outros arquivos da pasta. Há proteção para não sobrescrever a entrada. `--input` e `--output-dir` permitem outros caminhos; `--cutoff`, `--observation` e `--no-full-followup` explicitam mudanças nas premissas. Cada configuração fica registrada no relatório.

Os testes incluem uma execução em diretório temporário sem notebook, duas exportações comparadas byte a byte, hashes de entrada/saídas e as contagens reais da EDA. Testes de transformações aprendidas fazem `fit` em subconjuntos de treino controlados e verificam que `transform` não muda medianas, escalas ou vocabulário, inclusive com categorias/ausências novas. Isso testa componentes; não treina modelos e não ajusta um pré-processador global da base real.
