# Imputação: protocolo, limites e proveniência

## Escopo autorizado

O usuário autorizou aplicar a imputação e comparar mediana e KNN nos preditores numéricos, utilizando moda para categorias. Esta escolha de política foi autorizada na conversa; não afirmamos que a moda é universalmente superior nem que consultamos os PDFs das aulas. Eles continuam ausentes do workspace.

O notebook permanece como diagnóstico histórico anterior ao preenchimento. `scripts/preprocessing.py` não é alterado e não aprende parâmetros. Sua base canônica continua com NA quando o valor é desconhecido ou inválido. `scripts/run_imputation.py` gera uma visão derivada com estimativas identificadas, sem mudar o CSV original ou os sete arquivos determinísticos existentes. Nenhum slide é alterado nesta etapa; sua atualização depende do aval solicitado pelo usuário.

## Política por tipo de atributo

| Tipo | Estratégia |
|---|---|
| Mensalidade, atrasos e contagens de suporte | Comparação entre mediana e KNN, com seleção exclusiva no treino. |
| Contrato e pagamento | Moda do treino. Empates seguem a ordenação determinística do `SimpleImputer`. |
| Indicadores binários | Moda do treino. Na base real não há ausências nesses dois campos. |
| Identificador, datas essenciais, alvo | Não são imputados. Não entram no cálculo das vizinhanças. |

Um campo vazio literal de suporte já significa zero registrado, conforme o exercício. Não é imputado novamente. Um histórico comprometido continua NA na fonte canônica; na visão preenchida recebe uma estimativa e flag. Não afirmamos que os chamados realmente aconteceram nessas quantidades. A auditoria preserva também o status do histórico.

## Desenho da comparação

1. Construir o Nível A em memória, sem exportar por cima das suas saídas.
2. Separar os 4.976 clientes da coorte principal em 80% treino e 20% teste, estratificados por `churn`, com semente 42. O alvo participa somente dessa estratificação, nunca da imputação ou do score de seleção.
3. Usar três folds internos do treino para comparar os métodos. Cada fold aprende parâmetros em sua parcela de ajuste e transforma uma parcela distinta de validação.
4. Ocultar 10% dos valores conhecidos por atributo avaliado na parcela de validação, com semente determinada pelo fold. As três contagens de suporte são ocultadas juntas, refletindo a perda conjunta observada nos históricos comprometidos reais. Valores naturalmente ausentes não são usados como verdade de referência.
5. Avaliar MAE e RMSE por atributo numérico. Dividir o MAE pelo desvio padrão da parcela de ajuste para comparação entre escalas. O score do fold dá peso igual a cada atributo ou bloco: mensalidade, atrasos e a média das três janelas de suporte. Permanência não entra no score quando não há ausências nesse atributo no treino.
6. Escolher KNN somente se a média dos scores internos for mais de 1% inferior à mediana, margem definida antes da execução. Empates e melhorias menores favorecem a mediana, mais simples. Não houve busca pelo melhor k; usa-se k=5 por padrão.
7. Medir também a acurácia da moda nos valores categóricos ocultados. Esse resultado é descritivo; não é um classificador de churn.
8. Ajustar o método escolhido em todo o treino externo. Transformar teste e os 14 clientes novos sem refazer o ajuste. O teste não é usado para escolher o método ou a margem.

O mascaramento artificial facilita comparar estimativas com valores conhecidos, mas não demonstra o desempenho nas ausências reais. O mecanismo de erro/ausência pode ser diferente, inclusive nos 33 históricos de suporte. Resultados não são métricas preditivas de churn e não demonstram benefício de retenção.

## KNN e domínios

`CustomerImputer` padroniza numéricos usando apenas a parcela de ajuste. Acrescenta os indicadores binários e categorias preenchidas por moda e codificadas por one-hot para definir semelhança. Categorias desconhecidas na transformação não alteram o vocabulário aprendido. Alvo, ID, datas, partição e flags não entram nessa distância.

As três janelas de suporte recebem peso de 1/√3 na distância, para que o bloco não tenha automaticamente o peso de três atributos independentes. Cada bloco one-hot recebe fator 1/√2, de modo que trocar de categoria contribua uma unidade de distância quadrática. Os binários permanecem em 0/1. Essas são escolhas analíticas explícitas, não regras atribuídas às aulas.

Somente estimativas para células ausentes são limitadas ao domínio. Contagens usam inteiro mais próximo, com empate x,5 para cima. As contagens de suporte estimadas respeitam 30 ≤ 60 ≤ 90 dias e as âncoras observadas. Valores observados inconsistentes são recusados, não corrigidos silenciosamente. Os valores presentes são mantidos exatamente.

## Arquivos em `data/processed/imputation/`

| Arquivo | Conteúdo |
|---|---|
| `churn_imputed_train.csv` | Clientes do treino, com preditores preenchidos, metadados, alvo e flags. |
| `churn_imputed_test.csv` | Teste transformado pelo imputador do treino. |
| `churn_imputed_new_customers.csv` | Novos transformados, sem participar do ajuste. |
| `churn_imputed.csv` | União das três visões, com a partição identificada; não representa ajuste global. |
| `imputation_audit.csv` | Uma ocorrência por célula preenchida, com método, estimativa, partição, origem e status de suporte. |
| `imputation_validation.csv` | Métricas por fold, método e atributo. |
| `imputation_validation_masks.csv` | Valores conhecidos ocultados, com fold, cliente e atributo para rastreabilidade. |
| `imputation_summary.json` | Configuração, scores, estratégia escolhida, contagens antes/depois, modas, versões e hashes das saídas. |

Todas as linhas elegíveis são preservadas; preenchemos preditores, sem alterar os rótulos. Metadados podem continuar vazios, por exemplo a data de churn de quem não cancelou. A ausência de NA é exigida somente nos dez preditores preenchidos.

## Uso nos próximos miniprojetos

`build_ml_preprocessor(numeric_imputation="median" ou "knn", categorical_imputation="most_frequent")` continua retornando um pipeline não ajustado. A alternativa categórica `constant` permanece disponível explicitamente. Ao usar validação cruzada para modelos futuros, o pipeline deve receber os dados canônicos de cada fold, reaprendendo imputação, codificação e escalas no respectivo treino.

Não executar validação cruzada diretamente na visão que já foi imputada com todo o treino externo: isso compartilharia parâmetros entre os folds. As saídas preenchidas documentam a aplicação e servem para uma avaliação holdout com a partição preservada. Não são uma nova base original para dividir novamente ao acaso.

Para o MP3, o desenho de ajuste/avaliação do agrupamento deve ser definido naquele experimento. `select_features()` exclui o alvo e metadados. As flags são preservadas para proveniência; a lista de dez preditores continua explícita e não as incorpora automaticamente às distâncias de ML.

## Execução e ambiente

```bash
.venv/bin/python -m scripts.run_imputation
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/test_preprocessing.py tests/test_imputation.py -q
```

A cópia atual não tem `.venv`. O usuário autorizou usar o Python existente com dependências temporárias em `.validation_deps/`, sem criar outro ambiente virtual. Essa exceção de validação deve constar do registro de execução; não implica trocar o ambiente recomendado do projeto.

Referências técnicas: [SimpleImputer](https://scikit-learn.org/1.7/modules/generated/sklearn.impute.SimpleImputer.html), [KNNImputer](https://scikit-learn.org/1.7/modules/generated/sklearn.impute.KNNImputer.html), [StandardScaler](https://scikit-learn.org/1.7/modules/generated/sklearn.preprocessing.StandardScaler.html) e [prevenção de vazamento](https://scikit-learn.org/1.7/common_pitfalls.html#data-leakage), na versão 1.7 do scikit-learn.

## Resultado da execução nesta cópia

Validação com Python 3.13.6, pandas 2.3.3, NumPy 2.2.6, scikit-learn 1.7.2 e pytest 9.2.0. A suíte original e a nova suíte totalizaram **95 testes aprovados**. Não foi criado ambiente virtual nem alterado o Python global; as dependências foram instaladas na pasta temporária autorizada.

O experimento usou 3.980 clientes de treino, 996 de teste e 14 novos somente para transformação. A seleção foi **KNN**, com score médio interno normalizado **0,71745**, contra **0,80279** da mediana, redução de aproximadamente **10,63%**. Essa porcentagem é a redução do score de erro de imputação no mascaramento artificial, não uma acurácia nem ganho de previsão de churn.

| Atributo numérico | MAE médio dos folds: mediana | MAE médio dos folds: KNN |
|---|---:|---:|
| Mensalidade (R$) | 16,35 | 9,18 |
| Atrasos | 0,70 | 0,80 |
| Suporte 30 dias | 0,59 | 0,68 |
| Suporte 60 dias | 0,87 | 0,91 |
| Suporte 90 dias | 1,22 | 1,18 |

KNN venceu pelo critério agregado predefinido, principalmente pela mensalidade, mas **não venceu em todos os atributos**. A escolha conjunta é uma política deste experimento, não prova de superioridade universal. Estratégias por atributo podem ser examinadas em experimentos futuros, mantendo a escolha dentro do treino.

As modas aprendidas no treino foram **Mensal** para contrato e **Boleto** para pagamento. No mascaramento interno, a moda acertou aproximadamente 55,68% dos contratos e 38,38% dos pagamentos ocultados. Esses resultados ilustram a limitação de substituir todos os desconhecidos pela categoria mais frequente.

| Preditor | NA canônicos | NA na visão preenchida |
|---|---:|---:|
| Mensalidade | 132 | 0 |
| Atrasos | 43 | 0 |
| Suporte 30 dias | 33 | 0 |
| Suporte 60 dias | 33 | 0 |
| Suporte 90 dias | 33 | 0 |
| Contrato | 25 | 0 |
| Pagamento | 20 | 0 |

Foram preenchidas **319 células**, preservando os **4.990 clientes**, os rótulos e os valores observados. As flags e a auditoria identificam cada estimativa. A canônica mantém as mesmas ausências; a EDA e os slides não foram modificados.

Comandos usados nesta exceção Windows, com o Python existente e `PYTHONPATH` apontando para as dependências temporárias:

```powershell
$env:PYTHONPATH = (Resolve-Path '.validation_deps').Path
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
python -m pytest tests/test_preprocessing.py tests/test_imputation.py -q --basetemp=.pytest_cache/imputation_tmp
python -m scripts.run_imputation
```
