# Orientações do projeto

## Referência e escopo

- Este é o miniprojeto de Sistemas Inteligentes do CIn/UFPE. As Aulas 1 e 2 do professor Germano Vasconcelos são a referência principal para EDA, consolidação, pré-processamento e modelagem. As regras abaixo foram explicitadas pelo usuário; os PDFs não estão no workspace.
- Priorize definições conceituais e temporais explícitas em relação a rótulos contraditórios de tabelas didáticas e a soluções genéricas de ML. Sinalize divergências, sem copiá-las para os dados reais.
- Diferencie regras das aulas, hipóteses analíticas e achados efetivamente observados. Não afirme causalidade a partir de associações.
- Preserve o CSV original integralmente. Derivações exploratórias devem estar em visões auxiliares com proveniência, sem exportar uma base corrigida como original.
- Aproveite o notebook existente e sua estrutura numerada. Use textos em PT-BR, código legível, caminhos relativos e gráficos justificados.
- Execute e valide com `.venv/bin/python`. Não crie outro ambiente. Não faça commits automaticamente.

## Problema e tempo

- Prever cancelamento de clientes de telecomunicações nos 60 dias seguintes ao corte de 21/09/2026. A observação informada é 21/11/2026.
- O alvo usa `(corte, corte + 60 dias]`, terminando em 20/11/2026. Cancelamento no corte já torna o cliente inelegível; no último dia recebe 1; depois do horizonte recebe 0 se há cobertura confiável, inclusive para datas posteriores à observação. Teste esses casos com a função aplicada à fotografia. A observação está 61 dias após o corte; não a use para estender o horizonte.
- A granularidade é uma fotografia por cliente e corte. Considere chave composta `id_cliente` + `data_corte` se houver múltiplos cortes.
- Mensalidade, contrato, pagamento, atrasos e indicadores são assumidos como disponíveis no corte conforme o exercício; o CSV não possui histórico de atualização para comprovação independente. Documente a premissa e valide a origem em uma aplicação real.
- Preditores devem ser conhecidos até o corte. Janelas de suporte usam `(corte − N dias, corte]`, para N = 30, 60 e 90; documente inclusão/exclusão das bordas.
- Permanência mensal exploratória usa meses completos de calendário. Maturação é medida em dias: o modelo principal exige histórico >=60 dias no corte.
- Preserve e analise os clientes com menos de 60 dias separadamente. Não os exclua definitivamente da EDA; avalie futuramente Early Churn com desenho próprio.
- Histórico anterior curto e falta de acompanhamento futuro são problemas distintos. Não atribua zero a desfechos não integralmente observados.
- O usuário confirmou acompanhamento completo até 21/11/2026 para todos os clientes elegíveis, inclusive os que têm `data_ocorrencia_churn` vazia. Essa é uma premissa didática do exercício, não comprovada independentemente pelo CSV: o alvo auxiliar `churn_60d` pode receber zero nesses casos, respeitando o horizonte. A origem da regra deve ficar explícita; revalide a cobertura em novas bases/cortes. Casos cronologicamente incompatíveis continuam sem rótulo utilizável.
- Conforme a semântica da Aula 1 fornecida pelo usuário, campo vazio de `data_chamado` representa zero chamados registrados nesta base didática. Não confunda com marcadores de informação indisponível, datas inválidas, cronologia inconsistente ou histórico comprovadamente incompleto: nesses casos, mantenha NA e sinalizadores, sem inventar zero. Preserve flags de exposição às janelas.

## Qualidade e etapas futuras

- Use as quatro dimensões da Aula 2: completude, validade, consistência e unicidade.
- Investigue categorias canônicas, ausências, duplicatas, cronologia e faixas de domínio. Os domínios explícitos da Aula 2 são critérios de validade da base didática: permanência 0–120 meses, mensalidade R$ 10–500, suporte inteiro 0–50 em 90 dias e atrasos inteiros 0–6; indicadores em {0,1} e categorias canônicas. Valores fora desses domínios são violações, sem correção ou remoção automática.
- Extremos estatísticos não são automaticamente erros. Não aplique tratamentos definitivos sem solicitação.
- Quando o pré-processamento for autorizado, implemente-o modularmente, independente da execução manual do notebook.
- Separe treino e teste antes de ajustar transformações que aprendem parâmetros; ajuste exclusivamente no treino e use `transform` no teste, com pipelines. Considere avaliação temporal e sobreposição de clientes quando aplicável.
- Recall será a métrica principal, acompanhado de falsos positivos, custos de intervenção e capacidade operacional. Não treine modelos antes de autorização.

## Segunda etapa: preparação autorizada e implementada

- O usuário autorizou o tratamento e pré-processamento modular. Preserve o notebook finalizado como histórico da EDA; não o reimplemente para executar a preparação.
- `scripts/preprocessing.py` coordena o Nível A determinístico; `scripts/ml_preprocessing.py` fornece componentes não ajustados do Nível B. Mantenha esses contratos separados e não faça fit na preparação global.
- Duplicatas exatas são removidas só na visão processada. Conflitos de identificador e datas essenciais inutilizáveis são segregados com auditoria. Violações numéricas e categorias desconhecidas viram ausências documentadas; não invente correções.
- As decisões e a comparação com os repositórios de referência estão em `docs/preprocessing_decisions.md`. Na branch de churn consultada (`8eb2af1`), `scripts/preprocessing.py` estava vazio; não atribua ao grupo um pipeline modular concluído.
- As saídas ficam em `data/processed/`. Toda linha de entrada deve ter destino auditado e os atributos devem manter proveniência. Não confunda a base canônica com uma matriz de ML já ajustada.
- Testes: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/test_preprocessing.py -q`. A variável evita plugins ROS externos; não instale dependências ROS para este projeto.
