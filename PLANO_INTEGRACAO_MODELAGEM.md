# Plano de integração da modelagem de séries temporais

## Objetivo

Integrar ao notebook central `projeto/cd_projeto_aplicado_IV_doc.ipynb` a etapa inédita de modelagem da parte 13 em diante de `projeto/ProjetoAplicadoIV__A3___Rev1_final.ipynb`, preservando a organização modular já adotada pelo projeto e evitando duplicação de carregamento, análises exploratórias e infraestrutura específica do Google Colab.

O notebook central continua sendo o ponto de execução e apresentação. Funções reutilizáveis ficam nos módulos de `projeto/ast_caged/`; resultados derivados dos dados são calculados durante a execução e armazenados junto às séries correspondentes.

## Decisões de integração

### Conteúdo e organização do notebook

- Aproveitar e revisar o conteúdo metodológico das partes 13 e 14 da Rev1, posicionando-o na seção **Modelos** do notebook central.
- Não transcrever resultados numéricos, vencedores ou conclusões da Rev1 como fatos fixos. Eles devem ser produzidos ou atualizados com a execução integrada.
- Remover a instalação de pacotes com `subprocess.check_call()` do notebook. Imports ficam nos módulos que os utilizam; o notebook importa apenas as funções necessárias.
- Declarar no `projeto/requirements.txt` as dependências efetivamente necessárias, incluindo Prophet se ele continuar como modelo concorrente.
- Não importar a conexão e os caminhos do Google Drive, a procura por arquivos `.gsheet`, o relatório automático da parte 21 nem a GUI baseada em widgets da parte 22.
- Manter a análise exploratória e o diagnóstico já existentes no notebook central. A nova etapa começa após o diagnóstico, sem repetir inspeção e carregamento das bases.

### Split temporal e registro dos períodos

- Criar `projeto/ast_caged/modules/inference/datasplit.py` para definir e aplicar os recortes temporais.
- A função recebe o nome da série e seu DataFrame e retorna os recortes de treino, validação e teste necessários à modelagem, além de um DataFrame de uma linha com:
  `uf`, `treino_inicio`, `treino_fim`, `teste_inicio`, `teste_fim`, `meses_treino`, `validacao_inicio` e `validacao_fim`.
- O valor de `uf` é o nome fornecido na entrada e deve corresponder à chave existente em `time_series_uf`.
- Manter validação e teste com horizontes de 12 meses, para cobrir ciclos anuais completos e permitir comparação consistente entre séries. O treino usa o histórico restante e cresce à medida que novos meses são adicionados.
- Parametrizar a regra e validar o tamanho mínimo do histórico. Se a série não comportar treino, validação e teste mínimos, interromper com erro explícito em vez de reduzir silenciosamente os horizontes.
- O cálculo dos metadados e a criação dos recortes devem vir da mesma função/regra, evitando discrepância entre a tabela exibida e os dados efetivamente usados no ajuste.
- Armazenar os resultados do split dentro do registro de cada série, por exemplo em `time_series_uf[uf]["datasplit"]`. Montar `TABELA_PERIODOS` iterando sobre as séries e concatenando os DataFrames de uma linha. Não criar uma chave de primeiro nível como `"periodos"` em `time_series_uf`, pois isso quebraria os loops que assumem que cada chave representa uma UF.

### Modelos e previsões

- Implementar funções de previsão em `projeto/ast_caged/modules/inference/prediction.py` e importá-las no notebook nas etapas em que forem usadas.
- Seguir o padrão do projeto: iterar pelas chaves de `time_series_uf` e buscar os DataFrames e recortes no registro de cada série, sem criar um fluxo paralelo de carga de dados.
- Calcular previsões de admissões e desligamentos separadamente. O saldo não é ajustado como uma terceira série: é derivado pela diferença entre os fluxos previstos.
- Armazenar previsões, parâmetros selecionados, métricas e informações suficientes para identificar o modelo usado dentro do registro de cada UF. Os resultados devem poder alimentar as tabelas, projeções e gráficos sem depender de valores digitados em Markdown.

#### Baseline sazonal ingênua

- Implementar a baseline sazonal em `prediction.py`, usando os últimos 12 meses disponíveis no segmento de treino e repetindo esse ciclo para o horizonte previsto.
- Aplicá-la separadamente a admissões e desligamentos de cada série.
- Expor as previsões calculadas no contexto de `time_series_uf`; a etapa da baseline deve também apresentar uma saída verificável, e não apenas definir a função.

#### SARIMAX

- Implementar em `prediction.py` a busca de parâmetros SARIMAX por série e por fluxo, com uma grade candidata explícita e limitada.
- Selecionar parâmetros pela validação temporal, sem consultar o teste. Registrar os parâmetros escolhidos e as falhas/avisos relevantes no registro da série.
- Manter visível qual SARIMAX foi selecionado para cada série e preservar parâmetros diferentes quando admissões e desligamentos justificarem ajustes distintos.
- Tratar falhas de ajuste e não convergência explicitamente. Se nenhum candidato válido restar, interromper com erro informativo, sem produzir um resultado com aparência de sucesso.

#### Prophet

- Manter Prophet como terceiro concorrente, sem sazonalidade diária/semanal nem efeitos de feriados diários: os dados são mensais e não contêm a resolução necessária para estimar esses efeitos.
- Explicar no texto metodológico as escolhas de tendência e sazonalidade. A ordem Fourier 5 e os demais hiperparâmetros não devem ser apresentados como escolhas automáticas sem justificativa.
- Se forem selecionados por busca, usar uma grade limitada e validação temporal, independente para cada série/fluxo, e armazenar parâmetros, resultados de validação e previsões. Não ajustar hiperparâmetros no teste.
- Validar disponibilidade e compatibilidade da dependência Prophet no ambiente definido pelo projeto.

### Comparação, seleção e resultados

- Calcular as métricas a partir das previsões e dos períodos reais correspondentes. Exibir tabelas geradas pelos cálculos; não usar tabelas fixas em Markdown para resultados quantitativos.
- Comparar os três modelos para cada UF e fluxo de forma individual. A escolha para admissões pode diferir da escolha para desligamentos.
- Não incluir o saldo na tabela de previsões como se fosse uma terceira variável ajustada. Se o objetivo de negócio exigir avaliar o saldo, calculá-lo pela diferença entre os dois fluxos previstos e apresentar suas métricas como avaliação de um resultado derivado.
- Definir e documentar antes da comparação qual métrica orienta a seleção. Usar validação para escolher configurações e, se a família vencedora for selecionada por desempenho, não usar o mesmo teste como avaliação independente final.
- Manter o teste fora da busca e do ajuste dos hiperparâmetros. Relatar a limitação de qualquer escolha de família feita com base no teste; nesse caso, não caracterizar o resultado como avaliação final independente.
- Escrever a interpretação e a discussão depois de apresentar as tabelas calculadas. Atualizar conclusões de acordo com o resultado da execução, explicitando limitações e evitando generalizações além do período observado.

### Projeções e visualização

- Após selecionar o modelo de cada fluxo, reajustá-lo usando todo o histórico disponível, com os parâmetros escolhidos sem consultar o teste, e gerar uma projeção de 12 meses.
- Construir uma tabela de projeção com admissões e desligamentos previstos por seus respectivos modelos e calcular o saldo pela diferença entre essas colunas.
- Integrar as projeções à parte de resultados do notebook central. Não manter a dissertação extensa da etapa de projeção da Rev1; conservar apenas o contexto e as ressalvas necessárias para interpretar as previsões.
- Criar ou adaptar uma função de visualização em `projeto/ast_caged/modules/visualization/plotters.py`, seguindo o padrão existente. O gráfico deve distinguir claramente o histórico observado da previsão e indicar a fronteira temporal entre ambos.
- No notebook, iterar pelas séries de `time_series_uf` e chamar a função de gráfico com o histórico e a projeção correspondentes. Não imprimir a série sem visualização nem portar a GUI de widgets da Rev1, salvo se ela vier a ser requisito explícito.
- Explicitar que a cobertura é agregada por UF. Se as bases não tiverem setor/CNAE, não apresentar seleção setorial como implementada.

## Etapas de implementação

1. **Inspecionar interfaces existentes**: confirmar o formato de `time_series_uf`, o índice temporal, as colunas, o carregamento atual e os padrões dos módulos de inferência e visualização.
2. **Implementar e testar o split**: criar `datasplit.py`, validar limites de tamanho e datas, armazenar recortes e metadados por série e gerar a tabela consolidada.
3. **Implementar previsões**: modularizar a baseline sazonal, SARIMAX e Prophet em `prediction.py`, com validação temporal, armazenamento de parâmetros e tratamento explícito de falhas.
4. **Integrar dependências**: atualizar `projeto/requirements.txt` somente com dependências efetivamente usadas e remover instalação inline do notebook.
5. **Integrar o notebook**: substituir o texto de proposta pelo fluxo de modelagem executável; apresentar splits, previsões e parâmetros a partir dos dados armazenados; remover conteúdo duplicado ou automático da Rev1.
6. **Produzir avaliação e projeções**: gerar tabelas de métricas por modelo, UF e fluxo; selecionar os modelos de cada fluxo conforme o critério documentado; reajustar no histórico completo; derivar o saldo.
7. **Integrar visualização e discussão**: apresentar histórico versus previsão para cada série e redigir conclusões após verificar a saída produzida.
8. **Validar ponta a ponta**: executar testes dos módulos e as células relevantes do notebook em ordem, conferir os resultados, as tabelas e os gráficos, e corrigir a documentação de acordo com a execução.

## Critérios de aceite

- A carga de dados continua centralizada no pipeline existente; não há uma segunda leitura dependente de Drive nem busca de `.gsheet`.
- O notebook não instala pacotes durante a execução e as dependências necessárias estão declaradas no manifesto do projeto.
- Todas as séries recebem recortes cronológicos coerentes, com validação e teste de 12 meses quando há histórico suficiente; séries curtas falham com mensagem clara.
- A tabela de períodos é gerada a partir dos recortes efetivamente usados e identifica cada série pelo nome da chave correspondente.
- Baseline, SARIMAX e Prophet produzem previsões individuais para admissões e desligamentos; parâmetros e modelos vencedores são calculados, armazenados e exibidos.
- Nenhum hiperparâmetro é selecionado com base no teste. O método de seleção e qualquer limitação da avaliação estão documentados.
- Métricas e tabelas são geradas durante a execução. O saldo é derivado dos fluxos previstos, não ajustado como variável independente.
- A projeção cobre os 12 meses seguintes à última observação disponível, e os gráficos distinguem valores reais de valores previstos.
- A documentação não afirma cobertura por setor/CNAE quando as bases disponíveis são agregadas por UF.
- Testes direcionados e execução das células integradas verificam formatos, alinhamento temporal, resultados e ausência de regressões no fluxo exploratório existente.

## Decisões fixadas na implementação

- `temporal_split` retorna um objeto com os recortes cronológicos e a tabela de período de uma linha; o notebook guarda esse objeto e o resumo dentro do registro de cada UF.
- O critério de seleção é o menor RMSE na validação, aplicado individualmente por UF e por fluxo. O teste de 12 meses permanece fora da seleção e é reportado como avaliação posterior.
- A grade Prophet é limitada a ordens Fourier anuais 3, 5 e 10 e valores de `changepoint_prior_scale` 0,01, 0,05 e 0,1; ambos são selecionados pela validação.
- As dependências dos modelos e do notebook são declaradas em `projeto/requirements.txt`; a integração foi verificada no ambiente local com Prophet 1.5.0.
- Foram adicionados testes automatizados para splits, métricas, baseline, alinhamento temporal do SARIMAX e visualização. As células integradas foram executadas com os quatro arquivos Parquet locais.
