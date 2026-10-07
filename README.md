# Buscador de jogos indie a partir das reviews dos jogadores

Projeto 1 da disciplina de Recuperação de Informação. Construímos um buscador sobre reviews da Steam em português. Ele tem um pipeline de pré-processamento voltado para texto informal, um índice invertido implementado à mão e três modelos de recuperação: booleano, TF-IDF e BM25. Os três foram comparados sobre as mesmas 20 consultas, com julgamento manual e as métricas Precision, Recall, F1 e nDCG.

Resumo dos resultados:

| Modelo | P@10 | R@10 | F1@10 | nDCG@10 |
|---|---|---|---|---|
| Booleano (conjunto inteiro) | 0,129* | 0,955 | 0,189 | — |
| TF-IDF + cosseno | 0,780 | 0,367 | 0,476 | 0,659 |
| BM25 (k1 = 1,2; b = 0,75) | 0,805 | 0,384 | 0,497 | 0,717 |

\* A precisão do booleano ficou subestimada por causa da forma como a avaliação foi feita (ver seção 5.4).

O BM25 teve o melhor desempenho. Em média, 8 dos 10 primeiros jogos retornados atendem à consulta, e os mais relevantes costumam aparecer nas primeiras posições.

## Como rodar

1. Coloque a pasta `Indie` do SteamBR no Google Drive. No nosso caso ela ficou em `MyDrive/dados/Indie`.
2. Abra `buscador_steam_indie.ipynb` no Colab e ajuste `PASTA_CORPUS` e `PASTA_SAIDA` na seção 0, se necessário.
3. Execute as células na ordem.

O índice (`indice.pkl`), o pool de julgamento, os julgamentos e os resultados são salvos em `PASTA_SAIDA`. Se o pré-processamento for alterado, é preciso apagar o `indice.pkl` para que o índice seja reconstruído.

## 1. Problema, domínio e corpus

### Problema
A busca da Steam funciona basicamente por nome e por tags. Uma consulta como "jogo relaxante sem combate" ou "roda em pc fraco" não funciona bem lá, porque essas informações aparecem no que os jogadores escrevem sobre o jogo, e não no título. A ideia do projeto foi indexar as reviews da comunidade brasileira para permitir esse tipo de busca.

### Corpus
Usamos o SteamBR (Jorge & Pardo, 2023, NILC/ICMC-USP), um corpus de reviews da Steam em português brasileiro em que o nome e o gênero dos jogos foram anotados manualmente. Trabalhamos com a pasta Indie, que tem 3.044 arquivos (um por jogo) no formato da API da Steam, com texto, recomendação, utilidade e autor de cada review. Escolhemos esse gênero por ser o que tem mais jogos no SteamBR e por reunir jogos muito variados (terror, puzzle, plataforma, simulação, narrativos), o que permite consultas de tipos diferentes. Também tem um tamanho que dá para processar no Colab e julgar manualmente.

### Unidade de documento
Como o objetivo é encontrar jogos, e não reviews soltas, cada documento corresponde a um jogo: o nome mais até 50 reviews. As reviews são ordenadas pela utilidade (`weighted_vote_score`), e as cópias idênticas são descartadas. Encontramos várias reviews copiadas e coladas por autores diferentes. O nome do jogo também entra no texto indexado, para que a busca pelo título continue funcionando.

- **Limite de 50 reviews:** sem ele, jogos muito populares virariam documentos enormes e tenderiam a dominar o ranking.
- **Mínimo de 3 reviews:** jogos com menos que isso foram descartados por terem pouco texto. Foram 69 descartados, e ficaram 2.975 documentos indexados.

### Estrutura usuário-item (Projeto 2)
O domínio tem uma estrutura usuário-item bem clara: cada review liga um usuário (`author.steamid`) a um jogo, com um feedback explícito (`voted_up`, se recomenda ou não) e um implícito (`playtime_forever`, minutos jogados).

| Medida | Valor |
|---|---|
| Interações (usuário, jogo) | 476.285 |
| Usuários | 284.828 |
| Jogos | 3.044 |
| Usuários com 2 ou mais jogos avaliados | 82.883 (29%) |
| Recomendações positivas | 93,6% |
| Esparsidade da matriz | 99,95% |

As interações foram exportadas para `interacoes_usuario_item.csv`, incluindo as dos jogos que ficaram fora do índice.

Dois pontos que vão pesar no Projeto 2:
- **Esparsidade:** a matriz é muito esparsa, já que 71% dos usuários avaliaram um único jogo.
- **Viés positivo:** com 93,6% de recomendações positivas, o feedback explícito é bem desbalanceado, e o tempo de jogo pode servir como sinal complementar.

## 2. Pré-processamento

Cada técnica foi implementada como uma função separada no notebook (seção 2) e aplicada na ordem abaixo. A mesma função é usada para documentos e para consultas, para que os termos da consulta casem com os do índice.

| # | Técnica | O que fizemos | Por quê |
|---|---|---|---|
| 1 | Limpeza de marcação | Remoção de URLs, BBCode (`[b]`, `[spoiler]`) e cabeçalhos `<...>`. Em checklists, descartamos as linhas não marcadas `( )` e mantivemos as marcadas `(x)` | Algumas reviews seguem um formulário do tipo `( ) Multiplayer` / `(x) Jogo Offline`. Se tudo fosse indexado, um jogo offline apareceria na busca por "multiplayer" |
| 2 | Case folding e acentos | Texto em minúsculas e sem acento, antes do stemming | Muita gente escreve "nao" e "historia". Testando as duas ordens, vimos que o RSLP gera radicais diferentes para *história* e *historia* (`histor` e `hist`), e o mesmo acontece com *jogável/jogavel* e *terrível/terrivel*. Tirando o acento antes, os pares passam a coincidir |
| 3 | Remoção de ruído | Remoção de risadas (`kkkk`, `kkskskss`, `hahaha`, `rsrs`) e redução de letras repetidas (`bastanteeee` → `bastante`) | Aparecem com muita frequência e não ajudam na busca |
| 4 | Tokenização | Sequências de letras e números, mantendo os números | Termos como "2d" e "60 fps" fazem sentido em jogos. O token "10" aparece em 67% dos jogos, por causa das notas "10/10" |
| 5 | Expansão de abreviações | `n`→`nao`, `mt`→`muito`, `vc`→`voce`, `pq`→`porque`, `pra`→`para`, entre outras | O "n" é bastante usado como negação ("eu n enjoei"). Sem a expansão ele seria descartado junto com os outros tokens de uma letra |
| 6 | Marcação de negação | A negação é juntada ao próximo termo de conteúdo, pulando stopwords e alguns verbos auxiliares (vou, vai, ter...): "não tem combate" → `nao_combate` | Como as reviews são opiniões, "recomendo" e "não recomendo" precisam ser termos diferentes. Os auxiliares são pulados porque, sem isso, "não vou mentir" viraria `nao_vou`, e a negação ficaria presa num termo sem conteúdo |
| 7 | Stopwords | Lista do NLTK para português, sem acento, retirando dela `nao`, `nem`, `sem`, `nunca` e `jamais` | A lista original remove as negações, que são muito frequentes no corpus: "não" aparece 48.563 vezes, "sem" 7.965 e "nem" 4.020 |
| 8 | Stemming | RSLP (NLTK), aplicado também à palavra negada (`nao_gostei` → `nao_gost`) | Reduziu o vocabulário em 47% (de 40.835 palavras para 21.839 radicais, medido em 1.000 jogos) |

Exemplo de uma frase passando pelo pipeline (célula de demonstração da seção 2):
```
entrada : Não vou mentir, eu n enjoei!! Jogo bassstanteeeee relaxante kkkkk, [b]sem combate[/b] e não é difícil.
saída   : ['nao_ment', 'nao_enjo', 'jog', 'bast', 'relax', 'sem_combat', 'nao_dificil']
```

### 2.1 Stopwords de domínio
Na análise de frequência (seção 2b), alguns termos apareceram em quase todos os jogos: "jogo" (99%), "bem" (91%), "bom" (90%), "game" (82%), "recomendo" (81%), "jogar" (77%), "tempo" (76%) e "vale" (69%).

Consideramos criar uma lista de stopwords de domínio com esses termos, mas decidimos não fazer isso. O primeiro motivo é que o IDF já reduz bastante o peso deles. "Jogo" aparece em cerca de 2.945 dos 2.975 jogos, o que dá um IDF de aproximadamente 0,01, enquanto um termo como "relaxante" tem IDF em torno de 1,50. O segundo é que alguns desses termos carregam opinião e aparecem nas consultas, como em "não vale a pena muito caro" (q19), em que `nao_val` é justamente um termo importante.

### 2.2 Stemming ou lematização
Optamos pelo RSLP, que é um stemmer feito para o português. A lematização com spaCy seria mais lenta para os cerca de 2,2 milhões de tokens do corpus e tende a errar mais em texto informal, cheio de gírias, erros de digitação e inglês misturado.

Na maior parte dos casos o agrupamento ficou bom: `jog` reúne jogo, jogar, jogos, joguei e jogador, e `compr` reúne comprar, comprei e compra. Também encontramos alguns casos de super-stemming:
- `lev` junta "leve" e "levar", o que afeta a consulta q13 ("roda em pc fraco"), cuja versão booleana usa "leve";
- `dev` junta "deve", "devido" e "devs" (desenvolvedores);
- `tir` junta "tiro" e "tirar";
- `cont` junta "conta", "contar" e "conto".

### 2.3 Termos em inglês
Mantivemos os termos em inglês e não aplicamos stopwords em inglês. Palavras como "gameplay" (4.082 ocorrências), "soundtrack" (523) e "coop" (475) são usadas normalmente pelos jogadores brasileiros. O "the" (11.818 ocorrências) também ficou, mas ele tem IDF baixo, assim como "jogo".

## 3. Índice invertido

O índice tem 44.463 termos e 1.059.043 postings. Cada termo aponta para uma lista de pares `(documento, tf)`, ordenada pelo número do documento. Além disso, guardamos o tamanho de cada documento (média de 751 tokens), usado pelo BM25, e a norma TF-IDF de cada documento, calculada uma vez na construção e usada no cosseno. Não guardamos posições, porque o sistema não faz busca por frase exata. O índice é salvo em `indice.pkl` para evitar reprocessar o corpus a cada execução.

Exemplo: o termo `relax` aparece em 663 jogos, e as primeiras postings são `[(8, 2), (14, 1), (18, 2), (19, 2), (32, 4)]`.

Mantivemos as postings ordenadas porque o AND do modelo booleano é feito por merge de duas listas ordenadas, percorrendo cada uma uma única vez, em O(n + m). Como os documentos são indexados em ordem crescente, as listas já ficam ordenadas na construção, sem custo extra.

## 4. Modelos de recuperação

Os três modelos usam o mesmo índice e o mesmo pré-processamento.

| Modelo | Consulta | Pontuação | Saída |
|---|---|---|---|
| Booleano | Listas `todos` (AND), `algum` (OR) e `nenhum` (NOT) | Não há: o documento casa ou não com a consulta | Conjunto sem ordem |
| TF-IDF | Texto livre | Peso `(1 + log tf) · log(N/df)` na consulta e no documento; score pelo cosseno | Ranking |
| BM25 | Texto livre | `Σ idf(t) · tf·(k1+1) / (tf + k1·(1 − b + b·|d|/|d|médio))`, com `idf = ln(1 + (N−df+0,5)/(df+0,5))`, `k1 = 1,2` e `b = 0,75` | Ranking |

- **TF-IDF:** usamos tf sublinear (`1 + log tf`), para que a décima ocorrência de um termo não valha dez vezes a primeira. O cosseno normaliza pelo tamanho do vetor do documento.
- **BM25:** o `k1` controla a saturação do tf, e o `b` controla quanto documentos longos são penalizados. Usamos os valores padrão da literatura, 1,2 e 0,75.
- **BM25 (variante do IDF):** usamos `ln(1 + (N−df+0,5)/(df+0,5))` em vez da fórmula original de Robertson, `ln((N−df+0,5)/(df+0,5))`, porque a original fica negativa para termos presentes em mais da metade dos documentos. No nosso corpus isso aconteceria com termos como "jogo" e "bom", que passariam a tirar pontos dos documentos em vez de somar.
- **Booleano:** quando um termo tem mais de uma palavra ("sem bugs", "pixel art"), todas precisam aparecer no documento (AND dos tokens), já que o índice não guarda posições.
- **Formato da consulta booleana:** optamos por três listas (`todos`, `algum`, `nenhum`) em vez de um parser de expressões com parênteses. Esse formato cobre AND, OR e NOT, é simples de escrever e de explicar, e foi suficiente para as 20 consultas. A limitação é que só existe um grupo OR por consulta, então algumas consultas tiveram que ser simplificadas. Na q15, por exemplo, a versão booleana busca só "trilha sonora", "soundtrack" ou "ost", sem exigir um adjetivo como "incrível".

Escolhemos implementar dois modelos de ranking, e não apenas um, para ter uma comparação entre o modelo vetorial clássico e o probabilístico sobre os mesmos dados. Não usamos modelos densos (embeddings) porque a proposta foi implementar tudo à mão, e os modelos esparsos já permitem explicar exatamente por que cada jogo foi retornado.

## 5. Avaliação experimental

### 5.1 Metodologia
Definimos 20 consultas (seção 5a do notebook), divididas em cinco tipos:
- característica: q01, q04, q06, q09, q13, q15, q17, q18;
- gênero: q03, q07, q08, q10, q14, q16;
- negação: q02, q11;
- crítica: q12, q19;
- termo raro: q20.

Cada consulta tem uma versão em texto, para TF-IDF e BM25, uma versão booleana equivalente e um critério de relevância. Tanto as consultas quanto os critérios foram escritos antes de rodar os modelos.

Para montar o conjunto de julgamento, usamos pooling. Para cada consulta, juntamos o top-10 do TF-IDF, o top-10 do BM25 e até 30 jogos sorteados do resultado booleano. Isso deu 861 pares consulta-jogo, envolvendo 715 jogos diferentes. Avaliamos o top-10 porque corresponde a uma primeira página de resultados, que é o que o usuário de fato olha. No booleano, que chega a retornar mais de mil jogos, julgar tudo seria inviável, então sorteamos até 30 jogos por consulta, número que mantém o trabalho de julgamento possível e ainda dá uma amostra do conjunto. O arquivo foi embaralhado e não indica de qual modelo veio cada jogo, para que o julgamento não fosse influenciado. Para julgar, geramos um CSV auxiliar com trechos das reviews que mencionam termos da consulta e outro com todas as reviews dos jogos do pool.

A escala usada foi 0 (irrelevante), 1 (relevante) e 2 (muito relevante). Usamos três níveis, em vez de só relevante e irrelevante, para que o nDCG pudesse diferenciar um jogo que atende plenamente à consulta de um que atende só em parte. Para P, R e F1, consideramos relevantes os jogos com nota maior ou igual a 1; o nDCG usa a nota graduada.

### 5.2 Métricas
- **P@10:** proporção dos 10 primeiros jogos que atendem à consulta.
- **R@10:** proporção dos jogos relevantes conhecidos que aparecem no top-10.
- **F1@10:** média harmônica entre P e R.
- **nDCG@10:** mede se os jogos mais relevantes (nota 2) estão nas primeiras posições.

Para o booleano, que não ordena os resultados, P, R e F1 foram calculados sobre o conjunto inteiro retornado.

### 5.3 Resultados por consulta (F1)

| Consulta | Tipo | Booleano | TF-IDF | BM25 | Melhor |
|---|---|---|---|---|---|
| q01 relaxante para desestressar | característica | 0,03 | 0,61 | 0,52 | TF-IDF |
| q02 relaxante sem combate | negação | 0,06 | 0,48 | 0,61 | BM25 |
| q03 terror psicológico | gênero | 0,17 | 0,58 | 0,65 | BM25 |
| q04 história que faz chorar | característica | 0,17 | 0,43 | 0,43 | empate |
| q05 cooperativo com amigos | característica | 0,13 | 0,56 | 0,64 | BM25 |
| q06 difícil e desafiador | característica | 0,05 | 0,51 | 0,46 | TF-IDF |
| q07 roguelike | gênero | 0,39 | 0,42 | 0,42 | empate |
| q08 metroidvania | gênero | 0,60 | 0,43 | 0,43 | Booleano |
| q09 pixel art bonita | característica | 0,18 | 0,44 | 0,31 | TF-IDF |
| q10 visual novel com finais | gênero | 0,06 | 0,59 | 0,59 | empate |
| q11 sem bugs, otimizado | negação | 0,03 | 0,50 | 0,50 | empate |
| q12 cheio de bugs que trava | crítica | 0,07 | 0,41 | 0,28 | TF-IDF |
| q13 roda em pc fraco | característica | 0,02 | 0,40 | 0,56 | BM25 |
| q14 plataforma 2d | gênero | 0,13 | 0,57 | 0,64 | BM25 |
| q15 trilha sonora incrível | característica | 0,05 | 0,33 | 0,29 | TF-IDF |
| q16 simulador de fazenda | gênero | 0,75 | 0,45 | 0,45 | Booleano |
| q17 jogo curto | característica | 0,02 | 0,42 | 0,50 | BM25 |
| q18 puzzle que faz pensar | característica | 0,06 | 0,43 | 0,48 | BM25 |
| q19 não vale a pena, caro | crítica | 0,13 | 0,38 | 0,54 | BM25 |
| q20 deckbuilder | termo raro | 0,69 | 0,58 | 0,67 | Booleano |

Entre os dois modelos de ranking, o BM25 foi melhor em 9 consultas, o TF-IDF em 5, e houve 6 empates. O booleano teve o maior F1 em 3 consultas.

### 5.4 Discussão

**BM25 e TF-IDF.** O BM25 ficou à frente em P@10 (0,805 contra 0,780) e, com uma diferença maior, em nDCG@10 (0,717 contra 0,659). Ou seja, ele coloca com mais frequência os jogos muito relevantes nas primeiras posições. Acreditamos que isso tenha a ver com o tamanho dos documentos, que varia bastante (de 3 a 50 reviews, com média de 751 tokens). No BM25, o efeito do tamanho é controlado pelo `b`, e o tf satura por causa do `k1`. No cosseno, a norma do documento cresce com o número de termos distintos, e por isso jogos com muitas reviews, que costumam ter um vocabulário mais amplo, acabam penalizados de forma mais forte.

O TF-IDF foi melhor em q01, q06, q09, q12 e q15. São consultas sobre características comuns a muitos jogos (dificuldade, arte, música, bugs), em que nenhum termo se destaca muito, e nesses casos a normalização do cosseno parece ter favorecido documentos mais focados.

Os dois modelos de ranking retornam boa parte dos mesmos jogos no topo: 7 dos 10 primeiros em q01 e 6 dos 10 em q03. A diferença maior está na ordem, como se vê nas cinco primeiras posições de q03:

| Posição | q03 TF-IDF | q03 BM25 | q03 Booleano (233 jogos, sem ordem) |
|---|---|---|---|
| 1 | TheLandofPain | MissingChildren | ALongWayDown |
| 2 | ThePaperman | TheBoogieMan | ARobotNamedFight |
| 3 | MissingChildren | ThePaperman | Animality |
| 4 | DereEvilExe | IMSCARED | Ann |
| 5 | Horse | Horse | AnotherLostPhone |

No booleano, a "posição" é só a ordem do identificador do documento (que segue a ordem alfabética dos arquivos), já que o modelo não atribui pontuação. A comparação completa dos três modelos para q01, q02 e q03 está na seção 5b do notebook.

Os valores de score dos dois modelos não podem ser comparados diretamente: o cosseno varia entre 0 e 1, e o BM25 não tem limite superior.

**Modelo booleano.** O booleano foi melhor em q08 (metroidvania), q16 (simulador de fazenda) e q20 (deckbuilder). Nessas consultas, um termo específico e pouco frequente já define bem o que se procura, então o conjunto retornado é pequeno e quase todo relevante.

Nas consultas mais vagas acontece o contrário: ele retornou 1.023 jogos em q01, 768 em q02 e 233 em q03, sem nenhuma ordem. O recall fica alto (0,955), mas o usuário não tem como saber quais desses jogos são os melhores, e uma lista de mil jogos não é um resultado útil na prática.

A precisão de 0,129 também está subestimada. De cerca de mil jogos retornados em q01, por exemplo, só 30 foram sorteados para julgamento, e os não julgados contam como irrelevantes no cálculo. A precisão real deve ser maior, mas isso não muda o principal problema do modelo, que é a falta de ordenação.

**Negação.** Nas consultas com negação ou crítica, o BM25 foi melhor em q02 (0,61 contra 0,48) e q19 (0,54 contra 0,38), e empatou em q11 (0,50). Termos como `sem_combat` e `nao_val` são relativamente raros e por isso recebem IDF alto, que o BM25 aproveita bem. Sem a marcação de negação, "sem combate" viraria só "combate", e a consulta acabaria retornando justamente jogos de combate.

**Recall baixo.** O R@10 em torno de 0,38 tem mais a ver com a métrica do que com os modelos. Quando uma consulta tem mais de 10 jogos relevantes no pool, mesmo um top-10 perfeito não chega a R = 1. Pelos valores médios de precisão e recall, as consultas têm em torno de 20 jogos relevantes, o que limita o R@10 a cerca de 0,5 e também puxa o F1@10 para baixo. Por isso, para avaliar a qualidade do ranking, demos mais peso a P@10 e nDCG@10.

### 5.5 Sensibilidade do BM25 ao parâmetro b

Como o `b` controla a penalização de documentos longos, e os nossos documentos variam muito de tamanho, testamos alguns valores mantendo k1 = 1,2 (última célula do notebook):

| b | P@10 | R@10 | F1@10 | nDCG@10 |
|---|---|---|---|---|
| 0,0 (sem normalização) | 0,325 | 0,145 | 0,195 | 0,290 |
| 0,25 | 0,465 | 0,211 | 0,282 | 0,449 |
| 0,5 | 0,630 | 0,299 | 0,387 | 0,608 |
| 0,75 (padrão) | 0,805 | 0,384 | 0,497 | 0,717 |
| 1,0 (normalização total) | 0,700 | 0,340 | 0,436 | 0,641 |

O resultado é bastante sensível ao `b`:
- **Valores baixos:** sem normalização (b = 0), os jogos com mais texto, em geral os populares com 50 reviews, sobem no ranking só por terem mais ocorrências dos termos.
- **Normalização total (b = 1):** os documentos longos passam a ser penalizados demais.
- **Valor padrão:** o 0,75, que já tínhamos escolhido antes da avaliação, foi o melhor entre os testados.

Esses números precisam ser lidos com cuidado. Com outros valores de `b`, aparecem no top-10 jogos que não estavam no pool e que, portanto, contam como irrelevantes. Como o pool foi montado a partir do BM25 padrão e do TF-IDF, ele favorece essas configurações. Parte da queda pode vir desse viés, e para confirmar seria necessário julgar os jogos novos.

## 6. Limitações

No pré-processamento:
- **Escopo da negação:** a negação cobre só o termo seguinte. Em "não recomendo para quem quer algo relaxante", apenas "recomendo" é negado, e "relaxante" continua contando a favor.
- **Ironia:** não é tratada. Há no corpus reviews como "Muito ruim, não gostei" marcadas como recomendação positiva.
- **Super-stemming do RSLP:** alguns radicais juntam palavras sem relação, como `leve`/`levar`, `tiro`/`tirar` e `devs`/`deve`.
- **Termos em inglês:** o RSLP foi feito para o português, então os termos em inglês não recebem um stemming adequado.

Na avaliação:
- **Recall de pool:** o recall é calculado em relação aos relevantes encontrados no pool, e não a todos os relevantes da coleção, o que exigiria julgar os 2.975 jogos. O valor real deve ser menor.
- **Viés do pool:** configurações que trazem jogos de fora do pool são prejudicadas, como discutido na seção 5.5.
- **Precisão do booleano:** fica subestimada por conta da amostra de 30 jogos por consulta.
- **Um único avaliador:** os julgamentos foram feitos por uma só pessoa, então não há medida de concordância entre avaliadores.
- **Número de consultas:** com 20 consultas não é possível fazer testes de significância estatística, e diferenças pequenas entre os modelos devem ser vistas com cautela.

## Referência
Jorge, G. A. Z.; Pardo, T. A. S. *SteamBR: a dataset for game reviews and evaluation of a state-of-the-art method for helpfulness prediction*. BraSNAM/SBC, 2023. https://github.com/germanojorge/SteamBR
